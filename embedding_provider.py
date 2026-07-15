"""Embedding provider abstractions for optional semantic retrieval.

Providers are deliberately independent from the engine.  In particular, model
downloads and dimension discovery happen only through the explicit warmup
command; resolving a provider never performs network or disk-heavy work.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .config import LCMConfig
from .tokens import count_tokens

logger = logging.getLogger(__name__)

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_VOYAGE_MAX_BATCH_TOKENS = 80_000
_VOYAGE_MAX_DOCUMENT_TOKENS = 27_000
_MAX_ATTEMPTS = 3
_RETRY_AFTER_BUDGET_S = 60.0
_DEFAULT_FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"


class EmbeddingProvider(Protocol):
    """Minimal provider contract consumed by later embedding workers."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingProviderError(RuntimeError):
    """Base class for operator-readable provider failures."""


class ProviderUnavailable(EmbeddingProviderError):
    """The configured provider or its optional dependency is unavailable."""


class ProviderNotWarmedUp(EmbeddingProviderError):
    """A local embedding model has not been explicitly downloaded yet."""


class ProviderCircuitOpen(EmbeddingProviderError):
    """Calls are temporarily blocked after repeated provider failures."""


class ProviderRateLimited(EmbeddingProviderError):
    """The local embedding call-rate guard is temporarily open."""


class VoyageError(EmbeddingProviderError):
    """A classified Voyage API failure."""

    def __init__(self, kind: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


HttpTransport = Callable[..., HttpResponse]


def _default_http_transport(
    *,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    """POST JSON using urllib while preserving HTTP error response bodies."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=int(response.status),
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers else {},
            body=exc.read(),
        )


@dataclass
class EmbeddingCircuitBreaker:
    """Small process-local circuit breaker, separate from summarization."""

    failure_threshold: int = 2
    cooldown_seconds: float = 300.0
    _failures: int = 0
    _open_until: float = 0.0

    def allows(self, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if self._open_until and current >= self._open_until:
            self._open_until = 0.0
            self._failures = 0
        return current >= self._open_until

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self, *, now: float | None = None) -> None:
        self._failures += 1
        if self._failures >= max(1, int(self.failure_threshold or 1)):
            current = time.monotonic() if now is None else now
            self._open_until = current + max(0.0, float(self.cooldown_seconds or 0.0))


@dataclass
class EmbeddingSpendGuard:
    """Sliding-window call guard that bounds accidental embedding spend."""

    max_calls: int = 60
    window_seconds: float = 60.0
    backoff_seconds: float = 60.0
    _calls: list[float] = field(default_factory=list)
    _backoff_until: float = 0.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._calls = [called_at for called_at in self._calls if called_at >= cutoff]

    def allows(self, *, now: float | None = None) -> bool:
        if self.max_calls <= 0:
            return True
        current = time.monotonic() if now is None else now
        if current < self._backoff_until:
            return False
        self._prune(current)
        return len(self._calls) < self.max_calls

    def record_call(self, *, now: float | None = None) -> None:
        if self.max_calls <= 0:
            return
        current = time.monotonic() if now is None else now
        self._prune(current)
        self._calls.append(current)
        if len(self._calls) >= self.max_calls:
            self._calls.clear()
            self._backoff_until = current + max(0.0, self.backoff_seconds)

    def clear(self) -> None:
        self._calls.clear()
        self._backoff_until = 0.0


class _ResilientProvider:
    provider_id = "unknown"

    def __init__(
        self,
        *,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        self.breaker = breaker or EmbeddingCircuitBreaker()
        self.spend_guard = spend_guard or EmbeddingSpendGuard()

    def _guarded(self, call: Callable[[], Any]) -> Any:
        if not self.breaker.allows():
            raise ProviderCircuitOpen(
                f"{self.provider_id} embedding circuit is cooling down"
            )
        if not self.spend_guard.allows():
            raise ProviderRateLimited(
                f"{self.provider_id} embedding call-rate guard is cooling down"
            )
        self.spend_guard.record_call()
        try:
            result = call()
        except Exception:
            self.breaker.record_failure()
            raise
        self.breaker.record_success()
        return result


def _response_json(response: HttpResponse, *, provider: str) -> dict[str, Any]:
    try:
        decoded = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingProviderError(
            f"{provider} returned an invalid JSON response"
        ) from exc
    if not isinstance(decoded, dict):
        raise EmbeddingProviderError(f"{provider} returned an invalid response shape")
    return decoded


def _coerce_vectors(raw: Any, *, provider: str) -> list[list[float]]:
    if not isinstance(raw, list):
        raise EmbeddingProviderError(f"{provider} response did not contain embeddings")
    vectors: list[list[float]] = []
    for vector in raw:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)):
            raise EmbeddingProviderError(f"{provider} returned an invalid embedding")
        try:
            vectors.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"{provider} returned a non-numeric embedding"
            ) from exc
    return vectors


def _scrub_response_body(body: bytes) -> str:
    """Keep useful API errors while removing fields likely to echo inputs."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "<unparseable response body suppressed>"

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if any(
                        marker in str(key).strip().lower()
                        for marker in ("input", "texts", "documents")
                    )
                    else scrub(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return json.dumps(scrub(payload), ensure_ascii=True, sort_keys=True)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return None


def _retry_after_seconds(value: str | None, *, now: float | None = None) -> float:
    if not value:
        return 0.0
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            current = time.time() if now is None else now
            return max(0.0, parsed.timestamp() - current)
        except (TypeError, ValueError, OverflowError):
            return 0.0


class VoyageProvider(_ResilientProvider):
    provider_id = "voyage"

    def __init__(
        self,
        model: str,
        *,
        transport: HttpTransport | None = None,
        timeout: float = 3.0,
        sleeper: Callable[[float], None] = time.sleep,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("Voyage embedding model must not be empty")
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0
        self.last_skipped_documents: list[int] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _error(self, response: HttpResponse) -> VoyageError:
        status = response.status
        if status in {401, 403}:
            kind = "auth"
        elif status == 429:
            kind = "rate_limit"
        elif 400 <= status < 500:
            kind = "bad_request"
        else:
            kind = "server_error"
        scrubbed = _scrub_response_body(response.body)
        logger.warning("Voyage embedding error status=%s body=%s", status, scrubbed)
        return VoyageError(kind, f"Voyage embedding request failed ({status})", status_code=status)

    def _request(self, texts: Sequence[str], *, input_type: str) -> list[list[float]]:
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise VoyageError("auth", "VOYAGE_API_KEY is not set")
        payload = {
            "model": self.model_id,
            "input": list(texts),
            "input_type": input_type,
            "truncation": False,
        }
        retry_after_spent = 0.0
        last_error: VoyageError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._transport(
                    url=_VOYAGE_URL,
                    payload=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                last_error = VoyageError("network", f"Voyage network error: {exc}")
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise last_error from exc
                self._sleep(min(25.0, 0.5 * (2 ** attempt)))
                continue

            if 200 <= response.status < 300:
                data = _response_json(response, provider="Voyage")
                rows = data.get("data")
                if not isinstance(rows, list):
                    raise EmbeddingProviderError("Voyage response did not contain embedding data")
                ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
                vectors = _coerce_vectors(
                    [row.get("embedding") for row in ordered if isinstance(row, dict)],
                    provider="Voyage",
                )
                if len(vectors) != len(texts):
                    raise EmbeddingProviderError(
                        "Voyage returned a different number of embeddings than requested"
                    )
                return vectors

            last_error = self._error(response)
            retryable = response.status == 429 or response.status >= 500
            if not retryable or attempt + 1 >= _MAX_ATTEMPTS:
                raise last_error
            if response.status == 429:
                delay = _retry_after_seconds(_header(response.headers, "Retry-After"))
                remaining = _RETRY_AFTER_BUDGET_S - retry_after_spent
                if delay > remaining:
                    raise last_error
                retry_after_spent += delay
            else:
                delay = min(25.0, 0.5 * (2 ** attempt))
            self._sleep(delay)
        raise last_error or VoyageError("network", "Voyage request failed")

    def _remember_dim(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Voyage returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Voyage embedding dimension changed within the process")
            self._dim = len(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        accepted: list[tuple[str, int]] = []
        self.last_skipped_documents = []
        for index, text in enumerate(texts):
            text = str(text)
            tokens = count_tokens(text)
            if tokens > _VOYAGE_MAX_DOCUMENT_TOKENS:
                self.last_skipped_documents.append(index)
                logger.warning(
                    "Skipping Voyage document index=%d tokens=%d limit=%d",
                    index,
                    tokens,
                    _VOYAGE_MAX_DOCUMENT_TOKENS,
                )
                continue
            accepted.append((text, tokens))

        vectors: list[list[float]] = []
        batch: list[str] = []
        batch_tokens = 0
        for text, tokens in accepted:
            if batch and batch_tokens + tokens > _VOYAGE_MAX_BATCH_TOKENS:
                vectors.extend(
                    self._guarded(lambda current=tuple(batch): self._request(current, input_type="document"))
                )
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += tokens
        if batch:
            vectors.extend(
                self._guarded(lambda current=tuple(batch): self._request(current, input_type="document"))
            )
        self._remember_dim(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._guarded(
            lambda: self._request((str(text),), input_type="query")
        )
        self._remember_dim(vectors)
        return vectors[0]


class OllamaProvider(_ResilientProvider):
    provider_id = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        transport: HttpTransport | None = None,
        timeout: float = 3.0,
        sleeper: Callable[[float], None] = time.sleep,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("Ollama embedding model must not be empty")
        self.base_url = str(base_url).strip().rstrip("/") or "http://localhost:11434"
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _request(self, texts: Sequence[str]) -> list[list[float]]:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._transport(
                    url=f"{self.base_url}/api/embed",
                    payload={"model": self.model_id, "input": list(texts)},
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise EmbeddingProviderError(f"Ollama network error: {exc}") from exc
                self._sleep(min(25.0, 0.5 * (2 ** attempt)))
                continue
            if not 200 <= response.status < 300:
                raise EmbeddingProviderError(
                    f"Ollama embedding request failed ({response.status})"
                )
            payload = _response_json(response, provider="Ollama")
            raw = payload.get("embeddings")
            if raw is None and "embedding" in payload:
                raw = [payload["embedding"]]
            vectors = _coerce_vectors(raw, provider="Ollama")
            if len(vectors) != len(texts):
                raise EmbeddingProviderError(
                    "Ollama returned a different number of embeddings than requested"
                )
            return vectors
        raise EmbeddingProviderError("Ollama embedding request failed")

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._guarded(lambda: self._request(tuple(str(text) for text in texts)))
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Ollama returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Ollama embedding dimension changed within the process")
            self._dim = len(vector)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed((text,))[0]


def _load_fastembed():
    """Import the optional dependency in one patchable location."""
    from fastembed import TextEmbedding

    return TextEmbedding


class FastembedProvider(_ResilientProvider):
    provider_id = "fastembed"

    def __init__(
        self,
        model: str,
        *,
        cache_dir: str | Path | None = None,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("FastEmbed embedding model must not be empty")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_FASTEMBED_CACHE
        self._model: Any = None
        self._dim = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _construct(self, *, allow_download: bool) -> Any:
        try:
            text_embedding = _load_fastembed()
        except ImportError as exc:
            raise ProviderUnavailable(
                "FastEmbed is not installed; install the optional fastembed dependency"
            ) from exc
        try:
            return text_embedding(
                model_name=self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=not allow_download,
            )
        except Exception as exc:
            if allow_download:
                raise EmbeddingProviderError(
                    f"FastEmbed warmup failed for {self.model_id}: {exc}"
                ) from exc
            raise ProviderNotWarmedUp(
                f"FastEmbed model {self.model_id!r} is not cached locally; "
                "run /lcm embed warmup"
            ) from exc

    def _ensure_local(self) -> Any:
        if self._model is None:
            self._model = self._construct(allow_download=False)
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_local()
        vectors = self._guarded(lambda: _coerce_vectors(list(model.embed(list(texts))), provider="FastEmbed"))
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "FastEmbed returned a different number of embeddings than requested"
            )
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("FastEmbed returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("FastEmbed embedding dimension changed within the process")
            self._dim = len(vector)
        return vectors

    def warmup(self) -> list[float]:
        """The only path allowed to download a missing FastEmbed model."""
        if self._model is None:
            self._model = self._construct(allow_download=True)
        return self.embed_query("warmup")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(tuple(str(text) for text in texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed((str(text),))[0]


def resolve_provider(config: LCMConfig) -> EmbeddingProvider | None:
    """Resolve inert embedding config without making provider calls."""
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    model = str(getattr(config, "embedding_model", "") or "").strip()
    if not provider and not model:
        return None
    if not provider or not model:
        raise ProviderUnavailable(
            "LCM_EMBEDDING_PROVIDER and LCM_EMBEDDING_MODEL must both be set"
        )
    timeout = float(getattr(config, "embedding_query_timeout_s", 3.0))
    if provider in {"voyage", "voyageai"}:
        return VoyageProvider(model, timeout=timeout)
    if provider == "ollama":
        return OllamaProvider(
            model,
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
            timeout=timeout,
        )
    if provider in {"fastembed", "fast-embed"}:
        return FastembedProvider(model)
    raise ProviderUnavailable(
        f"Unsupported embedding provider {provider!r}; use voyage, ollama, or fastembed"
    )


def fastembed_download_size_note(model_id: str) -> str:
    """Return a deliberately approximate, operator-facing download size note."""
    known = {
        "BAAI/bge-small-en-v1.5": "about 130 MB",
        "sentence-transformers/all-MiniLM-L6-v2": "about 90 MB",
    }
    return known.get(model_id, "model-dependent; verify available disk space")
