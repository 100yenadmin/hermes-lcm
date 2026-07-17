"""Embedding provider abstractions for optional semantic retrieval.

Providers are deliberately independent from the engine.  In particular, model
downloads and dimension discovery happen only through the explicit warmup
command; resolving a provider never performs network or disk-heavy work.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .config import LCMConfig
from .tokens import count_tokens

logger = logging.getLogger(__name__)

_VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
_VOYAGE_MAX_BATCH_TOKENS = 80_000
_VOYAGE_MAX_DOCUMENT_TOKENS = 27_000
# Voyage caps a single request at 1000 input items regardless of token count;
# a batch bounded only by tokens can still exceed it (many short documents), so
# an item cap is enforced alongside the token budget.
_VOYAGE_MAX_BATCH_ITEMS = 1000
_VOYAGE_TOKEN_SAFETY_FACTOR = 0.9
_VOYAGE_BATCH_TOKEN_BUDGET = int(
    _VOYAGE_MAX_BATCH_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
_VOYAGE_DOCUMENT_TOKEN_BUDGET = int(
    _VOYAGE_MAX_DOCUMENT_TOKENS * _VOYAGE_TOKEN_SAFETY_FACTOR
)
_MAX_ATTEMPTS = 3
_RETRY_AFTER_BUDGET_S = 60.0
_DEFAULT_FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"
_LOCAL_EMBED_MAX_WORKERS = 4
_LOCAL_EMBED_WORKER_SLOTS = threading.BoundedSemaphore(_LOCAL_EMBED_MAX_WORKERS)

BeforeDispatch = Callable[[tuple[int, ...]], None]

# Default model per provider for the raw-history CHUNK corpus. Voyage maps to
# voyage-context-4 (its contextualized-chunks model); local providers reuse the
# configured model unchanged (local-first posture, plain per-chunk embedding).
_DEFAULT_CHUNK_MODELS = {"voyage": "voyage-context-4", "voyageai": "voyage-context-4"}


def default_chunk_model(provider: str, configured_model: str) -> str:
    """Return the chunk-corpus model for a provider.

    Voyage gets the contextualized voyage-context-4 default; every other
    provider reuses the configured summary model so the local-first posture is
    unchanged (fastembed/ollama chunk-embed with the same local model).
    """
    key = str(provider or "").strip().lower()
    mapped = _DEFAULT_CHUNK_MODELS.get(key)
    if mapped:
        return mapped
    return str(configured_model or "").strip()


def embed_contextualized(
    provider: "EmbeddingProvider",
    chunks_by_doc: "Sequence[Sequence[str]]",
) -> list[list[list[float]]]:
    """Embed per-document chunk lists, contextualizing when the provider can.

    Returns one vector list per input document, aligned to that document's
    chunks. If the provider exposes ``embed_contextualized`` (Voyage), that is
    used; otherwise the chunks are flattened, embedded with the plain
    ``embed_documents`` path (ollama/fastembed local-first fallback), and
    regrouped by document so the return shape is identical either way.
    """
    method = getattr(provider, "embed_contextualized", None)
    if callable(method):
        return method(chunks_by_doc)
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for chunks in chunks_by_doc:
        start = len(flat)
        flat.extend(str(chunk) for chunk in chunks)
        spans.append((start, len(flat)))
    vectors = provider.embed_documents(flat) if flat else []
    return [[list(vector) for vector in vectors[start:end]] for start, end in spans]


class EmbeddingProvider(Protocol):
    """Minimal provider contract consumed by later embedding workers."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator["EmbeddedDocumentBatch"]: ...

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


class ProviderPreDispatchError(EmbeddingProviderError):
    """Definitive outcome: durable marking ran, but provider I/O never started."""

    kind = "pre_dispatch"
    transport_started = False


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


@dataclass(frozen=True)
class EmbeddedDocumentBatch:
    """One provider-accepted request, preserving original document indexes."""

    indexes: tuple[int, ...]
    vectors: tuple[tuple[float, ...], ...]


def _run_blocking_with_deadline(
    call: Callable[[], Any], *, timeout: float, provider: str
) -> Any:
    """Bound a local blocking encoder without waiting for an overrun worker."""
    budget = max(0.0, float(timeout))
    if budget <= 0:
        raise EmbeddingProviderError(f"{provider} operation deadline exceeded")
    deadline = time.monotonic() + budget
    worker_slots = _LOCAL_EMBED_WORKER_SLOTS
    if not worker_slots.acquire(timeout=budget):
        raise EmbeddingProviderError(f"{provider} worker capacity exhausted")
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((True, call()))
        except BaseException as exc:  # propagated in the caller thread
            result.put((False, exc))
        finally:
            # A timed-out caller deliberately leaves the slot held until its
            # daemon worker really exits. Repeated timeouts therefore cannot
            # accumulate an unbounded number of live local encoders.
            worker_slots.release()

    worker = threading.Thread(
        target=run, name=f"lcm-{provider.lower()}-embedding", daemon=True
    )
    try:
        worker.start()
    except BaseException:
        worker_slots.release()
        raise
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        raise EmbeddingProviderError(f"{provider} operation deadline exceeded")
    ok, value = result.get_nowait()
    if not ok:
        raise value
    return value


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
        except Exception as exc:
            # Authentication failures are definitive operator/configuration
            # errors, not evidence that the provider is temporarily
            # unavailable. Keeping them out of the breaker preserves the
            # actionable VoyageError on every attempt instead of eventually
            # replacing it with a misleading cooling-down error.
            if not (isinstance(exc, VoyageError) and exc.kind == "auth"):
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
        if isinstance(value, str):
            return "[REDACTED]"
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
        max_batch_items: int = _VOYAGE_MAX_BATCH_ITEMS,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("Voyage embedding model must not be empty")
        self._transport = transport or _default_http_transport
        self.timeout = float(timeout)
        self._sleep = sleeper
        self._dim = 0
        # Voyage rejects any request above 1,000 input items; that hard limit is
        # authoritative regardless of config, so clamp down to it. A config value
        # above the cap (e.g. 2000) would otherwise emit a >1000-input request.
        self.max_batch_items = max(1, min(int(max_batch_items), _VOYAGE_MAX_BATCH_ITEMS))
        self.last_skipped_documents: list[int] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def _error(
        self, response: HttpResponse, *, include_body: bool = True
    ) -> VoyageError:
        status = response.status
        if status in {401, 403}:
            kind = "auth"
        elif status == 429:
            kind = "rate_limit"
        elif 400 <= status < 500:
            kind = "bad_request"
        else:
            kind = "server_error"
        scrubbed = (
            _scrub_response_body(response.body)
            if include_body
            else "[SKIPPED: response-processing deadline]"
        )
        logger.warning("Voyage embedding error status=%s body=%s", status, scrubbed)
        return VoyageError(kind, f"Voyage embedding request failed ({status})", status_code=status)

    def _request(
        self,
        texts: Sequence[str],
        *,
        input_type: str,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        retry_after_budget_s: float = _RETRY_AFTER_BUDGET_S,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise VoyageError("auth", "VOYAGE_API_KEY is not set")
        payload = {
            "model": self.model_id,
            "input": list(texts),
            "input_type": input_type,
            "truncation": False,
        }
        request_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        attempts = max(1, int(max_attempts))
        retry_after_spent = 0.0
        last_error: VoyageError | None = None
        # ONE monotonic deadline across every attempt AND its backoff sleep, so
        # a tiny budget returns in ~that budget rather than accumulating
        # per-attempt timeouts plus exponential backoff.
        deadline = (
            time.monotonic() + max(0.0, float(deadline_budget_s))
            if deadline_budget_s is not None
            else None
        )

        def _remaining() -> float | None:
            return None if deadline is None else deadline - time.monotonic()

        for attempt in range(attempts):
            remaining = _remaining()
            if remaining is not None:
                if remaining <= 0:
                    raise last_error or VoyageError(
                        "timeout", "Voyage operation deadline exceeded"
                    )
            if before_transport is not None:
                before_transport()
            # The durable callback is intentionally the final preflight. It may
            # block on SQLite, so recompute the budget AFTER it returns and do
            # not begin billable transport with a stale timeout.
            remaining = _remaining()
            if remaining is not None:
                if remaining <= 0:
                    raise ProviderPreDispatchError(
                        "Voyage operation deadline exceeded before transport"
                    )
                attempt_timeout = min(request_timeout, remaining)
            else:
                attempt_timeout = request_timeout
            try:
                response = self._transport(
                    url=_VOYAGE_URL,
                    payload=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=attempt_timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                last_error = VoyageError("network", f"Voyage network error: {exc}")
                # Once transport starts, a timeout/network exception cannot
                # prove the remote rejected the request. Never automatically
                # resend an identical potentially-billable operation.
                raise last_error from exc

            if 200 <= response.status < 300:
                # Transport completion does not stop the operation clock. A
                # large/malicious response can make JSON decode, sorting, and
                # vector coercion block, so keep the complete success parser
                # inside the same absolute budget and do not publish a result
                # that only became available after expiry.
                remaining = _remaining()
                if remaining is not None and remaining <= 0:
                    raise VoyageError(
                        "timeout",
                        "Voyage operation deadline exceeded before response processing",
                    )

                def parse_success() -> list[list[float]]:
                    data = _response_json(response, provider="Voyage")
                    rows = data.get("data")
                    if not isinstance(rows, list):
                        raise EmbeddingProviderError(
                            "Voyage response did not contain embedding data"
                        )
                    ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
                    parsed = _coerce_vectors(
                        [
                            row.get("embedding")
                            for row in ordered
                            if isinstance(row, dict)
                        ],
                        provider="Voyage",
                    )
                    if len(parsed) != len(texts):
                        raise EmbeddingProviderError(
                            "Voyage returned a different number of embeddings than requested"
                        )
                    return parsed

                try:
                    vectors = (
                        parse_success()
                        if remaining is None
                        else _run_blocking_with_deadline(
                            parse_success,
                            timeout=remaining,
                            provider="Voyage response processing",
                        )
                    )
                except EmbeddingProviderError as exc:
                    if deadline is not None and _remaining() <= 0:
                        raise VoyageError(
                            "timeout",
                            "Voyage operation deadline exceeded during response processing",
                        ) from exc
                    raise
                if deadline is not None and _remaining() <= 0:
                    raise VoyageError(
                        "timeout",
                        "Voyage operation deadline exceeded during response processing",
                    )
                return vectors

            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                raise self._error(response, include_body=False)
            if remaining is None:
                last_error = self._error(response)
            else:
                try:
                    last_error = _run_blocking_with_deadline(
                        lambda: self._error(response),
                        timeout=remaining,
                        provider="Voyage error response processing",
                    )
                except EmbeddingProviderError:
                    # The status code itself is already authoritative even if
                    # diagnostic body scrubbing cannot finish in-budget. Do
                    # not wait for it, and never turn a 5xx into an auto-retry.
                    last_error = self._error(response, include_body=False)
            if deadline is not None and _remaining() <= 0:
                raise self._error(response, include_body=False)
            # A 429 is an authoritative rejection and is safe to retry. A 5xx
            # may follow remote acceptance, so it is deliberately fail-closed.
            retryable = response.status == 429
            if not retryable or attempt + 1 >= attempts:
                raise last_error
            if response.status == 429:
                delay = _retry_after_seconds(_header(response.headers, "Retry-After"))
                remaining_retry_after = retry_after_budget_s - retry_after_spent
                if delay > remaining_retry_after:
                    raise last_error
                retry_after_spent += delay
            else:
                delay = min(25.0, 0.5 * (2 ** attempt))
            if not self._sleep_within_deadline(delay, deadline):
                raise last_error
        raise last_error or VoyageError("network", "Voyage request failed")

    def _sleep_within_deadline(self, delay: float, deadline: float | None) -> bool:
        """Sleep ``delay`` only if it fits the absolute deadline; else give up.

        Returns True if the caller may retry (the sleep happened within budget),
        False if the backoff would blow the deadline and the caller should stop.
        """
        if deadline is None:
            self._sleep(delay)
            return True
        remaining = deadline - time.monotonic()
        if delay >= remaining:
            return False
        self._sleep(delay)
        return True

    def _remember_dim(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Voyage returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Voyage embedding dimension changed within the process")
            self._dim = len(vector)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        deadline = time.monotonic() + max(0.0, self.timeout)
        accepted: list[tuple[int, str, int]] = []
        self.last_skipped_documents = []
        for index, text in enumerate(texts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                )

            def prepare_document(value=text) -> tuple[str, int]:
                prepared = str(value)
                return prepared, count_tokens(prepared)

            try:
                text, tokens = _run_blocking_with_deadline(
                    prepare_document,
                    timeout=remaining,
                    provider="Voyage document preprocessing",
                )
            except EmbeddingProviderError as exc:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                ) from exc
            if time.monotonic() >= deadline:
                raise VoyageError(
                    "timeout",
                    "Voyage operation deadline exceeded during document preprocessing",
                )
            if tokens > _VOYAGE_DOCUMENT_TOKEN_BUDGET:
                self.last_skipped_documents.append(index)
                logger.warning(
                    "Skipping Voyage document index=%d tokens=%d limit=%d",
                    index,
                    tokens,
                    _VOYAGE_DOCUMENT_TOKEN_BUDGET,
                )
                continue
            accepted.append((index, text, tokens))

        batch_indexes: list[int] = []
        batch: list[str] = []
        batch_tokens = 0
        for index, text, tokens in accepted:
            # Flush before the batch would exceed EITHER the token budget or the
            # provider's per-request item cap, so e.g. 1001 short documents split
            # into at least two requests.
            if batch and (
                batch_tokens + tokens > _VOYAGE_BATCH_TOKEN_BUDGET
                or len(batch) >= self.max_batch_items
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VoyageError("timeout", "Voyage operation deadline exceeded")
                indexes = tuple(batch_indexes)
                if before_dispatch is not None:
                    dispatch = partial(before_dispatch, indexes)
                else:
                    dispatch = None
                vectors = self._guarded(
                    lambda current=tuple(batch), budget=remaining: self._request(
                        current,
                        input_type="document",
                        max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
                        deadline_budget_s=budget,
                        before_transport=dispatch,
                    )
                )
                self._remember_dim(vectors)
                yield EmbeddedDocumentBatch(
                    indexes,
                    tuple(tuple(vector) for vector in vectors),
                )
                batch_indexes = []
                batch = []
                batch_tokens = 0
            batch_indexes.append(index)
            batch.append(text)
            batch_tokens += tokens
        if batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VoyageError("timeout", "Voyage operation deadline exceeded")
            indexes = tuple(batch_indexes)
            if before_dispatch is not None:
                dispatch = partial(before_dispatch, indexes)
            else:
                dispatch = None
            vectors = self._guarded(
                lambda current=tuple(batch), budget=remaining: self._request(
                    current,
                    input_type="document",
                    max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
                    deadline_budget_s=budget,
                    before_transport=dispatch,
                )
            )
            self._remember_dim(vectors)
            yield EmbeddedDocumentBatch(
                indexes,
                tuple(tuple(vector) for vector in vectors),
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            list(vector)
            for batch in self.embed_document_batches(texts)
            for vector in batch.vectors
        ]

    def embed_query(self, text: str) -> list[float]:
        # Normal (non-interactive) query embedding also gets ONE absolute
        # deadline across every attempt AND its backoff sleeps, so a tiny
        # configured timeout returns in ~that budget instead of stacking
        # per-attempt timeouts plus exponential backoff.
        vectors = self._guarded(
            lambda: self._request(
                (str(text),), input_type="query", deadline_budget_s=self.timeout
            )
        )
        self._remember_dim(vectors)
        return vectors[0]

    def embed_query_interactive(self, text: str, *, timeout: float) -> list[float]:
        """Embed a latency-sensitive query under one absolute time budget.

        Retries are allowed only insofar as they fit ``timeout`` — the single
        monotonic deadline covers every attempt plus any backoff, so a tiny
        budget returns within that budget instead of stacking per-attempt waits.
        """
        vectors = self._guarded(
            lambda: self._request(
                (str(text),),
                input_type="query",
                timeout=timeout,
                retry_after_budget_s=0.0,
                deadline_budget_s=timeout,
            )
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

    def _request(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        request_timeout = self.timeout if timeout is None else max(0.001, float(timeout))
        attempts = max(1, int(max_attempts))
        deadline = (
            time.monotonic() + max(0.0, float(deadline_budget_s))
            if deadline_budget_s is not None
            else None
        )
        for attempt in range(attempts):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EmbeddingProviderError("Ollama operation deadline exceeded")
            if before_transport is not None:
                before_transport()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProviderPreDispatchError(
                        "Ollama operation deadline exceeded before transport"
                    )
                attempt_timeout = min(request_timeout, remaining)
            else:
                attempt_timeout = request_timeout
            try:
                response = self._transport(
                    url=f"{self.base_url}/api/embed",
                    # truncate=false so oversized input fails loudly instead of
                    # being silently truncated to a misleading embedding.
                    payload={
                        "model": self.model_id,
                        "input": list(texts),
                        "truncate": False,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=attempt_timeout,
                )
            except (OSError, TimeoutError, urllib.error.URLError) as exc:
                # Remote/local daemon acceptance is ambiguous after transport
                # begins, so an automatic identical resend is not safe.
                raise EmbeddingProviderError(f"Ollama network error: {exc}") from exc
            if not 200 <= response.status < 300:
                raise EmbeddingProviderError(
                    f"Ollama embedding request failed ({response.status})"
                )
            remaining = (
                None if deadline is None else deadline - time.monotonic()
            )
            if remaining is not None and remaining <= 0:
                raise EmbeddingProviderError(
                    "Ollama operation deadline exceeded before response processing"
                )

            def parse_success() -> list[list[float]]:
                payload = _response_json(response, provider="Ollama")
                raw = payload.get("embeddings")
                if raw is None and "embedding" in payload:
                    raw = [payload["embedding"]]
                parsed = _coerce_vectors(raw, provider="Ollama")
                if len(parsed) != len(texts):
                    raise EmbeddingProviderError(
                        "Ollama returned a different number of embeddings than requested"
                    )
                return parsed

            vectors = (
                parse_success()
                if remaining is None
                else _run_blocking_with_deadline(
                    parse_success,
                    timeout=remaining,
                    provider="Ollama response processing",
                )
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise EmbeddingProviderError(
                    "Ollama operation deadline exceeded during response processing"
                )
            return vectors
        raise EmbeddingProviderError("Ollama embedding request failed")

    def _embed(
        self,
        texts: Sequence[str],
        *,
        timeout: float | None = None,
        max_attempts: int = _MAX_ATTEMPTS,
        deadline_budget_s: float | None = None,
        before_transport: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._guarded(
            lambda: self._request(
                tuple(str(text) for text in texts),
                timeout=timeout,
                max_attempts=max_attempts,
                deadline_budget_s=deadline_budget_s,
                before_transport=before_transport,
            )
        )
        for vector in vectors:
            if not vector:
                raise EmbeddingProviderError("Ollama returned an empty embedding")
            if self._dim and len(vector) != self._dim:
                raise EmbeddingProviderError("Ollama embedding dimension changed within the process")
            self._dim = len(vector)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Normal operations also get ONE absolute deadline across attempts +
        # backoff (self.timeout), so retryable failures cannot stack backoff
        # sleeps beyond the configured budget.
        return self._embed(texts, deadline_budget_s=self.timeout)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        normalized = tuple(str(text) for text in texts)
        if not normalized:
            return
        indexes = tuple(range(len(normalized)))
        vectors = self._embed(
            normalized,
            max_attempts=1 if before_dispatch is not None else _MAX_ATTEMPTS,
            deadline_budget_s=self.timeout,
            before_transport=(
                (lambda: before_dispatch(indexes))
                if before_dispatch is not None
                else None
            ),
        )
        yield EmbeddedDocumentBatch(
            indexes,
            tuple(tuple(vector) for vector in vectors),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed((text,), deadline_budget_s=self.timeout)[0]

    def embed_query_interactive(self, text: str, *, timeout: float) -> list[float]:
        """Embed a latency-sensitive query under one absolute time budget."""
        return self._embed(
            (text,), timeout=timeout, deadline_budget_s=timeout
        )[0]


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
        timeout: float = 3.0,
        breaker: EmbeddingCircuitBreaker | None = None,
        spend_guard: EmbeddingSpendGuard | None = None,
    ) -> None:
        super().__init__(breaker=breaker, spend_guard=spend_guard)
        self._model_id = str(model).strip()
        if not self._model_id:
            raise ValueError("FastEmbed embedding model must not be empty")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_FASTEMBED_CACHE
        self.timeout = float(timeout)
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

    def _embed(
        self,
        texts: Sequence[str],
        *,
        query: bool = False,
        before_dispatch: Callable[[], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        deadline = time.monotonic() + max(0.0, self.timeout)

        def encode() -> list[list[float]]:
            return _run_blocking_with_deadline(
                lambda: self._encode_local(
                    texts,
                    query=query,
                    before_dispatch=before_dispatch,
                    deadline=deadline,
                ),
                timeout=deadline - time.monotonic(),
                provider="FastEmbed",
            )

        vectors = self._guarded(encode)
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

    def _encode_local(
        self,
        texts: Sequence[str],
        *,
        query: bool,
        before_dispatch: Callable[[], None] | None = None,
        deadline: float | None = None,
    ) -> list[list[float]]:
        model = self._ensure_local()
        if deadline is not None and time.monotonic() >= deadline:
            raise ProviderPreDispatchError(
                "FastEmbed operation deadline exceeded before dispatch"
            )
        if before_dispatch is not None:
            before_dispatch()
        if deadline is not None and time.monotonic() >= deadline:
            raise ProviderPreDispatchError(
                "FastEmbed operation deadline exceeded before encode"
            )
        # FastEmbed models are asymmetric: queries and passages get different
        # instruction prefixes. Use the query-specific API for queries.
        encode = model.query_embed if query else model.embed
        return _coerce_vectors(list(encode(list(texts))), provider="FastEmbed")

    def warmup(self) -> list[float]:
        """The only path allowed to download a missing FastEmbed model."""
        if self._model is None:
            self._model = self._construct(allow_download=True)
        return self.embed_query("warmup")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(tuple(str(text) for text in texts), query=False)

    def embed_document_batches(
        self,
        texts: Sequence[str],
        *,
        before_dispatch: BeforeDispatch | None = None,
    ) -> Iterator[EmbeddedDocumentBatch]:
        normalized = tuple(str(text) for text in texts)
        if not normalized:
            return
        indexes = tuple(range(len(normalized)))
        vectors = self._embed(
            normalized,
            query=False,
            before_dispatch=(
                (lambda: before_dispatch(indexes))
                if before_dispatch is not None
                else None
            ),
        )
        yield EmbeddedDocumentBatch(
            indexes,
            tuple(tuple(vector) for vector in vectors),
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed((str(text),), query=True)[0]


def resolve_provider(
    config: LCMConfig, *, for_backfill: bool = False
) -> EmbeddingProvider | None:
    """Resolve inert embedding config without making provider calls.

    ``for_backfill`` selects the bulk-operation contract. It bypasses the
    interactive per-minute spend guard and uses the separate backfill timeout:
    bulk ``lcm embed backfill --apply`` embeds thousands of documents (e.g.
    ~1920 docs at batch 32 → ~60 provider calls), and neither its call volume
    nor a normal document batch/local model load should be governed by the
    latency-sensitive query policy. The backfill worker retains its own
    operation budget and renewable lease.
    """
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    model = str(getattr(config, "embedding_model", "") or "").strip()
    if not provider and not model:
        return None
    if not provider or not model:
        raise ProviderUnavailable(
            "LCM_EMBEDDING_PROVIDER and LCM_EMBEDDING_MODEL must both be set"
        )
    # max_calls=0 disables the sliding-window guard (allows() always True,
    # record_call() a no-op); the circuit breaker still trips on failures.
    spend_guard = EmbeddingSpendGuard(max_calls=0) if for_backfill else None
    timeout_field = (
        "embedding_backfill_timeout_s"
        if for_backfill
        else "embedding_query_timeout_s"
    )
    timeout_default = 120.0 if for_backfill else 3.0
    timeout = float(getattr(config, timeout_field, timeout_default))
    if provider in {"voyage", "voyageai"}:
        return VoyageProvider(
            model,
            timeout=timeout,
            spend_guard=spend_guard,
            max_batch_items=int(
                getattr(config, "embedding_max_batch_items", _VOYAGE_MAX_BATCH_ITEMS)
            ),
        )
    if provider == "ollama":
        return OllamaProvider(
            model,
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
            timeout=timeout,
            spend_guard=spend_guard,
        )
    if provider in {"fastembed", "fast-embed"}:
        return FastembedProvider(
            model, timeout=timeout, spend_guard=spend_guard
        )
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
