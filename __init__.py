"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a DAG-based context engine
that persists every message and provides structured retrieval tools.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import logging
import os

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _make_wrapped_handler(tool_name: str, engine):
    """Route a registered lcm_* tool through the engine dispatch path."""
    def _wrapped(args: dict, **kwargs) -> str:
        return engine.handle_tool_call(tool_name, args, **kwargs)
    return _wrapped


def register(ctx):
    """Plugin entry point — register the LCM context engine and tools."""
    from .config import LCMConfig
    from .engine import LCMEngine
    from .schemas import (
        LCM_GREP,
        LCM_LOAD_SESSION,
        LCM_DESCRIBE,
        LCM_EXPAND,
        LCM_EXPAND_QUERY,
        LCM_STATUS,
        LCM_DOCTOR,
    )

    config = LCMConfig.from_env()

    # Resolve hermes_home for profile-scoped storage
    hermes_home = ""
    try:
        from hermes_cli.config import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:
        import os
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

    engine = LCMEngine(config=config, hermes_home=hermes_home)

    # Register as the context engine (replaces ContextCompressor)
    ctx.register_context_engine(engine)

    # Register tools via the plugin registry so they are discoverable by
    # hosts that support plugin-provided tool registration. Keep the
    # context-engine toolset and dispatch path so platform_toolsets gating,
    # schema deduplication, and live messages=... ingestion remain equivalent
    # to Hermes Agent's native context-engine handling.
    _TOOLS = {
        "lcm_grep": LCM_GREP,
        "lcm_load_session": LCM_LOAD_SESSION,
        "lcm_describe": LCM_DESCRIBE,
        "lcm_expand": LCM_EXPAND,
        "lcm_expand_query": LCM_EXPAND_QUERY,
        "lcm_status": LCM_STATUS,
        "lcm_doctor": LCM_DOCTOR,
    }
    register_tool = getattr(ctx, "register_tool", None)
    if callable(register_tool):
        for name, schema in _TOOLS.items():
            try:
                register_tool(
                    name=name,
                    toolset="context_engine",
                    schema=schema,
                    handler=_make_wrapped_handler(name, engine),
                    description=schema.get("description", ""),
                )
            except Exception as exc:
                logger.warning(
                    "LCM tool registration failed for %s; "
                    "continuing with context-engine schemas: %s",
                    name,
                    exc,
                )
    else:
        logger.info(
            "LCM tool registration unavailable on this Hermes host; "
            "continuing with context-engine schemas"
        )

    register_command = getattr(ctx, "register_command", None)
    slash_enabled = _env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", default=False)
    if callable(register_command) and slash_enabled:
        from .command import handle_lcm_command

        register_command(
            "lcm",
            lambda raw_args: handle_lcm_command(raw_args, engine),
            description="LCM status and diagnostics",
        )
    elif callable(register_command):
        logger.info("LCM slash command registration disabled (set LCM_ENABLE_SLASH_COMMAND=1 to enable /lcm)")
    else:
        logger.info("LCM slash command registration unavailable on this Hermes host; continuing without /lcm")

    logger.info("LCM plugin loaded — lossless context management active")
