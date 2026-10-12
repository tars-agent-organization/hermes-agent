"""Structured context-usage events emitted after provider-reported usage."""

import logging

logger = logging.getLogger(__name__)


def emit_context_usage(agent) -> None:
    callback = getattr(agent, "context_usage_callback", None)
    compressor = getattr(agent, "context_compressor", None)
    if callback is None or compressor is None:
        return
    try:
        callback({
            "session_id": getattr(agent, "session_id", None),
            "model": getattr(agent, "model", ""),
            "provider": getattr(agent, "provider", ""),
            "prompt_tokens": getattr(compressor, "last_prompt_tokens", -1),
            "context_length": getattr(compressor, "context_length", 0),
            "compression_count": getattr(compressor, "compression_count", 0),
        })
    except Exception:
        logger.debug("context usage callback failed", exc_info=True)