"""Bind a drained request's transport context without borrowing the opening turn."""

from agent.runtime_cwd import reset_session_cwd, scoped_session_cwd, set_session_cwd
from gateway.session import SessionContext
from gateway.session_identity import replace_source


async def run_queued_request(runner, event, **kwargs):
    """The in-band queue bypasses inbound preparation, including its context binding.

    Reuse the normal gateway binder for the queued event. Restore tokens instead
    of clearing them: the outer turn still needs its own context during delivery.
    No event means no request metadata, never the prior message's authority.
    """
    source = replace_source(kwargs["source"], message_id=getattr(event, "message_id", None))
    context = SessionContext(source=source, connected_platforms=[], home_channels={},
                             session_key=kwargs.get("session_key") or "")
    cwd_token = set_session_cwd(scoped_session_cwd())
    tokens = []
    try:
        tokens = runner._set_session_env(context, trusted_request_metadata=getattr(event, "metadata", None))
        return await runner._run_agent(**kwargs)
    finally:
        for token in reversed(tokens):
            token.var.reset(token)
        reset_session_cwd(cwd_token)
