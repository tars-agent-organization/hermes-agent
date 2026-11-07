"""Opt-in Slack owner/thread-bound approval delivery. No text/FIFO fallback."""
import contextvars
import logging

from gateway.config import Platform
from tools.approval_handoff import owner_id

logger = logging.getLogger("gateway.run")


def notify_approval(turn, data):
    if not send_bound_approval(turn, data):
        _send_legacy_approval(turn, data)


def _send_legacy_approval(self, approval_data: dict) -> None:
    """Send the approval request from the agent thread: the adapter's interactive button
    approvals (``send_exec_approval``) when available, else plain text with ``/approve`` steps."""
    from gateway.run_turn_runner import _ExecApprovalDeclined, _renders_exec_approval_buttons
    from gateway.platforms.base_exec_approval import ea_default_reason_text
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    ctx = self._ctx
    adapter = ctx._status_adapter
    # Slack's assistant_threads_setStatus disables the compose box, so the user can't type
    # /approve while "is thinking..." shows. Pausing stops _keep_typing re-setting it; resumed
    # in approve/deny.
    adapter.pause_typing_for_chat(ctx._status_chat_id)
    self._close_native_stream_boundary("Approval")
    # Redact credentials before display: the raw command string can carry secrets. Both the button and plain-text paths use this value.
    cmd = _redact_approval_command(approval_data.get("command", ""))
    desc = approval_data.get("description") or ea_default_reason_text()
    flags = {k: approval_data.get(k, d) for k, d in (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
    # Check the *class*, not the instance — MagicMock auto-creates attributes in tests.
    if _renders_exec_approval_buttons(type(adapter)):
        try:
            fut = self._schedule(
                adapter.send_exec_approval(
                    chat_id=ctx._status_chat_id, command=cmd, session_key=ctx.session_key or "",
                    description=desc, metadata=ctx._status_thread_metadata, **flags,
                ),
                "send_exec_approval scheduling error",
            )
            if fut is None:
                raise RuntimeError("send_exec_approval: loop unavailable")
            outcome = _approval_send_outcome(fut, timeout=15)
            if outcome == "sent":
                # Without this, a card whose timer runs out keeps live buttons and nobody
                # learns the command did NOT run (only the TUI registered a settle hook).
                register_timeout_notice(
                    self, approval_data, command=cmd,
                    card_message_id=getattr(fut.result(timeout=0), "message_id", None))
                return
            if outcome == "ambiguous":
                # Timeout ≠ failure: the card may have posted with a late ack. The prompt
                # registration stays alive so a tap still resolves; re-sending made duplicate
                # cards + orphaned "/approve: nothing pending".
                logger.warning(
                    "Button-based approval send timed out — treating "
                    "as possibly-delivered (no re-send; the prompt "
                    "stays armed for a late tap)"
                )
                return
            if outcome == "declined":
                # P5(b): the connector AUTHORIZED this destination and
                # refused it. The text fallback below re-sends the same
                # content to the same chat, which would turn a refused
                # button card into a delivered plain-text one — the exact
                # leak the egress guard exists to stop. A decline is
                # definitive, so unlike `ambiguous` the registration is
                # torn down; unlike `failed`, nothing is re-sent.
                logger.warning(
                    "Button-based approval DECLINED by the connector's "
                    "egress guard — not falling back to text (the "
                    "destination is not approved for this connection)"
                )
                # RAISE, do not return. This function is the notify_cb for
                # `_await_gateway_decision`, which already has a correct
                # undeliverable path: a raising notify drops the queue entry
                # and returns `notify_failed`, unblocking the tool. Returning
                # quietly suppressed the text fallback (right) but left the
                # CENTRAL approval entry pending (wrong) — the dangerous
                # command then blocked until the approval timeout. My earlier
                # comment claimed the registration was torn down; only the
                # adapter's private prompt map was.
                raise _ExecApprovalDeclined(
                    "exec approval undeliverable: connector egress declined "
                    "this destination"
                )
            logger.warning("Button-based approval failed (send returned error), falling back to text")
        except _ExecApprovalDeclined:
            # Must escape this handler: the fallback below is a text send to
            # the destination the connector just refused.
            raise
        except Exception as e:
            logger.warning("Button-based approval failed, falling back to text: %s", e, exc_info=True)
    # Plain-text prompt with the adapter's typed prefix (e.g. `!approve`): typed "/" is blocked
    # in Slack threads and reserved by Matrix clients.
    msg = _format_exec_approval_fallback(cmd, desc, getattr(adapter, "typed_command_prefix", "/"), **flags)
    try:
        # Mark as approval prompt: WeCom routes it through the control lane and Telegram pushes it
        # in "important" mode (#132516). Never ``notify`` — A2A reads that as the turn-final reply.
        metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
        fut = self._schedule(
            adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
        )
        if fut is not None:
            fut.result(timeout=15)
            # No card to edit on the text path: the prompt has no buttons to drop and carries
            # the /approve instructions, so the timeout notice is posted as a new message.
            register_timeout_notice(self, approval_data, command=cmd, card_message_id=None)
    except Exception as e:
        logger.error("Failed to send approval request: %s", e, exc_info=True)


def send_bound_approval(turn, data):
    ctx = turn._ctx
    owner = owner_id()
    if not owner or ctx.source.platform != Platform.SLACK:
        return False
    from tools.approval import ack_gateway_approval
    from tools.approval_binding import bind_request
    from gateway.run import _approval_send_outcome, _redact_approval_command
    from gateway.run_turn_runner_approval_settle import register_timeout_notice

    adapter = ctx._status_adapter
    metadata = dict(ctx._status_thread_metadata or {})
    thread = adapter._resolve_thread_ts(None, metadata)
    team = adapter._metadata_team_id(metadata)
    # The transport source is immutable evidence. Never manufacture the owner's
    # identity for an internally injected completion or a different participant.
    source_thread = (ctx.source.thread_id or ctx.source.prospective_thread_id
                     or getattr(ctx, "inbound_message_id", None) or ctx.source.message_id)
    source_team = ctx.source.scope_id or ctx.source.guild_id
    if (ctx.source.user_id != owner or not thread or not team or thread != source_thread
            or team != source_team or ctx._status_chat_id != ctx.source.chat_id):
        raise RuntimeError("Bound approval requires the authenticated owner and exact Slack origin")
    origin = {"user_id": owner, "chat_id": ctx._status_chat_id, "thread_id": thread, "team_id": team}
    captured = contextvars.copy_context()

    def authorized():
        def check():
            return (owner_id() == owner
                    and turn._runner._is_user_authorized(ctx.source) is True
                    and adapter._is_interactive_user_authorized(
                        owner, channel_id=origin["chat_id"], user_name=ctx.source.user_name,
                        team_id=team) is True)
        return captured.copy().run(check)

    if not authorized():
        raise RuntimeError("Origin authorization no longer valid")
    request_id = data["request_id"]
    if not bind_request(ctx.session_key, request_id, origin, authorized):
        raise RuntimeError("Approval request is no longer active")
    metadata.update(approval_request_id=request_id, approval_origin=origin,
                    approval_execution_scope=data.get("execution_scope", {}))
    command = _redact_approval_command(data.get("command", ""))
    description = (f"{data.get('description', '')}\nOwner: <@{owner}>. "
                   f"Scope: one invocation; channel {origin['chat_id']}; thread {thread}; request {request_id}.")
    adapter.pause_typing_for_chat(ctx._status_chat_id)
    turn._close_native_stream_boundary("Approval")
    future = turn._schedule(adapter.send_exec_approval(
        chat_id=ctx._status_chat_id, command=command, session_key=ctx.session_key,
        description=description, metadata=metadata, allow_session=False, allow_permanent=False,
        smart_denied=data.get("smart_denied", False)), "Bound approval send failed")
    # An ambiguous send may have left a card in Slack. It is NOT approvable:
    # the waiter withdraws its request and a later click finds no matching id.
    if _approval_send_outcome(future, timeout=15) != "sent":
        raise RuntimeError("Approval delivery was not confirmed; no executable approval remains")
    receipt = future.result(timeout=0)
    if not receipt.message_id or not ack_gateway_approval(ctx.session_key, request_id):
        raise RuntimeError("Approval receipt missing or request no longer active")
    register_timeout_notice(turn, data, command=command, card_message_id=receipt.message_id)
    return True
