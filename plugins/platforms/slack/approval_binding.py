"""Exact-card binding for profile-opted-in Slack command approvals."""
import json

from agent.i18n import t
from agent.redact import redact_sensitive_text


async def send_prompt(self, prompt):
    """Block Kit approval prompt; the buttons call ``resolve_gateway_approval()`` to unblock the
    waiting agent thread — same mechanism as the text ``/approve`` flow."""
    from gateway.platforms.base import SendResult
    if error := prompt_error(self, prompt):
        return SendResult(success=False, error=error)

    def _build() -> tuple[str, list]:
        actions = [
            self._button(label, self._EA_ACTION_IDS[choice], button_value(prompt), style=style)
            for label, choice, style in prompt.actions]
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": prompt.text}},
            {"type": "actions", "elements": actions}]
        append_scope(blocks, prompt)
        return t("platform.slack.approval.fallback_text", command=prompt.command[:100]), blocks

    result = await self._send_interactive_prompt(
        prompt.chat_id, prompt.metadata, _build, "send_exec_approval",
        resolved=self._approval_resolved, resolved_max=self._APPROVAL_RESOLVED_MAX)
    record_prompt(self, prompt, result)
    return result


def button_value(prompt):
    request_id = (prompt.metadata or {}).get("approval_request_id")
    return f"{prompt.session_key}|{request_id}" if request_id else prompt.session_key


def scope_text(prompt):
    meta = prompt.metadata or {}
    return redact_sensitive_text(json.dumps({"origin": meta.get("approval_origin"),
        "execution_scope": meta.get("approval_execution_scope", {}),
        "request_id": meta.get("approval_request_id"), "consent": "this invocation only"},
        ensure_ascii=False), force=True)


def prompt_error(adapter, prompt):
    if not (prompt.metadata or {}).get("approval_origin"):
        return None
    if (not prompt.metadata.get("approval_request_id") or len(scope_text(prompt)) > 2900
            or len(prompt.command) > adapter._exec_approval_cmd_budget(prompt.description, prompt.smart_denied)):
        return "Bound approval requires an id and complete command/scope previews"
    return None


def append_scope(blocks, prompt):
    if (prompt.metadata or {}).get("approval_origin"):
        blocks.insert(1, {"type": "section", "text": {"type": "plain_text", "text": scope_text(prompt)}})


def record_prompt(adapter, prompt, result):
    binding = (prompt.metadata or {}).get("approval_origin")
    if binding and result.success and result.message_id:
        key = adapter._workspace_message_marker(adapter._metadata_team_id(prompt.metadata), result.message_id)
        adapter._approval_bindings[key] = {
            "origin": dict(binding), "request_id": prompt.metadata["approval_request_id"],
            "session_key": prompt.session_key, "value": button_value(prompt)}
        adapter._trim_oldest_dict_entries(adapter._approval_bindings, adapter._APPROVAL_RESOLVED_MAX)


async def handle_bound_action(adapter, started, choice):
    team, _, value, message, msg_ts, channel, user_name, user_id = started
    if "|" not in value:
        return False
    # A bound card whose memory was evicted/restarted NEVER becomes a legacy
    # FIFO approval. Slack's authenticated body, not button text, proves origin.
    key = adapter._workspace_message_marker(team, msg_ts)
    binding = adapter._approval_bindings.get(key)
    if binding is None:
        return True
    origin = {"user_id": user_id, "chat_id": channel,
              "thread_id": message.get("thread_ts"), "team_id": team}
    if origin != binding["origin"] or value != binding["value"]:
        return True
    from tools.approval import resolve_gateway_approval
    count = resolve_gateway_approval(binding["session_key"], choice,
                                     request_id=binding["request_id"], origin=origin)
    if count:
        adapter._approval_bindings.pop(key, None)
        adapter._approval_resolved[key] = True
    decision = (t(adapter._APPROVAL_DECISION_KEYS[choice], user=user_name) if count
                else t("platform.shared.approval_expired"))
    await adapter._finalize_interactive_message(channel, msg_ts, adapter._section_text(message), decision,
                                               "Command approval request", "approval", team or None)
    return True
