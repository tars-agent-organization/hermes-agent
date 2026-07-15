import json
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None,
                  conversational_mode=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if dm_policy is not None:
        extra["dm_policy"] = dm_policy
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_policy is not None:
        extra["group_policy"] = group_policy
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from
    if conversational_mode is not None:
        extra["conversational_mode"] = conversational_mode

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._free_response_chats = adapter._whatsapp_free_response_chats()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    return adapter


def _group_message(body="hello", **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": "120363001234567890@g.us",
        "senderId": "6281234567890@s.whatsapp.net",
        "from": "6281234567890@s.whatsapp.net",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net", "15551230000@lid"],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def _conversation_mode(**overrides):
    config = {
        "enabled": True,
        "allowed_from": ["6281234567890@s.whatsapp.net"],
        "idle_timeout_seconds": 300,
        "max_duration_seconds": 1800,
    }
    config.update(overrides)
    return config


def test_explicit_name_mention_opens_temporary_conversation():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )

    assert adapter._should_process_message(_group_message("Tars, olha isso")) is True
    assert adapter._should_process_message(_group_message("pesado, o bicho é monstro")) is True


def test_conversation_lease_is_bound_to_sender_and_group():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    adapter._should_process_message(_group_message("Tars, olha isso"))

    assert adapter._should_process_message(
        _group_message(
            "outro participante",
            senderId="6289999999999@s.whatsapp.net",
            **{"from": "6289999999999@s.whatsapp.net"},
        )
    ) is False
    assert adapter._should_process_message(
        _group_message("mesmo remetente, outro grupo", chatId="999999999999@g.us")
    ) is False
    assert adapter._should_process_message(_group_message("continuação do dono")) is True


def test_silence_request_closes_lease_and_is_consumed():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    adapter._should_process_message(_group_message("Tars, olha isso"))

    assert adapter._should_process_message(_group_message("fica quieto")) is False
    assert adapter._should_process_message(_group_message("não deve passar")) is False


def test_other_participant_cannot_close_lease():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    adapter._should_process_message(_group_message("Tars, olha isso"))

    assert adapter._should_process_message(
        _group_message(
            "fica quieto",
            senderId="6289999999999@s.whatsapp.net",
            **{"from": "6289999999999@s.whatsapp.net"},
        )
    ) is False
    assert adapter._should_process_message(_group_message("ainda ativo")) is True


def test_conversation_lease_expires_after_five_idle_minutes():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    adapter._should_process_message(_group_message("Tars, olha isso"))
    lease = adapter._conversation_lease_map()["120363001234567890@g.us"]
    lease.last_activity_at -= 301

    assert adapter._should_process_message(_group_message("tarde demais")) is False


def test_absolute_duration_expires_even_with_recent_activity():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    adapter._should_process_message(_group_message("Tars, olha isso"))
    lease = adapter._conversation_lease_map()["120363001234567890@g.us"]
    lease.opened_at -= 1801

    assert adapter._should_process_message(_group_message("sessão longa demais")) is False


def test_reply_and_slash_command_do_not_open_conversation_lease():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )

    assert adapter._should_process_message(
        _group_message("respondendo", quotedParticipant="15551230000@lid")
    ) is True
    assert adapter._should_process_message(_group_message("seguinte")) is False
    assert adapter._should_process_message(_group_message("/status")) is True
    assert adapter._should_process_message(_group_message("seguinte")) is False


def test_unlisted_sender_cannot_open_conversation_lease():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*@?tars\b"],
        group_policy="open",
        conversational_mode=_conversation_mode(),
    )
    other = {
        "senderId": "6289999999999@s.whatsapp.net",
        "from": "6289999999999@s.whatsapp.net",
    }

    assert adapter._should_process_message(_group_message("Tars, oi", **other)) is True
    assert adapter._should_process_message(_group_message("continuação", **other)) is False


def test_config_bridges_whatsapp_conversational_mode(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  require_mention: true\n"
        "  conversational_mode:\n"
        "    enabled: true\n"
        "    allowed_from:\n"
        "      - owner@lid\n"
        "    idle_timeout_seconds: 300\n"
        "    max_duration_seconds: 1800\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_REQUIRE_MENTION", raising=False)

    config = load_gateway_config()

    mode = config.platforms[Platform.WHATSAPP].extra["conversational_mode"]
    assert mode["enabled"] is True
    assert mode["allowed_from"] == ["owner@lid"]
    assert mode["idle_timeout_seconds"] == 300


def _dm_message(body="hello", **overrides):
    data = {
        "isGroup": False,
        "body": body,
        "senderId": "6281234567890@s.whatsapp.net",
        "from": "6281234567890@s.whatsapp.net",
        "botIds": [],
        "mentionedIds": [],
    }
    data.update(overrides)
    return data


# --- Existing tests (unchanged logic, updated helper) ---


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True, group_policy="open")

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(
        _group_message(
            "hi there",
            mentionedIds=["15551230000@s.whatsapp.net"],
        )
    ) is True
    assert adapter._should_process_message(
        _group_message(
            "replying",
            quotedParticipant="15551230000@lid",
        )
    ) is True
    assert adapter._should_process_message(_group_message("/status")) is True


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"(", r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_free_response_chats_bypass_mention_gating():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["120363001234567890@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_free_response_chats_does_not_bypass_other_groups():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["999999999999@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_mention_stripping_removes_bot_phone_from_body():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("@15551230000 what is the weather?")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert "15551230000" not in cleaned
    assert "weather" in cleaned


# --- New dm_policy tests ---


def test_dm_policy_disabled_still_allows_groups():
    adapter = _make_adapter(
        dm_policy="disabled",
        require_mention=False,
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello")) is True


# --- New group_policy tests ---


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert config.platforms[Platform.WHATSAPP].extra["group_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["group_allow_from"] == ["120363001234567890@g.us"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


# --- Broadcast / status / newsletter pseudo-chats are always dropped ---


def test_status_broadcast_chats_are_always_dropped():
    """Felipe's gateway.log showed the agent replying to status@broadcast
    (a contact's WhatsApp Story update). These pseudo-chats aren't real
    conversations and the adapter must drop them regardless of dm_policy.
    """

    # Even on the most permissive config — open DMs, no allowlist — Stories
    # and Channel posts must not reach the agent.
    adapter = _make_adapter(dm_policy="open")

    # Classic Story update — what Felipe was seeing in production.
    status_msg = _dm_message(
        body="[video received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(status_msg) is False

    # Channel / Newsletter broadcast posts.
    newsletter_msg = _dm_message(
        body="check out our latest post",
        chatId="120363999999999999@newsletter",
        senderId="120363999999999999@newsletter",
    )
    assert adapter._should_process_message(newsletter_msg) is False


def test_broadcast_filter_runs_before_allowlist():
    """A status@broadcast message from an allowlisted sender still drops —
    we never want to reply to Stories, even from authorized contacts.
    """
    adapter = _make_adapter(
        dm_policy="allowlist",
        allow_from=["34612345678@s.whatsapp.net"],
    )

    msg = _dm_message(
        body="[image received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(msg) is False

