from types import SimpleNamespace

from agent.context_usage import emit_context_usage
from run_agent import AIAgent


def test_emit_context_usage_reports_live_compressor_values():
    samples = []
    agent = SimpleNamespace(
        session_id="session",
        model="model",
        provider="provider",
        context_usage_callback=samples.append,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=123,
            context_length=456,
            compression_count=2,
        ),
    )

    emit_context_usage(agent)

    assert samples == [{
        "session_id": "session",
        "model": "model",
        "provider": "provider",
        "prompt_tokens": 123,
        "context_length": 456,
        "compression_count": 2,
    }]


def test_agent_constructor_preserves_optional_context_usage_callback():
    callback = lambda sample: None
    agent = AIAgent(
        api_key="test",
        base_url="https://example.invalid/v1",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        context_usage_callback=callback,
    )
    assert agent.context_usage_callback is callback
