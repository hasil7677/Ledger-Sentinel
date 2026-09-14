"""agent/llm.py: the injectable LLM boundary. This is the module that makes
the rest of the project testable at all without a network connection or an
API key -- so it gets tested directly, not just exercised incidentally by
whatever else happens to call it.

Covers: provider selection from config, the stub's deterministic behaviour
and override mechanism, and that the real provider never raises out of
judge() -- a network error, a bad response, or a non-JSON reply must all
become an "unavailable" Judgement so the caller can fall back, not an
exception that takes the invoice down with it.
"""

import pytest

from agent import heuristics
from agent.llm import (
    DisabledProvider,
    Judgement,
    OpenAICompatibleProvider,
    StubProvider,
    build_provider,
    using_provider,
)


# --- provider selection from config ------------------------------------------

def test_mode_stub_returns_stub_provider_even_with_a_key_present(monkeypatch):
    monkeypatch.setenv("TENSORMUX_API_KEY", "sk-not-a-real-key")
    provider = build_provider(mode="stub")
    assert isinstance(provider, StubProvider)


def test_mode_off_returns_disabled_provider():
    assert isinstance(build_provider(mode="off"), DisabledProvider)


def test_mode_openai_without_key_raises():
    with pytest.raises(RuntimeError):
        build_provider(mode="openai", env={})


def test_mode_openai_with_key_builds_real_provider():
    provider = build_provider(mode="openai", env={"TENSORMUX_API_KEY": "sk-fake"})
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.api_key == "sk-fake"


def test_mode_auto_with_no_key_falls_back_to_stub():
    assert isinstance(build_provider(mode="auto", env={}), StubProvider)


def test_mode_auto_with_key_picks_openai():
    provider = build_provider(mode="auto", env={"TENSORMUX_API_KEY": "sk-fake"})
    assert isinstance(provider, OpenAICompatibleProvider)


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        build_provider(mode="carrier-pigeon", env={})


def test_default_mode_is_auto(monkeypatch):
    monkeypatch.delenv("LEDGER_SENTINEL_LLM", raising=False)
    assert isinstance(build_provider(env={}), StubProvider)


# --- StubProvider behaviour ---------------------------------------------------

def test_stub_defaults_to_heuristics_module():
    stub = StubProvider()
    payload = {"invoice_vendor": "Acme Corp", "po_vendor": "Acme Corp"}
    judgement = stub.judge("normalize_vendor", payload)
    expected = heuristics.normalize_vendor(**payload)
    assert judgement.data["same_entity"] == expected["same_entity"]
    assert judgement.data["confidence"] == expected["confidence"]
    assert judgement.llm_used is False  # stub answers are never counted as real LLM calls
    assert judgement.source == "stub:stub-judge"


def test_stub_dict_override_merges_over_heuristic_answer():
    stub = StubProvider(overrides={"normalize_vendor": {"same_entity": False, "confidence": 0.01}})
    judgement = stub.judge("normalize_vendor", {"invoice_vendor": "Acme Corp", "po_vendor": "Acme Corp"})
    assert judgement.data["same_entity"] is False
    assert judgement.data["confidence"] == 0.01
    assert "reasoning" in judgement.data  # untouched fields still come through from the heuristic


def test_stub_callable_override_receives_payload():
    seen = {}

    def override(payload):
        seen.update(payload)
        return {"is_duplicate": True, "confidence": 1.0, "related_invoice_id": payload["candidates"][0]["invoice_id"]}

    stub = StubProvider(overrides={"reason_duplicate": override})
    payload = {"invoice": {"id": "INV-2"}, "candidates": [{"invoice_id": "INV-1"}]}
    judgement = stub.judge("reason_duplicate", payload)
    assert judgement.data["related_invoice_id"] == "INV-1"
    assert seen == payload


def test_stub_callable_override_returning_none_is_unavailable():
    stub = StubProvider(overrides={"reason_duplicate": lambda payload: None})
    judgement = stub.judge("reason_duplicate", {"invoice": {"id": "INV-1"}, "candidates": []})
    assert judgement.data is None
    assert judgement.source == "unavailable"


def test_stub_records_every_call():
    stub = StubProvider()
    stub.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    stub.judge("normalize_vendor", {"invoice_vendor": "C", "po_vendor": "D"})
    assert len(stub.calls) == 2


def test_stub_rejects_unknown_task():
    with pytest.raises(ValueError):
        StubProvider().judge("not_a_real_task", {})


def test_disabled_provider_is_always_unavailable():
    disabled = DisabledProvider()
    judgement = disabled.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data is None
    assert judgement.llm_used is False


# --- using_provider scoping ----------------------------------------------------

def test_using_provider_restores_previous_on_exit():
    from agent import llm as llm_module

    llm_module.set_provider(DisabledProvider())
    outer = llm_module.get_provider()
    with using_provider(StubProvider()) as inner:
        assert llm_module.get_provider() is inner
    assert llm_module.get_provider() is outer


# --- OpenAICompatibleProvider never raises out of judge() ---------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content, usage=True):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage() if usage else None


class _FakeCompletions:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises

    def create(self, **kwargs):
        if self._raises:
            raise self._raises
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, response=None, raises=None):
        self.chat = _FakeChat(_FakeCompletions(response=response, raises=raises))


def _provider_with_fake_client(client):
    provider = OpenAICompatibleProvider(api_key="sk-fake", base_url="https://example.invalid/v1", model="fake-model")
    provider._client = client
    return provider


def test_openai_provider_returns_unavailable_when_the_call_raises():
    provider = _provider_with_fake_client(_FakeClient(raises=ConnectionError("network is down")))
    judgement = provider.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data is None
    assert judgement.source == "unavailable"


def test_openai_provider_returns_unavailable_on_non_json_content():
    provider = _provider_with_fake_client(_FakeClient(response=_FakeResponse("not json at all")))
    judgement = provider.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data is None
    assert judgement.source == "unavailable"
    # the call still happened and tokens were spent -- that must be visible
    # even though there's no usable answer, or usage accounting would undercount.
    assert judgement.llm_used is True
    assert judgement.prompt_tokens == 10


def test_openai_provider_returns_unavailable_when_json_is_not_an_object():
    provider = _provider_with_fake_client(_FakeClient(response=_FakeResponse('["a", "list", "not", "a", "dict"]')))
    judgement = provider.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data is None


def test_openai_provider_returns_data_on_valid_json_object():
    payload = '{"same_entity": true, "confidence": 0.9, "reasoning": "close enough"}'
    provider = _provider_with_fake_client(_FakeClient(response=_FakeResponse(payload)))
    judgement = provider.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data == {"same_entity": True, "confidence": 0.9, "reasoning": "close enough"}
    assert judgement.llm_used is True
    assert judgement.source == "llm:fake-model"


def test_openai_provider_client_construction_failure_is_unavailable():
    provider = OpenAICompatibleProvider(api_key="sk-fake", base_url="https://example.invalid/v1", model="fake-model")

    def boom():
        raise RuntimeError("no network")

    provider._get_client = boom
    judgement = provider.judge("normalize_vendor", {"invoice_vendor": "A", "po_vendor": "B"})
    assert judgement.data is None
