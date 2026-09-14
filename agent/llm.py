"""The LLM boundary.

Everything the agent asks a model is funnelled through one interface with
exactly one method. There are three implementations of it:

  OpenAICompatibleProvider  a real chat-completions endpoint (TensorMux, or
                            anything else that speaks the OpenAI API)
  StubProvider              deterministic, no network, answers from
                            agent/heuristics.py, scriptable per task
  DisabledProvider          answers nothing at all, forcing callers onto the
                            heuristic fallback

Why this exists at all: before, `agent/tools.py` reached for an API key at
import time and silently degraded to string-similarity if it did not find one.
That is fine for a demo and useless for verification -- there was no way to
run the pipeline against a *known* model verdict, so no way to test the one
behaviour the project claims to guarantee (a ledger near-match escalates even
when the model says it is not a duplicate). With a provider you can inject,
that claim becomes a test: script the stub to insist it is not a duplicate,
and assert the invoice escalates anyway.

Selection is by config, not by whether a key happens to be lying around:

  LEDGER_SENTINEL_LLM=auto    (default) openai if TENSORMUX_API_KEY is set,
                              otherwise stub
  LEDGER_SENTINEL_LLM=openai  real endpoint; hard error if no key
  LEDGER_SENTINEL_LLM=stub    deterministic stub, never touches the network
  LEDGER_SENTINEL_LLM=off     no provider; every judgement takes the fallback

`auto` is the default rather than `stub` so that setting a key is still all
you have to do. But `stub` is what CI and the test suite pin explicitly, so a
key present in someone's environment can never change a test outcome.
"""

import json
import os
import time

from agent import heuristics

TASKS = ("normalize_vendor", "reason_duplicate", "summarize_exception")

SYSTEM_PROMPTS = {
    "normalize_vendor": (
        "You are an accounts-payable analyst. Decide whether two vendor name "
        "strings refer to the same legal entity (legal-suffix differences, "
        "punctuation, capitalisation, abbreviation). Respond as JSON: "
        '{"same_entity": bool, "confidence": float 0-1, "reasoning": str}.'
    ),
    "reason_duplicate": (
        "You are an accounts-payable analyst checking for duplicate invoice "
        "submissions. You are given a new invoice and one or more already-posted "
        "invoices with the same vendor and a very close amount. Decide if the new "
        "invoice is a duplicate of one of them (common pattern: same amount/vendor/date, "
        "invoice number modified with a suffix or transposed digits). Respond as JSON: "
        '{"is_duplicate": bool, "related_invoice_id": str or null, "confidence": float 0-1, '
        '"reasoning": str}.'
    ),
    "summarize_exception": (
        "You are an accounts-payable analyst writing an exception summary for a "
        "clerk who has 10 seconds to read it. Be concrete: name the amounts and "
        "fields involved, no jargon, no JSON-speak in the prose fields. Respond as JSON: "
        '{"summary": str, "proposed_resolution": str, "confidence": float 0-1}. '
        "Be conservative: only propose auto-approve-style resolutions when the evidence "
        "is unambiguous; otherwise propose the safer action (reject / route to approver)."
    ),
}


class Judgement:
    """One answer to one judgement question, plus how it was obtained.

    `data` is None when the provider could not answer; the caller is then
    responsible for falling back. `llm_used` is True only when a real remote
    model produced the answer -- the stub sets it False, because a metric that
    counts stub answers as model answers is a lie in a report.
    """

    __slots__ = ("data", "model", "llm_used", "source", "prompt_tokens",
                 "completion_tokens", "latency_s")

    def __init__(self, data, model, llm_used, source,
                 prompt_tokens=0, completion_tokens=0, latency_s=0.0):
        self.data = data
        self.model = model
        self.llm_used = llm_used
        self.source = source
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_s = latency_s

    @classmethod
    def unavailable(cls, model=None, reason="provider returned no answer"):
        return cls(None, model, False, "unavailable")

    def __repr__(self):
        return f"<Judgement source={self.source} data={'set' if self.data else 'none'}>"


class Provider:
    """The whole interface. One method.

    `judge(task, payload)` takes a task name from TASKS and a JSON-serialisable
    payload, and returns a Judgement. Implementations must not raise: a
    judgement that cannot be made returns Judgement.unavailable() so the caller
    can fall back deterministically.
    """

    name = "provider"
    model = None

    def judge(self, task, payload):
        raise NotImplementedError


class DisabledProvider(Provider):
    """Answers nothing. Every caller takes the heuristic fallback.

    Useful as a control: run the eval under `off` and under `stub` and the
    difference is exactly the contribution of model judgement.
    """

    name = "off"

    def judge(self, task, payload):
        return Judgement.unavailable()


class StubProvider(Provider):
    """A deterministic model that never leaves the process.

    Default behaviour is agent/heuristics.py. Tests override individual tasks
    to script a specific verdict:

        StubProvider(overrides={"reason_duplicate": {"is_duplicate": False,
                                                     "confidence": 0.99}})

    An override may be a dict (returned as-is, merged over the heuristic answer
    so a test only has to state the fields it cares about) or a callable taking
    the payload and returning a dict. Returning None from a callable makes that
    task unavailable, which is how you test the fallback path.
    """

    name = "stub"

    def __init__(self, model="stub-judge", overrides=None, latency_s=0.0):
        self.model = model
        self.overrides = dict(overrides or {})
        self._latency_s = latency_s
        self.calls = []

    def judge(self, task, payload):
        if task not in TASKS:
            raise ValueError(f"unknown judgement task {task!r}")
        self.calls.append((task, payload))

        base = self._heuristic(task, payload)
        override = self.overrides.get(task)
        if override is None:
            data = base
        elif callable(override):
            data = override(payload)
            if data is None:
                return Judgement.unavailable(self.model)
        else:
            data = {**base, **override}

        return Judgement(
            data=dict(data),
            model=self.model,
            llm_used=False,
            source=f"stub:{self.model}",
            latency_s=self._latency_s,
        )

    @staticmethod
    def _heuristic(task, payload):
        if task == "normalize_vendor":
            return heuristics.normalize_vendor(payload["invoice_vendor"], payload["po_vendor"])
        if task == "reason_duplicate":
            return heuristics.reason_duplicate(payload["invoice"], payload["candidates"])
        return heuristics.summarize_exception(
            payload["invoice"], payload.get("po"), payload.get("grn"), payload["findings"]
        )


class OpenAICompatibleProvider(Provider):
    """A real chat-completions endpoint.

    Never raises out of `judge`: a network error, a timeout, a refusal, or a
    response that is not the JSON we asked for all become an unavailable
    Judgement, and the caller falls back. An invoice must not fail to be
    processed because a model had a bad minute.
    """

    name = "openai"

    def __init__(self, api_key, base_url, model, temperature=0, timeout=30):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout)
        return self._client

    def judge(self, task, payload):
        if task not in TASKS:
            raise ValueError(f"unknown judgement task {task!r}")
        try:
            client = self._get_client()
        except Exception:
            return Judgement.unavailable(self.model)

        start = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[task]},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
        except Exception:
            return Judgement.unavailable(self.model)
        elapsed = time.monotonic() - start

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        try:
            data = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, TypeError, IndexError, AttributeError):
            data = None

        if not isinstance(data, dict):
            # The call happened and the tokens were spent, so report the usage,
            # but there is no usable answer -- the caller falls back.
            return Judgement(None, self.model, True, "unavailable",
                             prompt_tokens, completion_tokens, elapsed)

        return Judgement(data, self.model, True, f"llm:{self.model}",
                         prompt_tokens, completion_tokens, elapsed)


# --- configuration -----------------------------------------------------------

DEFAULT_BASE_URL = "https://api.tensormux.com/v1"
DEFAULT_MODEL = "glm-4-7-flash"

_provider = None


def load_dotenv(path=None):
    """Populate os.environ from a .env file at the repo root, without adding a
    python-dotenv dependency. Existing environment variables always win, so a
    stale .env cannot override what CI or a test set deliberately."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def build_provider(mode=None, env=None):
    """Construct a provider from configuration without installing it globally."""
    env = os.environ if env is None else env
    mode = (mode or env.get("LEDGER_SENTINEL_LLM") or "auto").strip().lower()
    api_key = env.get("TENSORMUX_API_KEY") or env.get("OPENAI_API_KEY")
    base_url = env.get("LEDGER_SENTINEL_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = env.get("LEDGER_SENTINEL_LLM_MODEL", DEFAULT_MODEL)

    if mode == "off":
        return DisabledProvider()
    if mode == "stub":
        return StubProvider()
    if mode == "openai":
        if not api_key:
            raise RuntimeError(
                "LEDGER_SENTINEL_LLM=openai but neither TENSORMUX_API_KEY nor "
                "OPENAI_API_KEY is set. Use LEDGER_SENTINEL_LLM=stub to run offline."
            )
        return OpenAICompatibleProvider(api_key, base_url, model)
    if mode == "auto":
        if api_key:
            return OpenAICompatibleProvider(api_key, base_url, model)
        return StubProvider()
    raise ValueError(
        f"unknown LEDGER_SENTINEL_LLM={mode!r}; expected auto, openai, stub or off"
    )


def get_provider():
    global _provider
    if _provider is None:
        load_dotenv()
        _provider = build_provider()
    return _provider


def set_provider(provider):
    """Install a provider for the process. Returns the previous one, so a test
    or a script can restore it."""
    global _provider
    previous = _provider
    _provider = provider
    return previous


def reset_provider():
    """Drop the cached provider so the next call re-reads configuration."""
    global _provider
    _provider = None


class using_provider:
    """Context manager for a scoped provider swap.

        with using_provider(StubProvider(overrides={...})):
            ...
    """

    def __init__(self, provider):
        self.provider = provider
        self._previous = None
        self._had_previous = False

    def __enter__(self):
        global _provider
        self._previous = _provider
        self._had_previous = True
        set_provider(self.provider)
        return self.provider

    def __exit__(self, *exc):
        global _provider
        _provider = self._previous
        return False
