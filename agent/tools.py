"""The three judgement calls the agent is allowed to make.

Each one builds a tight payload, hands it to whatever provider is configured
(`agent/llm.py`), and falls back to `agent/heuristics.py` if the provider
cannot answer. Callers get the same dict shape either way plus a `source`
string saying which path produced it, which is what ends up on the trace step
in the escalation packet.

What is deliberately *not* here: arithmetic. These functions never compute a
variance, never decide a tolerance, and never decide whether to escalate. They
answer questions of the form "are these the same thing" and "how would you say
this to a person".

Payloads are trimmed on purpose. `reason_duplicate` sends four fields of the
invoice and the ledger candidates, not the line items -- the question is about
identity, not content, and the agent loop resends context on every invoice.

`_judge` is also the last line of defence for the boundary's own contract.
`agent/llm.py::Provider.judge` documents "implementations must not raise" and
"a judgement that cannot be made returns Judgement.unavailable()", but that is
a promise made by each implementation, not something enforced structurally --
a provider that returns malformed data (a non-numeric confidence, a missing
field) or that raises despite the contract (a real HTTP client can surprise
you) was, before this, able to take the invoice down with it: an uncaught
exception mid-judgement propagated straight out of `process_invoice`, and a
batch eval died on the first bad response rather than reporting one more
fallback. `_judge` now catches provider exceptions itself and coerces the
answer shape (`confidence` in particular) so a garbled response degrades to
the heuristic fallback instead of crashing -- see tests/test_llm_veto_adversarial.py.
"""

import math
import os

from agent import heuristics
from agent.llm import Judgement, get_provider

_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
          "fallbacks": 0}


def get_usage():
    return dict(_usage)


def reset_usage():
    _usage.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "latency_s": 0.0, "fallbacks": 0})


def price_per_1k_inr():
    """Rupees per 1000 tokens. Not hardcoded: the provider's real pricing is not
    something this repo can assert, so it reports 0 until someone sets it."""
    return float(os.environ.get("TENSORMUX_PRICE_PER_1K_INR", "0"))


# Kept as a module attribute for callers that read it directly; prefer the
# function above, which re-reads the environment.
PRICE_PER_1K_TOKENS_INR = price_per_1k_inr()


def _judge(task, payload, fallback):
    """Ask the provider; fall back deterministically if it cannot answer.

    Two defensive layers beyond the provider's own contract, both added after
    tests/test_llm_veto_adversarial.py found they were missing:

    1. `provider.judge()` can raise despite being documented not to (a custom
       or future provider implementation, not just the two shipped here) --
       caught here and treated exactly like an unavailable Judgement, so a
       provider bug degrades one invoice to the heuristic fallback instead of
       taking down whatever batch is processing it.
    2. `judgement.data` can be present but not a dict (a provider that hands
       back a JSON list or a string is still "valid JSON", just not a usable
       answer) -- also treated as unavailable rather than trusted.
    """
    try:
        judgement = get_provider().judge(task, payload)
    except Exception:
        judgement = Judgement.unavailable()

    if judgement.llm_used:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += judgement.prompt_tokens
        _usage["completion_tokens"] += judgement.completion_tokens
        _usage["latency_s"] += judgement.latency_s

    if judgement.data is None or not isinstance(judgement.data, dict):
        _usage["fallbacks"] += 1
        result = dict(fallback())
        result["llm_used"] = False
        result["model"] = None
        result["source"] = "heuristic-fallback"
        return result

    result = dict(judgement.data)
    result["llm_used"] = judgement.llm_used
    result["model"] = judgement.model
    result["source"] = judgement.source
    return result


def _safe_confidence(value, default=0.0):
    """Coerce whatever a provider sent back for `confidence` into a float in
    [0, 1], defaulting rather than raising on anything that isn't one.

    Added after tests/test_llm_veto_adversarial.py showed a non-numeric or
    missing-but-present (None) confidence value crashed `process_invoice`
    downstream -- not on a format the code assumed but never checked, `:.2f`
    on the trace line in agent/orchestrator.py, which raises on a string or
    None rather than silently misformatting. Out-of-range numbers (a model
    saying 1.4 or -0.2) are clamped rather than rejected: they're a
    calibration problem worth noticing, not a shape problem worth crashing
    over, and clamping keeps them comparable to DUPLICATE_CONFIDENCE /
    VENDOR_NORMALIZE_CONFIDENCE without a caller needing to know this ever
    happened.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return max(0.0, min(1.0, v))


def normalize_vendor(invoice_vendor, po_vendor):
    """Are these two vendor strings the same legal entity?"""
    payload = {"invoice_vendor": invoice_vendor, "po_vendor": po_vendor}
    result = _judge("normalize_vendor", payload,
                    lambda: heuristics.normalize_vendor(invoice_vendor, po_vendor))
    result.setdefault("same_entity", False)
    result.setdefault("confidence", 0.0)
    result.setdefault("reasoning", "")
    result["confidence"] = _safe_confidence(result["confidence"])
    return result


def reason_duplicate(invoice, candidates):
    """Is this invoice a re-submission of one of these already-posted ones?

    The answer never decides whether to escalate -- see the orchestrator. It
    shapes severity, confidence and wording on an escalation that is already
    happening.
    """
    payload = {
        "invoice": {"id": invoice["id"], "vendor": invoice["vendor"],
                    "total_amount": invoice["total_amount"],
                    "invoice_date": invoice["invoice_date"]},
        "candidates": candidates,
    }
    result = _judge("reason_duplicate", payload,
                    lambda: heuristics.reason_duplicate(invoice, candidates))
    result.setdefault("is_duplicate", False)
    result.setdefault("related_invoice_id", None)
    result.setdefault("confidence", 0.0)
    result.setdefault("reasoning", "")
    result["confidence"] = _safe_confidence(result["confidence"])
    return result


def summarize_exception(invoice, po, grn, findings):
    """Plain-English summary and a proposed resolution for the escalation packet."""
    payload = {"invoice": invoice, "po": po, "grn": grn, "findings": findings}
    result = _judge("summarize_exception", payload,
                    lambda: heuristics.summarize_exception(invoice, po, grn, findings))
    fallback = heuristics.summarize_exception(invoice, po, grn, findings)
    # A provider that answers with a missing prose field would otherwise put a
    # KeyError between the clerk and the invoice; fill from the deterministic
    # summary rather than shipping a packet with a hole in it.
    result.setdefault("summary", fallback["summary"])
    result.setdefault("proposed_resolution", fallback["proposed_resolution"])
    result.setdefault("confidence", 0.5)
    result["confidence"] = _safe_confidence(result["confidence"], default=0.5)
    return result
