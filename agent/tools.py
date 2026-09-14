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
"""

import os

from agent import heuristics
from agent.llm import get_provider

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
    """Ask the provider; fall back deterministically if it cannot answer."""
    judgement = get_provider().judge(task, payload)

    if judgement.llm_used:
        _usage["calls"] += 1
        _usage["prompt_tokens"] += judgement.prompt_tokens
        _usage["completion_tokens"] += judgement.completion_tokens
        _usage["latency_s"] += judgement.latency_s

    if judgement.data is None:
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


def normalize_vendor(invoice_vendor, po_vendor):
    """Are these two vendor strings the same legal entity?"""
    payload = {"invoice_vendor": invoice_vendor, "po_vendor": po_vendor}
    result = _judge("normalize_vendor", payload,
                    lambda: heuristics.normalize_vendor(invoice_vendor, po_vendor))
    result.setdefault("same_entity", False)
    result.setdefault("confidence", 0.0)
    result.setdefault("reasoning", "")
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
    return result
