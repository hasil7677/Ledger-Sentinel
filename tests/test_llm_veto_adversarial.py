"""The single most important invariant in this project, attacked rather than
merely confirmed: a ledger near-match on vendor + amount + date escalates as
a suspected duplicate NO MATTER WHAT the model returns -- not just when it
returns a well-formed "not a duplicate", which is the only case
tests/test_duplicate_veto.py exercises.

That file proves the happy-path shape of the invariant (a *valid* JSON
response disagreeing with the escalation still escalates). This file goes
looking for the ways a real, live LLM endpoint could violate its own
contract -- malformed confidence values, an exception raised mid-call despite
agent/llm.py's Provider.judge() being documented not to raise, and
technically-valid-but-nonsensical JSON -- and checks the *same* invariant
holds under every one of them.

Building this file found two real crashes (see agent/tools.py::_judge and
::_safe_confidence for the fix):

  1. A non-numeric or `None` `confidence` value crashed `process_invoice`
     downstream, in agent/orchestrator.py's trace line
     (`f"...confidence {result.get('confidence', 0):.2f}..."`), which raises
     on `:.2f` formatting of a string or None rather than misformatting it.
     For a batch eval, that is not "fail safe" or "fail open" -- it is
     "fail total": the whole run dies on the first bad response instead of
     falling back for that one invoice.
  2. A provider that raises out of `judge()` (violating its own documented
     contract, e.g. an HTTP client hitting an edge case nobody wrote a
     try/except for) propagated straight out of `process_invoice` with the
     same effect.

Both are fixed at the boundary in `agent/tools.py::_judge`, which now catches
provider exceptions and coerces the answer shape before anything downstream
sees it. Every test below passed against the pre-fix code by accident for
some inputs (the ones that survived `:.2f` formatting, e.g. `999` or `-5`,
which are floats in disguise) and crashed for others (a bare string or
`None`) -- which is exactly the kind of thing "confirming the happy path"
would have missed.
"""

import math

import pytest

from agent.llm import StubProvider, using_provider
from agent.orchestrator import process_invoice
from core.ledger import Ledger
from tests.conftest import days_after, make_clean_triple, make_grn, make_po


def _seed_posted_original(ledger, vendor="Acme Corp"):
    po, grn, original_invoice = make_clean_triple(
        vendor=vendor, po_id="PO-100", grn_id="GRN-100", inv_id="INV-1000",
        po_date="2026-07-01", grn_date="2026-07-03", inv_date="2026-07-06",
    )
    ledger.post(original_invoice, resolution="auto_clear")
    return original_invoice


def _resubmission(original_invoice, suffix="-A", days_later=5):
    dup = dict(original_invoice)
    dup["id"] = original_invoice["id"] + suffix
    dup["invoice_date"] = days_after(original_invoice["invoice_date"], days_later)
    dup["lines"] = [dict(l) for l in original_invoice["lines"]]
    return dup


def _po_grn_for(invoice, po_id="PO-100", grn_id="GRN-100"):
    po = make_po(vendor=invoice["vendor"], lines=invoice["lines"], po_id=po_id, requester_id=invoice["requester_id"])
    grn = make_grn(po, grn_id=grn_id)
    return po, grn


def _escalates_as_duplicate_under(tmp_path, override):
    """Shared drive: post an original, resubmit a near-duplicate, run it
    through process_invoice with `reason_duplicate` scripted to `override`
    (a dict merged over the heuristic answer, or a callable -- see
    agent/llm.py::StubProvider), and return the result. Never lets an
    exception escape uncaught -- the test asserting on the result is what
    decides pass/fail, not this helper.
    """
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    stub = StubProvider(overrides={"reason_duplicate": override})
    with using_provider(stub):
        return process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)


# --- malformed confidence values ---------------------------------------------

MALFORMED_CONFIDENCE_CASES = [
    ("non_numeric_string", {"is_duplicate": False, "confidence": "high", "related_invoice_id": None}),
    ("none", {"is_duplicate": False, "confidence": None, "related_invoice_id": None}),
    ("nan", {"is_duplicate": False, "confidence": float("nan"), "related_invoice_id": None}),
    ("positive_infinity", {"is_duplicate": False, "confidence": float("inf"), "related_invoice_id": None}),
    ("negative_infinity", {"is_duplicate": False, "confidence": float("-inf"), "related_invoice_id": None}),
    ("above_range", {"is_duplicate": False, "confidence": 999, "related_invoice_id": None}),
    ("below_range", {"is_duplicate": False, "confidence": -5, "related_invoice_id": None}),
    ("boolean_true", {"is_duplicate": False, "confidence": True, "related_invoice_id": None}),
    ("empty_string", {"is_duplicate": False, "confidence": "", "related_invoice_id": None}),
    ("list_instead_of_number", {"is_duplicate": False, "confidence": [0.99], "related_invoice_id": None}),
]


@pytest.mark.parametrize("label,override", MALFORMED_CONFIDENCE_CASES, ids=[c[0] for c in MALFORMED_CONFIDENCE_CASES])
def test_malformed_confidence_still_escalates_and_does_not_crash(tmp_path, label, override):
    result = _escalates_as_duplicate_under(tmp_path, override)
    assert result["decision"] == "escalate", f"{label}: expected escalate, got {result['decision']}"
    assert result["packet"]["exception_type"] == "suspected_duplicate"
    # The packet's confidence field is read by the review UI and must itself
    # be a real, in-range number -- a malformed model confidence must not
    # leak into what a clerk sees.
    conf = result["packet"]["confidence"]
    assert isinstance(conf, (int, float)) and not isinstance(conf, bool)
    assert not math.isnan(conf)
    assert 0.0 <= conf <= 1.0


# --- technically-valid-but-nonsensical responses -----------------------------

def test_is_duplicate_as_truthy_string_still_escalates(tmp_path):
    """A provider that answers `"is_duplicate": "False"` (a non-empty string,
    truthy in Python) instead of the boolean `false` the system prompt asks
    for is valid JSON and a plausible model slip. It must not flip the
    decision even though `bool("False")` is `True`."""
    result = _escalates_as_duplicate_under(
        tmp_path, {"is_duplicate": "False", "confidence": 0.1, "related_invoice_id": None}
    )
    assert result["decision"] == "escalate"


def test_missing_fields_entirely_still_escalates(tmp_path):
    """An empty dict: every field the system prompt promises is absent. Valid
    JSON, valid dict, answers nothing."""
    result = _escalates_as_duplicate_under(tmp_path, {})
    assert result["decision"] == "escalate"


def test_extra_unexpected_fields_are_ignored_not_fatal(tmp_path):
    result = _escalates_as_duplicate_under(tmp_path, {
        "is_duplicate": False, "confidence": 0.9, "related_invoice_id": None,
        "unexpected_field": {"nested": ["garbage", 1, None]},
    })
    assert result["decision"] == "escalate"


def test_related_invoice_id_of_wrong_type_still_escalates(tmp_path):
    """related_invoice_id is documented as `str or null`; a provider handing
    back a number or a list is still nonsensical-but-valid JSON."""
    result = _escalates_as_duplicate_under(
        tmp_path, {"is_duplicate": False, "confidence": 0.9, "related_invoice_id": 12345}
    )
    assert result["decision"] == "escalate"


def test_provider_returns_a_list_instead_of_an_object(tmp_path):
    """Valid JSON, not a JSON *object* -- the shape OpenAICompatibleProvider
    itself already rejects (see tests/test_llm_boundary.py), exercised here
    through the full orchestrator path instead of the provider in isolation."""
    result = _escalates_as_duplicate_under(tmp_path, lambda payload: ["not", "a", "dict"])
    assert result["decision"] == "escalate"


# --- the provider raises mid-judgement, despite the documented contract -----

def test_provider_exception_mid_judgement_still_escalates_not_crashes(tmp_path):
    """agent/llm.py::Provider documents "Implementations must not raise" --
    this simulates one that does anyway (a real HTTP client can surprise
    you), and checks the *caller* enforces the contract defensively rather
    than trusting it. Before the agent/tools.py::_judge fix, this propagated
    an uncaught TimeoutError out of process_invoice."""
    def boom(payload):
        raise TimeoutError("simulated provider timeout mid-judgement")

    result = _escalates_as_duplicate_under(tmp_path, boom)
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "suspected_duplicate"


def test_provider_raises_a_non_standard_exception_type(tmp_path):
    """Not every bad client raises a builtin exception type -- make sure the
    catch in _judge isn't accidentally scoped to a specific exception class."""
    class WeirdProviderError(Exception):
        pass

    def boom(payload):
        raise WeirdProviderError("the endpoint returned something the client couldn't even parse as an error")

    result = _escalates_as_duplicate_under(tmp_path, boom)
    assert result["decision"] == "escalate"


def test_provider_hangs_then_returns_scripted_latency_still_escalates(tmp_path):
    """Not a real hang (tests must stay fast and offline) -- a stub with
    latency_s set models the shape of a slow-but-eventually-successful call.
    The orchestrator must not treat "slow" any differently from "fast" when
    deciding whether to escalate."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    slow_stub = StubProvider(
        overrides={"reason_duplicate": {"is_duplicate": False, "confidence": 0.99, "related_invoice_id": None}},
        latency_s=5.0,
    )
    with using_provider(slow_stub):
        result = process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)
    assert result["decision"] == "escalate"


# --- the veto survives even when EVERY judgement task is simultaneously broken --

def test_reason_duplicate_and_summarize_exception_both_broken_at_once(tmp_path):
    """Worst case for this scenario: not just reason_duplicate but
    summarize_exception (used to build the packet's prose, called once the
    escalation is already decided) is also malformed. The duplicate veto must
    not depend on summarize_exception succeeding either -- a broken summary
    should degrade the packet's wording, never the decision."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    stub = StubProvider(overrides={
        "reason_duplicate": {"is_duplicate": False, "confidence": "very sure", "related_invoice_id": None},
        "summarize_exception": lambda payload: (_ for _ in ()).throw(RuntimeError("summariser is down")),
    })
    with using_provider(stub):
        result = process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)

    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "suspected_duplicate"
    # summarize_exception fell back to the heuristic rather than crashing;
    # the packet must still carry usable prose.
    assert result["packet"]["summary"]
    assert result["packet"]["proposed_resolution"]


def test_usage_fallback_counter_increments_on_provider_exception(tmp_path):
    """The fallback path must be visible in the usage counters agent/tools.py
    tracks, not just silently taken -- otherwise a provider that is
    permanently broken looks, from the metrics, identical to one performing
    perfectly."""
    from agent import tools

    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    def boom(payload):
        raise ConnectionError("simulated network failure")

    stub = StubProvider(overrides={"reason_duplicate": boom})
    tools.reset_usage()
    with using_provider(stub):
        process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)
    usage = tools.get_usage()
    assert usage["fallbacks"] >= 1
