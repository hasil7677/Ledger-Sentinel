"""The one invariant the README is proudest of: a ledger near-match on
vendor + amount + date escalates as a suspected duplicate NO MATTER WHAT the
model concludes. The model may sharpen severity, confidence and wording; it
may never suppress the escalation.

This exists because of a real bug (see README, "Design decisions worth
knowing about"): escalation used to be gated on the model agreeing it was a
duplicate, and across repeated runs the model occasionally said "not a
duplicate" for an invoice that, in fact, was one -- which would have paid it
twice. These tests fail immediately if that gate is ever reintroduced,
because they script the stub model to insist -- at high confidence -- that
the invoice is NOT a duplicate, and assert it escalates anyway.
"""

import pytest

from agent.llm import StubProvider, using_provider
from agent.orchestrator import process_invoice
from core.ledger import Ledger
from tests.conftest import days_after, make_clean_triple


def _seed_posted_original(ledger, vendor="Acme Corp", amount_lines=None):
    """Post one clean invoice to the ledger, standing in for an
    already-processed original that a later resubmission will collide with."""
    po, grn, original_invoice = make_clean_triple(
        vendor=vendor, po_id="PO-100", grn_id="GRN-100", inv_id="INV-1000",
        po_date="2026-07-01", grn_date="2026-07-03", inv_date="2026-07-06",
    )
    ledger.post(original_invoice, resolution="auto_clear")
    return original_invoice


def _resubmission(original_invoice, suffix="-A", days_later=5):
    """A near-duplicate of `original_invoice`: same vendor, same amount
    (within the ledger's Rs 1 tolerance), a suffixed invoice number, and a
    clean three-way match in its own right -- the only thing wrong with it is
    that it looks like something already on the ledger."""
    dup = dict(original_invoice)
    dup["id"] = original_invoice["id"] + suffix
    dup["invoice_date"] = days_after(original_invoice["invoice_date"], days_later)
    dup["lines"] = [dict(l) for l in original_invoice["lines"]]
    return dup


def _po_grn_for(invoice, po_id="PO-100", grn_id="GRN-100"):
    from tests.conftest import make_grn, make_po
    po = make_po(vendor=invoice["vendor"], lines=invoice["lines"], po_id=po_id, requester_id=invoice["requester_id"])
    grn = make_grn(po, grn_id=grn_id)
    return po, grn


def test_duplicate_escalates_even_when_stub_model_insists_it_is_not(tmp_path):
    """The core invariant. If someone changes orchestrator.py to gate
    suspected_duplicate's escalation on the model's verdict, this test fails:
    the stub is scripted to say 'not a duplicate' at 0.99 confidence, and the
    invoice must escalate regardless."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    stubborn_stub = StubProvider(overrides={
        "reason_duplicate": {
            "is_duplicate": False,
            "related_invoice_id": None,
            "confidence": 0.99,
            "reasoning": "The stub is confident these are unrelated invoices.",
        }
    })

    with using_provider(stubborn_stub):
        result = process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)

    assert result["decision"] == "escalate"
    assert "suspected_duplicate" in result["findings"]
    packet = result["packet"]
    assert packet is not None
    assert packet["exception_type"] == "suspected_duplicate"

    # The trace must show the rule fired despite the model's disagreement --
    # not just that it happened to escalate for some other reason.
    veto_steps = [s for s in packet["steps"] if s["action"] == "rule_escalation"
                 and "regardless of the model" in s["detail"]]
    assert veto_steps, f"expected an explicit veto-override step in the trace, got: {packet['steps']}"

    judgement_steps = [s for s in packet["steps"] if s["action"] == "judgement_duplicate"]
    assert judgement_steps and "not a duplicate" in judgement_steps[0]["detail"]


def test_duplicate_escalates_when_stub_model_agrees_too(tmp_path):
    """Complementary sanity check: the ordinary path (model agrees) also
    escalates, so the invariant test above isn't passing merely because
    everything escalates unconditionally."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    agreeable_stub = StubProvider(overrides={
        "reason_duplicate": {"is_duplicate": True, "related_invoice_id": original["id"], "confidence": 0.95},
    })
    with using_provider(agreeable_stub):
        result = process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)

    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "suspected_duplicate"
    assert result["packet"]["evidence"]["related"][0]["id"] == original["id"]


def test_no_ledger_near_match_does_not_raise_suspected_duplicate(tmp_path):
    """True-negative check: an ordinary clean invoice with nothing posted
    against it must NOT be flagged as a duplicate. Without this test, a
    detector that escalates every invoice unconditionally would still pass
    the two tests above."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(inv_id="INV-2000")

    result = process_invoice(invoice, {po["id"]: po}, {po["id"]: grn}, ledger)

    assert result["decision"] == "auto_clear"
    assert "suspected_duplicate" not in result["findings"]


def test_duplicate_veto_survives_even_at_zero_model_confidence_threshold(tmp_path):
    """Belt and braces: even if DUPLICATE_CONFIDENCE were misconfigured to 0
    (i.e. any non-zero model confidence in "not a duplicate" would trivially
    clear the old, removed gate), the ledger near-match must still escalate --
    because the escalation no longer depends on the model's confidence value
    at all. This pins down that the finding, not a confidence comparison, is
    what drives the decision."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    original = _seed_posted_original(ledger)
    dup_invoice = _resubmission(original)
    po, grn = _po_grn_for(dup_invoice)

    stub = StubProvider(overrides={
        "reason_duplicate": {"is_duplicate": False, "confidence": 0.0, "related_invoice_id": None},
    })
    with using_provider(stub):
        result = process_invoice(dup_invoice, {po["id"]: po}, {po["id"]: grn}, ledger)

    assert result["decision"] == "escalate"
