"""core/approval.py: per-requester authority limits, and split-invoice
detection -- the classic evasion of breaking one purchase into several
invoices that each individually stay under the limit.
"""

import pytest

from core.approval import (
    MIN_SPLIT_GROUP_SIZE,
    check_invoice,
    check_single_invoice,
    check_split,
    required_approval_ok,
)

REQUESTER = {"employee_id": "EMP-002", "name": "Arjun Iyer", "approval_limit": 75_000.0}


def _invoice(amount, inv_id="INV-1", vendor="Acme Corp", invoice_date="2026-07-10"):
    return {"id": inv_id, "vendor": vendor, "total_amount": amount, "invoice_date": invoice_date}


def _ledger_entry(amount, inv_id, requester_id="EMP-002", invoice_date="2026-07-05"):
    return {"invoice_id": inv_id, "total_amount": amount, "invoice_date": invoice_date, "requester_id": requester_id}


# --- single invoice over limit ------------------------------------------------

def test_required_approval_ok_boundary():
    assert required_approval_ok(75_000.0, 75_000.0) is True
    assert required_approval_ok(75_000.01, 75_000.0) is False


def test_single_invoice_exactly_at_limit_is_ok():
    assert check_single_invoice(75_000.0, REQUESTER) is None


def test_single_invoice_one_rupee_over_limit_flags():
    finding = check_single_invoice(75_000.01, REQUESTER)
    assert finding is not None
    assert finding["type"] == "approval_threshold_exceeded"
    assert finding["detail"]["over_by"] == pytest.approx(0.01, abs=1e-6)


def test_single_invoice_with_no_requester_always_flags():
    finding = check_single_invoice(100.0, None)
    assert finding is not None
    assert finding["detail"]["reason"] == "no_requester_on_record"


# --- split detection -----------------------------------------------------------

def test_split_detected_when_two_under_limit_invoices_sum_over_it():
    invoice = _invoice(50_000.0, inv_id="INV-2")
    prior = [_ledger_entry(50_000.0, "INV-1")]  # 50k + 50k = 100k > 75k limit, both individually under
    finding = check_split(invoice, REQUESTER, prior)
    assert finding is not None
    assert finding["type"] == "threshold_splitting"
    assert finding["detail"]["group_total"] == pytest.approx(100_000.0)
    assert {m["invoice_id"] for m in finding["detail"]["members"]} == {"INV-1", "INV-2"}


def test_no_split_when_sum_still_under_limit():
    invoice = _invoice(30_000.0, inv_id="INV-2")
    prior = [_ledger_entry(30_000.0, "INV-1")]  # 60k total, under 75k
    assert check_split(invoice, REQUESTER, prior) is None


def test_no_split_when_only_one_invoice_in_the_would_be_group():
    """MIN_SPLIT_GROUP_SIZE guards against calling a single invoice a
    'pattern' -- two is the minimum for 'split'."""
    invoice = _invoice(80_000.0, inv_id="INV-1")  # over the limit alone
    assert MIN_SPLIT_GROUP_SIZE == 2
    assert check_split(invoice, REQUESTER, prior_entries=[]) is None


def test_no_split_across_different_requesters():
    other_requester_entry = _ledger_entry(50_000.0, "INV-1", requester_id="EMP-999")
    invoice = _invoice(50_000.0, inv_id="INV-2")
    assert check_split(invoice, REQUESTER, [other_requester_entry]) is None


def test_no_split_when_this_invoice_alone_exceeds_the_limit():
    """A member individually over the limit is check_single_invoice's finding,
    not check_split's -- calling it "splitting" too would double-count the
    same fact under two names."""
    invoice = _invoice(80_000.0, inv_id="INV-2")  # already over 75k alone
    prior = [_ledger_entry(1_000.0, "INV-1")]
    assert check_split(invoice, REQUESTER, prior) is None


def test_no_split_when_a_prior_member_individually_exceeds_the_limit():
    invoice = _invoice(10_000.0, inv_id="INV-2")
    prior = [_ledger_entry(90_000.0, "INV-1")]  # this one alone was already over the limit
    assert check_split(invoice, REQUESTER, prior) is None


def test_split_with_no_requester_is_none():
    invoice = _invoice(50_000.0, inv_id="INV-2")
    assert check_split(invoice, None, [_ledger_entry(50_000.0, "INV-1")]) is None


def test_split_group_can_have_more_than_two_members():
    invoice = _invoice(30_000.0, inv_id="INV-3")
    prior = [_ledger_entry(30_000.0, "INV-1"), _ledger_entry(30_000.0, "INV-2")]
    finding = check_split(invoice, REQUESTER, prior)
    assert finding is not None
    assert len(finding["detail"]["members"]) == 3


# --- check_invoice wrapper -------------------------------------------------------

def test_check_invoice_reports_single_over_limit_not_also_split():
    invoice = _invoice(80_000.0, inv_id="INV-2")
    prior = [_ledger_entry(1_000.0, "INV-1")]
    findings = check_invoice(invoice, REQUESTER, prior)
    assert len(findings) == 1
    assert findings[0]["type"] == "approval_threshold_exceeded"


def test_check_invoice_reports_split_when_individually_ok():
    invoice = _invoice(50_000.0, inv_id="INV-2")
    prior = [_ledger_entry(50_000.0, "INV-1")]
    findings = check_invoice(invoice, REQUESTER, prior)
    assert len(findings) == 1
    assert findings[0]["type"] == "threshold_splitting"


def test_check_invoice_reports_nothing_when_clean():
    invoice = _invoice(10_000.0, inv_id="INV-2")
    prior = [_ledger_entry(10_000.0, "INV-1")]
    assert check_invoice(invoice, REQUESTER, prior) == []
