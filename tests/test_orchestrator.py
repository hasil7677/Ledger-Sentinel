"""agent/orchestrator.py end to end: process_invoice wired up with a real
Ledger, a VendorMaster, and an employee master, run through the stub LLM
provider. These are the tests that exercise the new detectors (vendor bank
change, new-vendor hold, employee collision, approval threshold, split
invoicing) as the orchestrator actually calls them, not just as unit tests of
core/anomalies.py in isolation.
"""

import pytest

from agent.orchestrator import process_invoice
from core.ledger import Ledger
from core.vendor_master import VendorMaster
from tests.conftest import days_after, make_clean_triple, make_grn, make_invoice, make_po, sample_employees


@pytest.fixture
def vendor_master():
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-001", canonical_name="Acme Corp", bank_account="111122223333",
                 bank_ifsc="HDFC0001001", remit_to_name="Acme Corp", contact_email="accounts@acme.example",
                 onboarded_date="2020-01-01", aliases=["Acme Corporation Pvt Ltd"])
    return vm


@pytest.fixture
def employees(sample_employees):
    return sample_employees


def _run(invoice, po, grn, ledger, vendor_master=None, employees_by_id=None):
    return process_invoice(invoice, {po["id"]: po}, {po["id"]: grn}, ledger,
                           vendor_master=vendor_master, employees_by_id=employees_by_id)


# --- baseline: clean invoice, no vendor master or employees supplied at all -----

def test_clean_invoice_auto_clears(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple()
    result = _run(invoice, po, grn, ledger)
    assert result["decision"] == "auto_clear"
    assert len(ledger.all()) == 1


def test_quantity_variance_out_of_tolerance_escalates(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(qty=50)
    grn["lines"][0]["qty_received"] = 47  # short by 3, outside the 1-unit tolerance
    result = _run(invoice, po, grn, ledger)
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "quantity_variance"


def test_missing_po_escalates_without_touching_vendor_or_approval_checks(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple()
    result = process_invoice(invoice, {}, {}, ledger)  # PO/GRN maps empty -> no PO on file
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "missing_po"


# --- vendor-master: bank detail change -------------------------------------------

def test_unverified_bank_change_holds_an_otherwise_clean_invoice(tmp_path, vendor_master, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    vendor_master.record_change("VEND-001", "bank_account", "999900001111", effective_at="2026-07-05",
                                actor="email_request", channel="email")
    po, grn, invoice = make_clean_triple(vendor="Acme Corp", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vendor_master, employees)

    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "vendor_bank_detail_change"
    pending = result["packet"]["evidence"]["vendor_master"]["pending_changes"]
    assert pending[0]["new_value"].startswith("*")  # masked, never the raw account number


def test_verified_bank_change_no_longer_holds_the_invoice(tmp_path, vendor_master, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    change = vendor_master.record_change("VEND-001", "bank_account", "999900001111",
                                         effective_at="2026-07-05", actor="email_request", channel="email")
    vendor_master.record_verification("VEND-001", verifies_seq=change["seq"], actor="ap_clerk_1",
                                      effective_at="2026-07-06", channel="phone_callback")
    po, grn, invoice = make_clean_triple(vendor="Acme Corp", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vendor_master, employees)

    assert result["decision"] == "auto_clear"


def test_bank_change_effective_after_invoice_date_does_not_hold_it(tmp_path, vendor_master, employees):
    """A change that hasn't happened yet, relative to this invoice, shouldn't
    retroactively hold it -- pending_payment_changes(as_of=...) enforces this."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    vendor_master.record_change("VEND-001", "bank_account", "999900001111",
                                effective_at="2026-08-01", actor="email_request", channel="email")
    po, grn, invoice = make_clean_triple(vendor="Acme Corp", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vendor_master, employees)
    assert result["decision"] == "auto_clear"


# --- vendor-master: new vendor first invoice -------------------------------------

def test_first_invoice_from_recently_onboarded_vendor_escalates(tmp_path, employees):
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-002", canonical_name="Fresh Supplies Co", bank_account="1",
                 bank_ifsc="X", remit_to_name="Fresh Supplies Co", contact_email="x@fresh.example",
                 onboarded_date="2026-06-20")  # 20 days before the invoice
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(vendor="Fresh Supplies Co", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vm, employees)
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "new_vendor_first_invoice"


def test_second_invoice_after_new_vendor_window_auto_clears(tmp_path, employees):
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-002", canonical_name="Fresh Supplies Co", bank_account="1",
                 bank_ifsc="X", remit_to_name="Fresh Supplies Co", contact_email="x@fresh.example",
                 onboarded_date="2026-01-01")  # long before the new-vendor window
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(vendor="Fresh Supplies Co", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vm, employees)
    assert result["decision"] == "auto_clear"


# --- vendor-master: employee collision -------------------------------------------

def test_vendor_bank_account_matching_employee_escalates(tmp_path, employees):
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-003", canonical_name="Shell Trading Co",
                 bank_account=employees["EMP-007"]["bank_account"],  # the collision
                 bank_ifsc="X", remit_to_name="Shell Trading Co", contact_email="x@shell.example",
                 onboarded_date="2020-01-01")
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(vendor="Shell Trading Co", inv_date="2026-07-10")

    result = _run(invoice, po, grn, ledger, vm, employees)
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "vendor_employee_collision"
    assert result["packet"]["evidence"]["vendor_master"]["employee_collision"]["employee_id"] == "EMP-007"


# --- approval policy: single invoice over limit ----------------------------------

def test_invoice_over_requesters_limit_escalates(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(unit_price=10_000.0, qty=20, requester_id="EMP-002")  # Rs 236,000 > 75k
    result = _run(invoice, po, grn, ledger, employees_by_id=employees)
    assert result["decision"] == "escalate"
    assert result["packet"]["exception_type"] == "approval_threshold_exceeded"


def test_invoice_under_requesters_limit_auto_clears(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(unit_price=100.0, qty=10, requester_id="EMP-002")  # well under 75k
    result = _run(invoice, po, grn, ledger, employees_by_id=employees)
    assert result["decision"] == "auto_clear"


def test_default_high_limit_requester_never_trips_approval_check(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(unit_price=10_000.0, qty=20, requester_id="EMP-001")  # same amount, director
    result = _run(invoice, po, grn, ledger, employees_by_id=employees)
    assert result["decision"] == "auto_clear"


# --- approval policy: split invoicing --------------------------------------------

def test_first_invoice_of_a_split_pair_auto_clears_alone(tmp_path, employees):
    """Documented, real limitation: detection is retrospective. The first of
    two invoices that will together exceed the limit has no history to be
    compared against yet, so it clears on its own."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po1, grn1, inv1 = make_clean_triple(unit_price=5_000.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-1", grn_id="GRN-1", inv_id="INV-1", inv_date="2026-07-01")
    result1 = _run(inv1, po1, grn1, ledger, employees_by_id=employees)
    assert result1["decision"] == "auto_clear"


def test_second_invoice_of_a_split_pair_escalates_as_threshold_splitting(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    # Deliberately different amounts for the two members: if they matched
    # exactly, the ledger's duplicate near-match (vendor + amount +/- Rs1)
    # would also fire and, being the same "high" severity, could win the
    # primary-finding tie-break -- masking whether the split check itself
    # actually ran. Distinct amounts isolate the thing this test checks.
    po1, grn1, inv1 = make_clean_triple(unit_price=4_700.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-1", grn_id="GRN-1", inv_id="INV-1", inv_date="2026-07-01")
    _run(inv1, po1, grn1, ledger, employees_by_id=employees)  # Rs 55,460, posts, clears

    po2, grn2, inv2 = make_clean_triple(unit_price=6_300.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-2", grn_id="GRN-2", inv_id="INV-2", inv_date="2026-07-05")
    result2 = _run(inv2, po2, grn2, ledger, employees_by_id=employees)  # Rs 74,340; sum Rs 129,800 > 75k limit

    assert result2["decision"] == "escalate"
    assert result2["packet"]["exception_type"] == "threshold_splitting"
    members = {m["invoice_id"] for m in result2["packet"]["evidence"]["approval"]["members"]}
    assert members == {"INV-1", "INV-2"}


def test_split_check_ignores_a_different_requester_at_the_same_vendor(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po1, grn1, inv1 = make_clean_triple(unit_price=4_700.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-1", grn_id="GRN-1", inv_id="INV-1", inv_date="2026-07-01")
    _run(inv1, po1, grn1, ledger, employees_by_id=employees)

    po2, grn2, inv2 = make_clean_triple(unit_price=6_300.0, qty=10, requester_id="EMP-007",  # different requester
                                        po_id="PO-2", grn_id="GRN-2", inv_id="INV-2", inv_date="2026-07-05")
    result2 = _run(inv2, po2, grn2, ledger, employees_by_id=employees)
    assert result2["decision"] == "auto_clear"


def test_split_check_outside_window_does_not_group(tmp_path, employees):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po1, grn1, inv1 = make_clean_triple(unit_price=4_700.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-1", grn_id="GRN-1", inv_id="INV-1", inv_date="2026-07-01")
    _run(inv1, po1, grn1, ledger, employees_by_id=employees)

    far_later = days_after("2026-07-01", 30)  # split window is 14 days
    po2, grn2, inv2 = make_clean_triple(unit_price=6_300.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-2", grn_id="GRN-2", inv_id="INV-2", inv_date=far_later)
    result2 = _run(inv2, po2, grn2, ledger, employees_by_id=employees)
    assert result2["decision"] == "auto_clear"


# --- vendor-master lookup miss doesn't break the pipeline ------------------------

def test_vendor_not_in_master_skips_vendor_master_checks_without_erroring(tmp_path, employees):
    vm = VendorMaster.empty()  # no vendors registered at all
    ledger = Ledger(str(tmp_path / "ledger.json"))
    po, grn, invoice = make_clean_triple(vendor="Totally Unknown Vendor")
    result = _run(invoice, po, grn, ledger, vm, employees)
    assert result["decision"] == "auto_clear"
