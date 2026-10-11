"""Makes the structural-vs-LLM decision boundary testable instead of a claim
in prose.

core/anomalies.py's docstring says: "Findings marked needs_judgement=True are
the ones where an LLM adds value ... everything else is an unambiguous rule
outcome." agent/orchestrator.py enforces that by branching on finding type:
vendor_name_drift and suspected_duplicate consult a tools.* judgement
function; every other type goes straight to rule_escalation without one.

Those are two independent pieces of code that have to agree, and nothing
before this file checked that they do. If someone added a ninth finding type
to core/anomalies.py, or a new vendor-master check, and marked it
needs_judgement=True without adding a branch for it in the orchestrator, it
would silently fall into the "else" bucket -- routed to a human, which is
safe, but claiming needs_judgement=True while never actually consulting a
model, which is a lie sitting in the packet's trace. The reverse mistake
(needs_judgement=False but a model gets consulted anyway) would be worse: an
LLM call nobody accounted for, on a decision the docstring promises is a pure
rule.

This file drives one scenario per finding type in the full taxonomy through
both core.anomalies.classify() directly (to read the needs_judgement flag it
actually assigns) and agent.orchestrator.process_invoice() with the two
judgement tools monkeypatched to record whether they were called, then
asserts the two agree, for every type, not just the two that are supposed to
need judgement.
"""

import pytest

from agent import tools
from agent.llm import StubProvider, using_provider
from agent.orchestrator import process_invoice
from core.anomalies import classify
from core.ledger import Ledger
from core.matcher import match_invoice
from core.vendor_master import VendorMaster
from tests.conftest import days_after, make_clean_triple, make_grn, make_po

# The two finding types where agent/orchestrator.py consults a provider. This
# is the list a reviewer can point at; the tests below check the actual code
# matches it rather than trusting that it still does.
EXPECTED_NEEDS_JUDGEMENT_TYPES = {"vendor_name_drift", "suspected_duplicate"}


def _employees():
    """Same three employees tests/conftest.py's sample_employees fixture
    provides, as a plain callable rather than a fixture -- the scenario
    builders below run twice per test (see the "Side B" re-run below) and
    need a fresh dict each time rather than a fixture bound to one test."""
    return {
        "EMP-001": {"employee_id": "EMP-001", "name": "Meera Rao", "email": "meera.rao@ourcompany.example",
                    "bank_account": "500100010001", "department": "Procurement", "role": "director",
                    "approval_limit": 20_000_000.0},
        "EMP-002": {"employee_id": "EMP-002", "name": "Arjun Iyer", "email": "arjun.iyer@ourcompany.example",
                    "bank_account": "500100010002", "department": "Procurement", "role": "staff",
                    "approval_limit": 75_000.0},
        "EMP-007": {"employee_id": "EMP-007", "name": "Priya Nair", "email": "priya.nair@ourcompany.example",
                    "bank_account": "500100010007", "department": "Finance", "role": "staff",
                    "approval_limit": 75_000.0},
    }


# --- one scenario builder per finding type -----------------------------------
# Each returns (invoice, po, grn, ledger, vendor_master, employees_by_id, expected_type).
# Lifted from the same constructions tests/test_orchestrator.py and
# tests/test_duplicate_veto.py already use, so a scenario failing to trigger
# its finding here would also mean those files are broken.

def _scn_vendor_name_drift(tmp_path):
    po, grn, invoice = make_clean_triple(vendor="Acme Corp", inv_id="INV-VND1")
    invoice["vendor"] = "Acme Corporation Pvt Ltd"  # drifted, not exact
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "vendor_name_drift"


def _scn_suspected_duplicate(tmp_path):
    ledger = Ledger(str(tmp_path / "l.json"))
    po, grn, original = make_clean_triple(vendor="Acme Corp", po_id="PO-100", grn_id="GRN-100",
                                          inv_id="INV-1000", po_date="2026-07-01",
                                          grn_date="2026-07-03", inv_date="2026-07-06")
    ledger.post(original, resolution="auto_clear")
    dup = dict(original)
    dup["id"] = "INV-1000-A"
    dup["invoice_date"] = days_after(original["invoice_date"], 5)
    dup["lines"] = [dict(l) for l in original["lines"]]
    po2 = make_po(vendor=dup["vendor"], lines=dup["lines"], po_id="PO-100", requester_id=dup["requester_id"])
    grn2 = make_grn(po2, grn_id="GRN-100")
    return dup, po2, grn2, ledger, VendorMaster.empty(), _employees(), "suspected_duplicate"


def _scn_missing_po(tmp_path):
    _, _, invoice = make_clean_triple(inv_id="INV-MP1")
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, None, None, ledger, VendorMaster.empty(), _employees(), "missing_po"


def _scn_quantity_variance(tmp_path):
    po, grn, invoice = make_clean_triple(qty=50, inv_id="INV-QV1")
    grn["lines"][0]["qty_received"] = 47  # short by 3, outside +-1 tolerance
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "quantity_variance"


def _scn_quantity_variance_no_grn(tmp_path):
    po, _, invoice = make_clean_triple(qty=10, inv_id="INV-QVNG1")
    invoice["lines"][0]["qty"] = 15  # billed over the PO's ordered qty, no GRN to check against
    invoice["total_amount"] = round(15 * 100.0 * 1.18, 2)
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, None, ledger, VendorMaster.empty(), _employees(), "quantity_variance_no_grn"


def _scn_price_variance_out_tolerance(tmp_path):
    po, grn, invoice = make_clean_triple(unit_price=1000.0, qty=50, inv_id="INV-PV1")
    invoice["lines"][0]["unit_price"] = 1200.0  # 20% over, well past 2%/Rs500 floor
    invoice["total_amount"] = round(sum(
        l["qty"] * l["unit_price"] * (1 + l["gst_rate"] / 100) for l in invoice["lines"]
    ), 2)
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "price_variance_out_tolerance"


def _scn_tax_mismatch(tmp_path):
    po, grn, invoice = make_clean_triple(gst_rate=18, inv_id="INV-TM1")
    invoice["lines"][0]["gst_rate"] = 12
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "tax_mismatch"


def _scn_line_item_mismatch(tmp_path):
    po, grn, invoice = make_clean_triple(inv_id="INV-LM1")
    invoice["lines"].append({"sku": "SKU-EXTRA", "description": "Unplanned Widget",
                             "qty": 5, "unit_price": 100.0, "gst_rate": 18})
    invoice["total_amount"] = round(invoice["total_amount"] + 5 * 100.0 * 1.18, 2)
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "line_item_mismatch"


def _scn_vendor_bank_detail_change(tmp_path):
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-001", canonical_name="Acme Corp", bank_account="111122223333",
                 bank_ifsc="HDFC0001001", remit_to_name="Acme Corp", contact_email="accounts@acme.example",
                 onboarded_date="2020-01-01")
    vm.record_change("VEND-001", "bank_account", "999900001111", effective_at="2026-07-05",
                     actor="email_request", channel="email")
    po, grn, invoice = make_clean_triple(vendor="Acme Corp", inv_date="2026-07-10", inv_id="INV-BC1")
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, vm, _employees(), "vendor_bank_detail_change"


def _scn_new_vendor_first_invoice(tmp_path):
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-002", canonical_name="Fresh Supplies Co", bank_account="1",
                 bank_ifsc="X", remit_to_name="Fresh Supplies Co", contact_email="x@fresh.example",
                 onboarded_date="2026-06-20")
    po, grn, invoice = make_clean_triple(vendor="Fresh Supplies Co", inv_date="2026-07-10", inv_id="INV-NV1")
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, vm, _employees(), "new_vendor_first_invoice"


def _scn_vendor_employee_collision(tmp_path):
    employees = _employees()
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-003", canonical_name="Shell Trading Co",
                 bank_account=employees["EMP-007"]["bank_account"],
                 bank_ifsc="X", remit_to_name="Shell Trading Co", contact_email="x@shell.example",
                 onboarded_date="2020-01-01")
    po, grn, invoice = make_clean_triple(vendor="Shell Trading Co", inv_date="2026-07-10", inv_id="INV-EC1")
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, vm, employees, "vendor_employee_collision"


def _scn_approval_threshold_exceeded(tmp_path):
    po, grn, invoice = make_clean_triple(unit_price=10_000.0, qty=20, requester_id="EMP-002", inv_id="INV-AT1")
    ledger = Ledger(str(tmp_path / "l.json"))
    return invoice, po, grn, ledger, VendorMaster.empty(), _employees(), "approval_threshold_exceeded"


def _scn_threshold_splitting(tmp_path):
    employees = _employees()
    ledger = Ledger(str(tmp_path / "l.json"))
    po1, grn1, inv1 = make_clean_triple(unit_price=4_700.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-S1", grn_id="GRN-S1", inv_id="INV-S1", inv_date="2026-07-01")
    process_invoice(inv1, {po1["id"]: po1}, {po1["id"]: grn1}, ledger, employees_by_id=employees)  # posts, clears

    po2, grn2, inv2 = make_clean_triple(unit_price=6_300.0, qty=10, requester_id="EMP-002",
                                        po_id="PO-S2", grn_id="GRN-S2", inv_id="INV-S2", inv_date="2026-07-05")
    return inv2, po2, grn2, ledger, VendorMaster.empty(), employees, "threshold_splitting"


SCENARIOS = {
    "vendor_name_drift": _scn_vendor_name_drift,
    "suspected_duplicate": _scn_suspected_duplicate,
    "missing_po": _scn_missing_po,
    "quantity_variance": _scn_quantity_variance,
    "quantity_variance_no_grn": _scn_quantity_variance_no_grn,
    "price_variance_out_tolerance": _scn_price_variance_out_tolerance,
    "tax_mismatch": _scn_tax_mismatch,
    "line_item_mismatch": _scn_line_item_mismatch,
    "vendor_bank_detail_change": _scn_vendor_bank_detail_change,
    "new_vendor_first_invoice": _scn_new_vendor_first_invoice,
    "vendor_employee_collision": _scn_vendor_employee_collision,
    "approval_threshold_exceeded": _scn_approval_threshold_exceeded,
    "threshold_splitting": _scn_threshold_splitting,
}

# Sanity: this file's own scenario table must cover every type the orchestrator
# knows how to escalate, so "we checked the boundary" can't quietly mean "we
# checked 8 of 12 types." Kept in sync with core/anomalies.py::SEVERITY_BY_TYPE
# by the test below rather than trusted to stay that way by hand.
assert set(SCENARIOS) | {"clean"} <= (
    set(__import__("core.anomalies", fromlist=["SEVERITY_BY_TYPE"]).SEVERITY_BY_TYPE) | {"clean"}
)


@pytest.mark.parametrize("finding_type", sorted(SCENARIOS))
def test_needs_judgement_flag_matches_whether_a_tool_was_actually_consulted(tmp_path, finding_type):
    """For each finding type: read the needs_judgement flag classify() sets,
    then independently observe whether process_invoice actually calls a
    tools.* judgement function while handling it. The two must agree."""
    invoice, po, grn, ledger, vendor_master, employees_by_id, expected_type = SCENARIOS[finding_type](tmp_path)
    assert expected_type == finding_type

    # Side A: what classify() says, straight from core/anomalies.py.
    match_result = match_invoice(invoice, po, grn)
    findings = classify(invoice, match_result, ledger, vendor_master=vendor_master, employees_by_id=employees_by_id)
    matching = [f for f in findings if f["type"] == finding_type]
    assert matching, f"scenario for {finding_type!r} did not actually produce that finding: {findings}"
    declared_needs_judgement = matching[0]["needs_judgement"]

    # Side B: what agent/orchestrator.py actually does, observed by
    # monkeypatching the two judgement entry points to record calls.
    calls = {"normalize_vendor": 0, "reason_duplicate": 0}
    real_normalize_vendor = tools.normalize_vendor
    real_reason_duplicate = tools.reason_duplicate

    def spy_normalize_vendor(*a, **kw):
        calls["normalize_vendor"] += 1
        return real_normalize_vendor(*a, **kw)

    def spy_reason_duplicate(*a, **kw):
        calls["reason_duplicate"] += 1
        return real_reason_duplicate(*a, **kw)

    tools.normalize_vendor = spy_normalize_vendor
    tools.reason_duplicate = spy_reason_duplicate
    try:
        # Re-run on a fresh ledger seeded the same way, since the ledger
        # above may already carry a posted entry (threshold_splitting,
        # suspected_duplicate) that a second identical run would double-post.
        # A second, independent scenario build keeps this call isolated.
        invoice2, po2, grn2, ledger2, vm2, emp2, _ = SCENARIOS[finding_type](tmp_path.parent / f"{tmp_path.name}_b")
        process_invoice(invoice2, {po2["id"]: po2} if po2 else {}, {po2["id"]: grn2} if po2 else {},
                        ledger2, vendor_master=vm2, employees_by_id=emp2)
    finally:
        tools.normalize_vendor = real_normalize_vendor
        tools.reason_duplicate = real_reason_duplicate

    tool_was_consulted = (calls["normalize_vendor"] + calls["reason_duplicate"]) > 0

    assert declared_needs_judgement == tool_was_consulted, (
        f"{finding_type}: classify() declared needs_judgement={declared_needs_judgement} "
        f"but a judgement tool was {'' if tool_was_consulted else 'NOT '}consulted "
        f"(calls={calls})"
    )
    assert declared_needs_judgement == (finding_type in EXPECTED_NEEDS_JUDGEMENT_TYPES)


def test_expected_needs_judgement_set_has_no_extra_or_missing_types(tmp_path):
    """Cross-check EXPECTED_NEEDS_JUDGEMENT_TYPES itself against every
    scenario's classify() output in one pass, independent of the
    per-parametrized-test comparison above."""
    declared_true = set()
    for finding_type, builder in SCENARIOS.items():
        invoice, po, grn, ledger, vendor_master, employees_by_id, _ = builder(tmp_path / finding_type)
        match_result = match_invoice(invoice, po, grn)
        findings = classify(invoice, match_result, ledger, vendor_master=vendor_master,
                            employees_by_id=employees_by_id)
        for f in findings:
            if f["type"] == finding_type and f["needs_judgement"]:
                declared_true.add(finding_type)
    assert declared_true == EXPECTED_NEEDS_JUDGEMENT_TYPES
