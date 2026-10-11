"""core/matcher.py: the deterministic three-way match, and specifically every
tolerance boundary the README claims (2% / Rs 500 on price, 1 unit on
quantity, and the "no tolerance at all" rule on GST rate). Each boundary gets
a test on both sides of it -- a test suite that only tests the middle of a
range would pass even if someone flipped a `<` to `<=` at the edge.
"""

import pytest

from core.matcher import match_invoice, DEFAULT_TOLERANCES
from tests.conftest import make_grn, make_invoice, make_line, make_po


def _variance(result, kind):
    matches = [v for v in result["variances"] if v["type"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind} variance, got {matches}"
    return matches[0]


# --- price: percentage clause -----------------------------------------------
# Chosen so the Rs 500 floor is clearly NOT the reason a case passes or fails:
# qty x price is large enough that line_impact is always well above 500.

def _price_case(pct_delta):
    po = make_po(lines=[make_line(unit_price=1000.0, qty=100, gst_rate=18)])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(unit_price=1000.0 * (1 + pct_delta), qty=100, gst_rate=18)])
    return match_invoice(invoice, po, grn)


def test_price_percentage_exactly_at_tolerance_clears():
    result = _price_case(DEFAULT_TOLERANCES["price_pct"])  # exactly 2.00%
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is True
    assert result["clean"] is True


def test_price_percentage_just_over_tolerance_escalates():
    result = _price_case(DEFAULT_TOLERANCES["price_pct"] + 0.0001)  # 2.01%
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is False
    assert result["clean"] is False


def test_price_percentage_just_under_tolerance_clears():
    result = _price_case(DEFAULT_TOLERANCES["price_pct"] - 0.0001)  # 1.99%
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is True


# --- price: absolute (Rs 500) floor -----------------------------------------
# Chosen so the percentage clause is clearly NOT satisfied (delta_pct always
# far over 2%): qty=1 isolates the line_impact = price_delta * 1 = price_delta,
# so the Rs 500 floor is being tested in isolation from the percentage clause.

def _price_floor_case(impact):
    po = make_po(lines=[make_line(unit_price=100.0, qty=1, gst_rate=18)])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(unit_price=100.0 + impact, qty=1, gst_rate=18)])
    return match_invoice(invoice, po, grn)


def test_price_floor_exactly_500_clears():
    result = _price_floor_case(DEFAULT_TOLERANCES["price_abs"])  # exactly Rs 500 impact
    v = _variance(result, "price_variance")
    assert v["delta_pct"] > 2.0  # confirms this is testing the floor, not the percentage clause
    assert v["within_tolerance"] is True


def test_price_floor_just_over_500_escalates():
    result = _price_floor_case(DEFAULT_TOLERANCES["price_abs"] + 1)  # Rs 501
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is False


def test_price_floor_just_under_500_clears():
    result = _price_floor_case(DEFAULT_TOLERANCES["price_abs"] - 1)  # Rs 499
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is True


def test_price_tolerance_is_evaluated_on_line_impact_not_per_unit_delta():
    """The regression the README calls out by name: 'an earlier version checked
    the wrong number and let an 18.8%-overpriced invoice slip through as in
    tolerance' -- by comparing the raw per-unit delta against the Rs 500
    floor instead of qty x delta.

    Rs 1.00/unit delta at qty=1000 is Rs 1000 of real exposure. A per-unit-only
    floor check would compare Rs 1.00 to Rs 500 and wrongly clear it; the
    percentage clause is also unambiguously over 2% here (20%), so there is no
    route to "within tolerance" except the bug this test guards against."""
    po = make_po(lines=[make_line(unit_price=5.0, qty=1000, gst_rate=18)])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(unit_price=6.0, qty=1000, gst_rate=18)])  # +Rs1/unit, 20%
    result = match_invoice(invoice, po, grn)
    v = _variance(result, "price_variance")
    assert v["delta"] == pytest.approx(1.0, abs=0.01)  # small in isolation...
    assert v["delta_pct"] == pytest.approx(20.0, abs=0.1)  # ...and unambiguously over the 2% clause
    assert v["within_tolerance"] is False  # ...but qty x delta = Rs 1000, over the Rs 500 floor


# --- quantity: 1-unit tolerance ----------------------------------------------

def _qty_case(delta):
    po = make_po(lines=[make_line(qty=50)])
    grn = make_grn(po, qty_overrides={"SKU-1": 50 - delta})
    invoice = make_invoice(po, lines=[make_line(qty=50)])
    return match_invoice(invoice, po, grn)


@pytest.mark.parametrize("delta", [-1, 1])
def test_quantity_delta_within_one_unit_clears(delta):
    result = _qty_case(delta)
    v = _variance(result, "quantity_variance")
    assert v["within_tolerance"] is True


@pytest.mark.parametrize("delta", [-2, 2])
def test_quantity_delta_beyond_one_unit_escalates(delta):
    result = _qty_case(delta)
    v = _variance(result, "quantity_variance")
    assert v["within_tolerance"] is False


def test_quantity_exact_match_raises_no_variance():
    result = _qty_case(0)
    assert not any(v["type"] == "quantity_variance" for v in result["variances"])


# --- tax: no tolerance at all -------------------------------------------------

@pytest.mark.parametrize("po_rate,invoice_rate", [(18, 17), (12, 18), (18, 12)])
def test_any_gst_rate_mismatch_escalates_with_no_tolerance(po_rate, invoice_rate):
    """Unlike price and quantity, the matcher has no tolerance floor for tax at
    all -- a 1-point rate difference must flag exactly like a 6-point one."""
    po = make_po(lines=[make_line(gst_rate=po_rate)])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(gst_rate=invoice_rate)])
    result = match_invoice(invoice, po, grn)
    v = _variance(result, "tax_mismatch")
    assert v["within_tolerance"] is False
    assert v["po_value"] == po_rate
    assert v["invoice_value"] == invoice_rate


def test_matching_gst_rate_raises_no_variance():
    po = make_po(lines=[make_line(gst_rate=18)])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(gst_rate=18)])
    result = match_invoice(invoice, po, grn)
    assert not any(v["type"] == "tax_mismatch" for v in result["variances"])


# --- structural cases ---------------------------------------------------------

def test_missing_po_short_circuits_before_line_comparison():
    invoice = make_invoice(make_po(), inv_id="INV-orphan")
    result = match_invoice(invoice, None, None)
    assert result["status"] == "missing_po"
    assert result["variances"] == []
    assert result["clean"] is False


def test_po_with_no_grn_never_reports_a_three_way_quantity_variance():
    """No GRN means there's no receiving evidence, so quantity can never be
    reported as the three-way `quantity_variance` type -- that type claims a
    goods receipt was checked, and none exists. This holds whether or not the
    invoice and PO quantities actually agree; see the two tests below for
    what happens on the two-way fallback in each case."""
    po = make_po(lines=[make_line(unit_price=100.0, qty=10, gst_rate=18)])
    invoice = make_invoice(po, lines=[make_line(unit_price=200.0, qty=10, gst_rate=18)])  # 100% price bump
    result = match_invoice(invoice, po, None)
    assert result["status"] == "matched"
    assert not any(v["type"] == "quantity_variance" for v in result["variances"])
    v = _variance(result, "price_variance")
    assert v["within_tolerance"] is False


# --- quantity, no GRN: two-way fallback against the PO's own ordered qty ----
# This is the two-way-match gap the README used to call out by name: "PO + no
# GRN auto-clears if price/tax match, without quantity verification." An
# invoice with no goods receipt on file could bill any quantity it liked and
# still clear, as long as price and tax matched. core/matcher.py now falls
# back to comparing the invoice's billed qty against what the PO itself
# ordered whenever there's no GRN line to check against -- weaker evidence
# than a real goods receipt, but not nothing.

def test_po_with_no_grn_and_matching_quantity_raises_no_variance():
    po = make_po(lines=[make_line(unit_price=100.0, qty=10, gst_rate=18)])
    invoice = make_invoice(po, lines=[make_line(unit_price=100.0, qty=10, gst_rate=18)])
    result = match_invoice(invoice, po, None)
    assert result["variances"] == []
    assert result["clean"] is True


def test_po_with_no_grn_and_mismatched_quantity_is_flagged_and_not_clean():
    po = make_po(lines=[make_line(unit_price=100.0, qty=10, gst_rate=18)])
    invoice = make_invoice(po, lines=[make_line(unit_price=100.0, qty=15, gst_rate=18)])  # +5 vs PO
    result = match_invoice(invoice, po, None)
    v = _variance(result, "quantity_variance_no_grn")
    assert v["po_value"] == 10
    assert v["grn_value"] is None  # no receiving evidence -- explicitly not claimed
    assert v["invoice_value"] == 15
    assert v["delta"] == 5
    assert v["within_tolerance"] is False
    assert result["clean"] is False


@pytest.mark.parametrize("delta", [-1, 1])
def test_po_with_no_grn_quantity_within_one_unit_clears(delta):
    po = make_po(lines=[make_line(qty=50)])
    invoice = make_invoice(po, lines=[make_line(qty=50 + delta)])
    result = match_invoice(invoice, po, None)
    v = _variance(result, "quantity_variance_no_grn")
    assert v["within_tolerance"] is True
    assert result["clean"] is True


@pytest.mark.parametrize("delta", [-2, 2])
def test_po_with_no_grn_quantity_beyond_one_unit_escalates(delta):
    po = make_po(lines=[make_line(qty=50)])
    invoice = make_invoice(po, lines=[make_line(qty=50 + delta)])
    result = match_invoice(invoice, po, None)
    v = _variance(result, "quantity_variance_no_grn")
    assert v["within_tolerance"] is False
    assert result["clean"] is False


def test_line_on_invoice_not_on_po_is_line_item_mismatch():
    po = make_po(lines=[make_line(sku="SKU-1")])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(sku="SKU-1"), make_line(sku="SKU-EXTRA")])
    result = match_invoice(invoice, po, grn)
    v = _variance(result, "line_item_mismatch")
    assert v["sku"] == "SKU-EXTRA"
    assert v["within_tolerance"] is False


def test_extra_po_line_not_billed_is_not_an_exception():
    """Partial billing is normal: a PO line the invoice doesn't bill for yet is
    not a finding, only the reverse (invoice lines absent from the PO) is."""
    po = make_po(lines=[make_line(sku="SKU-1"), make_line(sku="SKU-2")])
    grn = make_grn(po)
    invoice = make_invoice(po, lines=[make_line(sku="SKU-1")])  # SKU-2 not billed
    result = match_invoice(invoice, po, grn)
    assert result["variances"] == []
    assert result["clean"] is True


def test_vendor_mismatch_marks_not_exact_but_does_not_itself_add_a_variance():
    po = make_po(vendor="Acme Corp")
    grn = make_grn(po)
    invoice = make_invoice(po, vendor="Acme Corporation Pvt Ltd")
    result = match_invoice(invoice, po, grn)
    assert result["vendor_exact_match"] is False
    assert result["clean"] is False
    assert result["variances"] == []  # vendor drift is not itself a "variance" entry
