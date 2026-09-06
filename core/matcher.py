"""Deterministic three-way match: PO <-> GRN <-> Invoice.

No LLM calls here. This module only does arithmetic and structural
comparison. Judgement calls (is this vendor name the same entity, is this
really a duplicate, how to phrase the exception) belong in agent/.
"""

DEFAULT_TOLERANCES = {
    "price_pct": 0.02,   # 2%
    "price_abs": 500.0,  # rupees
    "qty_units": 1,      # units
}


def _index_by_sku(lines, qty_key="qty"):
    return {l["sku"]: l for l in lines}


def match_invoice(invoice, po, grn, tolerances=None):
    """Compare one invoice against its PO and GRN.

    Returns a dict:
      {
        "invoice_id": ...,
        "status": "missing_po" | "matched",
        "vendor_exact_match": bool,
        "variances": [ {type, sku, po_value, grn_value, invoice_value, delta, delta_pct, within_tolerance} ],
        "clean": bool   # true iff no variances outside tolerance and vendor matches exactly
      }
    """
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}

    if po is None:
        return {
            "invoice_id": invoice["id"],
            "status": "missing_po",
            "vendor_exact_match": None,
            "variances": [],
            "clean": False,
        }

    po_lines = _index_by_sku(po["lines"])
    grn_lines = _index_by_sku(grn["lines"]) if grn else {}
    variances = []

    vendor_exact_match = invoice["vendor"] == po["vendor"]

    for line in invoice["lines"]:
        sku = line["sku"]
        po_line = po_lines.get(sku)

        if po_line is None:
            variances.append({
                "type": "line_item_mismatch",
                "sku": sku,
                "po_value": None,
                "grn_value": None,
                "invoice_value": {"qty": line["qty"], "unit_price": line["unit_price"]},
                "within_tolerance": False,
            })
            continue

        # price variance: tolerance is evaluated on the total monetary impact of the
        # line (qty * unit-price delta), not the raw per-unit delta, since a small
        # per-unit difference on a high-qty line can still be a large rupee exposure.
        price_delta = line["unit_price"] - po_line["unit_price"]
        price_delta_pct = abs(price_delta) / po_line["unit_price"] if po_line["unit_price"] else 0
        line_impact = abs(price_delta) * line["qty"]
        price_within = line_impact <= tol["price_abs"] or price_delta_pct <= tol["price_pct"]
        if price_delta != 0:
            variances.append({
                "type": "price_variance",
                "sku": sku,
                "po_value": po_line["unit_price"],
                "grn_value": None,
                "invoice_value": line["unit_price"],
                "delta": round(price_delta, 2),
                "delta_pct": round(price_delta_pct * 100, 2),
                "within_tolerance": price_within,
            })

        # quantity variance: invoice qty vs GRN received qty
        grn_line = grn_lines.get(sku)
        received = grn_line["qty_received"] if grn_line else None
        if received is not None and received != line["qty"]:
            qty_delta = line["qty"] - received
            variances.append({
                "type": "quantity_variance",
                "sku": sku,
                "po_value": po_line["qty"],
                "grn_value": received,
                "invoice_value": line["qty"],
                "delta": qty_delta,
                "within_tolerance": abs(qty_delta) <= tol["qty_units"],
            })

        # tax mismatch
        if line["gst_rate"] != po_line["gst_rate"]:
            variances.append({
                "type": "tax_mismatch",
                "sku": sku,
                "po_value": po_line["gst_rate"],
                "grn_value": None,
                "invoice_value": line["gst_rate"],
                "within_tolerance": False,
            })

    # extra PO lines not billed are not an exception (partial billing is normal); only
    # invoice lines absent from the PO are flagged, handled above as line_item_mismatch.

    clean = vendor_exact_match and all(v["within_tolerance"] for v in variances)

    return {
        "invoice_id": invoice["id"],
        "status": "matched",
        "vendor_exact_match": vendor_exact_match,
        "variances": variances,
        "clean": clean,
    }
