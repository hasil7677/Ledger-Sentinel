"""Anomaly taxonomy: turns raw matcher output + ledger signals into a
classified list of findings.

Still deterministic. Findings marked needs_judgement=True are the ones
where an LLM adds value (is this vendor string the same entity, is this
really a duplicate); everything else is an unambiguous rule outcome.
"""

SEVERITY_BY_TYPE = {
    "missing_po": "high",
    "suspected_duplicate": "high",
    "price_variance_out_tolerance": "medium",
    "tax_mismatch": "medium",
    "quantity_variance": "medium",
    "line_item_mismatch": "medium",
    "vendor_name_drift": "low",
}


def classify(invoice, match_result, ledger, trace=None):
    """`trace`, when supplied, is any object with .add(action, detail); it records
    the ledger duplicate search as it happens, including a negative result."""
    """Return a list of findings: [{type, severity, needs_judgement, detail}]."""
    findings = []

    if match_result["status"] == "missing_po":
        findings.append({
            "type": "missing_po",
            "severity": SEVERITY_BY_TYPE["missing_po"],
            "needs_judgement": False,
            "detail": {"invoice_id": invoice["id"]},
        })
        return findings  # no PO means no further line-level comparison is possible

    if not match_result["vendor_exact_match"]:
        findings.append({
            "type": "vendor_name_drift",
            "severity": SEVERITY_BY_TYPE["vendor_name_drift"],
            "needs_judgement": True,
            "detail": {"invoice_vendor": invoice["vendor"]},
        })

    for v in match_result["variances"]:
        if v["within_tolerance"]:
            continue
        findings.append({
            "type": v["type"] if v["type"] != "price_variance" else "price_variance_out_tolerance",
            "severity": SEVERITY_BY_TYPE.get(
                v["type"] if v["type"] != "price_variance" else "price_variance_out_tolerance", "medium"
            ),
            "needs_judgement": False,
            "detail": v,
        })

    near_matches = ledger.find_near_matches(
        vendor=invoice["vendor"],
        total_amount=invoice["total_amount"],
        invoice_date=invoice["invoice_date"],
        exclude_invoice_id=invoice["id"],
    )
    if trace is not None:
        scope = (f"Searched posted ledger for {invoice['vendor']} invoices near "
                 f"Rs {invoice['total_amount']:,.2f} within 30 days of {invoice['invoice_date']}")
        if near_matches:
            listed = ", ".join(f"{m['invoice_id']} (posted {m['invoice_date']})" for m in near_matches)
            trace.add("ledger_duplicate_search",
                      f"{scope}: found {len(near_matches)} candidate(s) -- {listed}.")
        else:
            trace.add("ledger_duplicate_search", f"{scope}: no near-matches found.")

    if near_matches:
        findings.append({
            "type": "suspected_duplicate",
            "severity": SEVERITY_BY_TYPE["suspected_duplicate"],
            "needs_judgement": True,
            "detail": {"candidates": near_matches},
        })

    return findings
