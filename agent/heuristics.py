"""Deterministic stand-ins for the three judgement calls.

These are string-similarity rules, not reasoning. They exist for two reasons
and both of them matter:

1. They are what the stub model returns (`agent/llm.py`), so the whole
   pipeline -- orchestrator, eval, tests, UI -- runs to completion with no
   API key and no network.
2. They are the fallback when a real provider is configured but the call
   fails or comes back unparseable. A judgement call that errors must not
   take the invoice down with it; it falls back here and the trace records
   that it did.

Nothing in this module talks to a network. Each function returns the same
shape the corresponding provider task is contracted to return, so callers
cannot tell structurally which path produced an answer -- only the `source`
tag on the trace step can.
"""

import difflib

VENDOR_SIMILARITY_THRESHOLD = 0.6
DUPLICATE_SIMILARITY_THRESHOLD = 0.7


def normalize_vendor(invoice_vendor, po_vendor):
    ratio = difflib.SequenceMatcher(None, invoice_vendor.lower(), po_vendor.lower()).ratio()
    return {
        "same_entity": ratio >= VENDOR_SIMILARITY_THRESHOLD,
        "confidence": round(ratio, 2),
        "reasoning": (
            f"String similarity {ratio:.2f} between '{invoice_vendor}' and '{po_vendor}' "
            f"(threshold {VENDOR_SIMILARITY_THRESHOLD:.2f})."
        ),
    }


def reason_duplicate(invoice, candidates):
    if not candidates:
        return {
            "is_duplicate": False,
            "related_invoice_id": None,
            "confidence": 0.0,
            "reasoning": "No posted ledger entries were supplied to compare against.",
        }
    best = candidates[0]
    similarity = difflib.SequenceMatcher(None, invoice["id"], best["invoice_id"]).ratio()
    is_duplicate = similarity >= DUPLICATE_SIMILARITY_THRESHOLD
    return {
        "is_duplicate": is_duplicate,
        "related_invoice_id": best["invoice_id"] if is_duplicate else None,
        "confidence": round(similarity, 2),
        "reasoning": (
            f"Vendor and amount match posted invoice {best['invoice_id']}; "
            f"invoice-number similarity {similarity:.2f} "
            f"(threshold {DUPLICATE_SIMILARITY_THRESHOLD:.2f})."
        ),
    }


# Proposed resolutions are keyed by finding type and ordered by how much the
# finding constrains the human's next action. A duplicate has exactly one
# sensible resolution; a price variance needs a conversation with the vendor.
_RESOLUTION_BY_TYPE = {
    "vendor_bank_detail_change": (
        "Do not pay. Call the vendor on the number already on file -- not one from "
        "the invoice or the covering email -- and confirm the new bank account before "
        "releasing the hold."
    ),
    "vendor_employee_collision": (
        "Do not pay. Route to internal audit: the vendor's payment or contact details "
        "match an employee record."
    ),
    "suspected_duplicate": "Reject as duplicate. Related invoice already posted: {related}.",
    "threshold_splitting": (
        "Hold all invoices in the group and route to the approver the combined amount "
        "actually requires."
    ),
    "sod_violation": "Route to a second approver; the requester cannot approve their own purchase.",
    "new_vendor_first_invoice": (
        "Verify the vendor through onboarding before the first payment is released."
    ),
    "missing_po": "Route to approver for non-PO invoice sign-off.",
}

_SEVERITY_ORDER = ["high", "medium", "low"]


def _pick_primary(findings):
    by_severity = {s: [f for f in findings if f.get("severity") == s] for s in _SEVERITY_ORDER}
    for s in _SEVERITY_ORDER:
        if by_severity[s]:
            return by_severity[s][0]
    return findings[0]


def summarize_exception(invoice, po, grn, findings):
    """A flat, factual summary. Deliberately not written to sound like prose --
    if the real model is unavailable, the clerk should get the facts rather than
    an imitation of a model's voice."""
    if not findings:
        return {"summary": f"Invoice {invoice['id']} raised no findings.",
                "proposed_resolution": "No action required.", "confidence": 0.5}

    types = [f["type"] for f in findings]
    primary = _pick_primary(findings)
    summary = (
        f"Invoice {invoice['id']} from {invoice['vendor']} for Rs {invoice['total_amount']:,.2f} "
        f"flagged for: {', '.join(t.replace('_', ' ') for t in types)}."
    )

    resolution = _RESOLUTION_BY_TYPE.get(primary["type"])
    if resolution is None:
        resolution = (
            f"Confirm {primary['type'].replace('_', ' ')} with the vendor or receiving "
            f"before posting."
        )
    elif "{related}" in resolution:
        candidates = primary.get("detail", {}).get("candidates") or [{}]
        resolution = resolution.format(related=candidates[0].get("invoice_id", "unknown"))

    return {"summary": summary, "proposed_resolution": resolution, "confidence": 0.5}
