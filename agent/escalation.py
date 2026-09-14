"""Build the escalation packet a clerk reads in the review UI.

Base schema matches STARTER.md section 7. Two evidence blocks were added on
top of it -- `vendor_master` and `approval` -- because a detector whose
output a human never sees is not finished; see core/vendor_master.py and
core/approval.py for what feeds them.
"""

from core.vendor_master import mask_account

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}

_VENDOR_MASTER_TYPES = {"vendor_bank_detail_change", "vendor_employee_collision",
                        "new_vendor_first_invoice"}
_APPROVAL_TYPES = {"approval_threshold_exceeded", "threshold_splitting"}


def _primary_finding(findings):
    return max(findings, key=lambda f: SEVERITY_RANK.get(f["severity"], 0))


def _po_evidence(po):
    if po is None:
        return None
    return {"id": po["id"], "lines": po["lines"]}


def _grn_evidence(grn):
    if grn is None:
        return None
    return {"id": grn["id"], "lines": grn["lines"], "received_date": grn["received_date"]}


def _mask_change(change):
    """A pending payment-detail change, with any account number masked before
    it goes anywhere a clerk's screen or a JSON dump can see it."""
    field = change.get("field")
    old_value = change.get("old_value")
    new_value = change.get("new_value")
    if field == "bank_account":
        old_value = mask_account(old_value) if old_value else old_value
        new_value = mask_account(new_value) if new_value else new_value
    return {
        "seq": change.get("seq"),
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "effective_at": change.get("effective_at"),
        "actor": change.get("actor"),
        "channel": change.get("channel"),
    }


def _vendor_master_evidence(findings):
    """Collapse whichever vendor-master findings fired into one evidence block.
    None of the three types below are mutually exclusive in principle, though
    the generator never seeds more than one on the same vendor at once."""
    relevant = [f for f in findings if f["type"] in _VENDOR_MASTER_TYPES]
    if not relevant:
        return None

    evidence = {}
    for f in relevant:
        if f["type"] == "vendor_bank_detail_change":
            evidence["pending_changes"] = [_mask_change(c) for c in f["detail"]["changes"]]
            evidence["hold"] = "Payment on hold pending out-of-band verification of the change(s) above."
        elif f["type"] == "vendor_employee_collision":
            d = f["detail"]
            evidence["employee_collision"] = {
                "match_field": d["match_field"],
                "employee_id": d["employee_id"],
                "employee_name": d["employee_name"],
            }
        elif f["type"] == "new_vendor_first_invoice":
            evidence["new_vendor"] = {"onboarded_date": f["detail"]["onboarded_date"]}
    return evidence


def _approval_evidence(findings):
    relevant = [f for f in findings if f["type"] in _APPROVAL_TYPES]
    if not relevant:
        return None
    f = relevant[0]
    if f["type"] == "approval_threshold_exceeded":
        return {"kind": "single_invoice_over_limit", **f["detail"]}
    return {"kind": "threshold_splitting", **f["detail"]}


def build_packet(invoice, po, grn, findings, judgements, summary_result):
    """
    findings: list from core.anomalies.classify
    judgements: dict of judgement results keyed by finding type
                (e.g. {"vendor_name_drift": {...normalize_vendor result...},
                       "suspected_duplicate": {...reason_duplicate result...}})
    summary_result: dict from agent.tools.summarize_exception
    """
    primary = _primary_finding(findings)

    vendor_normalised_from = None
    vendor_judgement = judgements.get("vendor_name_drift")
    if vendor_judgement and vendor_judgement.get("same_entity"):
        vendor_normalised_from = invoice["vendor"]

    related = []
    dup_judgement = judgements.get("suspected_duplicate")
    if dup_judgement:
        related_id = dup_judgement.get("related_invoice_id")
        for cand in primary.get("detail", {}).get("candidates", []) if primary["type"] == "suspected_duplicate" else []:
            if related_id is None or cand["invoice_id"] == related_id:
                related.append({"id": cand["invoice_id"], "status": cand["resolution"], "date": cand["invoice_date"]})

    confidence = summary_result.get("confidence", 0.5)
    if dup_judgement and primary["type"] == "suspected_duplicate":
        if dup_judgement.get("is_duplicate"):
            confidence = dup_judgement.get("confidence", confidence)
        else:
            # The ledger near-match escalates regardless of the model's call
            # (see orchestrator.py). dup_judgement["confidence"] here is the
            # model's confidence in "not a duplicate" -- the opposite claim --
            # so it must be inverted rather than reported as-is, or a forced
            # escalation would misleadingly read as high-confidence duplicate.
            confidence = 1 - dup_judgement.get("confidence", 0.5)
    elif vendor_judgement and primary["type"] == "vendor_name_drift":
        confidence = vendor_judgement.get("confidence", confidence)

    return {
        "invoice_id": invoice["id"],
        "po_id": invoice.get("po_id"),
        "vendor": po["vendor"] if (po and vendor_normalised_from) else invoice["vendor"],
        "vendor_normalised_from": vendor_normalised_from,
        "amount": invoice["total_amount"],
        "invoice_date": invoice["invoice_date"],
        "exception_type": primary["type"],
        "severity": primary["severity"],
        "confidence": round(confidence, 2),
        "summary": summary_result["summary"],
        "evidence": {
            "po": _po_evidence(po),
            "grn": _grn_evidence(grn),
            "invoice": {"qty_lines": invoice["lines"], "total_amount": invoice["total_amount"],
                        "invoice_date": invoice["invoice_date"]},
            "related": related,
            "vendor_master": _vendor_master_evidence(findings),
            "approval": _approval_evidence(findings),
        },
        "findings_detail": findings,
        "proposed_resolution": summary_result["proposed_resolution"],
        "requires": "ap_clerk_approval",
    }
