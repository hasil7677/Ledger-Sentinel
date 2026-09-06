"""Build the escalation packet a clerk reads in the review UI.

Schema matches STARTER.md section 7.
"""

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}


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
        },
        "findings_detail": findings,
        "proposed_resolution": summary_result["proposed_resolution"],
        "requires": "ap_clerk_approval",
    }
