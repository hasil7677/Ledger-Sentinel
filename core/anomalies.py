"""Anomaly taxonomy: turns matcher output plus ledger, vendor-master and
approval-policy signals into a classified list of findings.

Still deterministic. Findings marked needs_judgement=True are the ones where
an LLM adds value (is this vendor string the same entity, is this really a
duplicate); everything else is an unambiguous rule outcome. The vendor-master
and approval findings added below are all needs_judgement=False -- whether a
bank account changed, whether a vendor was onboarded 40 days ago, and whether
a requester's invoices to one vendor sum past their limit are facts, not
judgement calls, and the model is not consulted on any of them.
"""

from core import approval

SEVERITY_BY_TYPE = {
    "missing_po": "high",
    "suspected_duplicate": "high",
    "price_variance_out_tolerance": "medium",
    "tax_mismatch": "medium",
    "quantity_variance": "medium",
    "line_item_mismatch": "medium",
    "vendor_name_drift": "low",
    # Vendor-master fraud controls. A bank-detail change or an employee/vendor
    # collision is the failure mode that costs the most and looks the most
    # like nothing is wrong -- every line on the three-way match agrees, the
    # only thing wrong is who gets paid. Both rank alongside suspected_duplicate.
    "vendor_bank_detail_change": "high",
    "vendor_employee_collision": "high",
    "new_vendor_first_invoice": "medium",
    # Approval / segregation of duties.
    "approval_threshold_exceeded": "medium",
    "threshold_splitting": "high",
}


def _vendor_master_findings(invoice, vendor_master, employees, trace=None):
    """Bank-detail-change hold, first-invoice-from-new-vendor, and
    vendor/employee collision. Returns a list of findings (0 to 2 -- a vendor
    can be both new *and* mid-collision, in principle, though the generator
    never seeds both on the same vendor)."""
    findings = []
    if vendor_master is None:
        return findings

    record = vendor_master.by_name(invoice["vendor"])
    if record is None:
        if trace is not None:
            trace.add("vendor_master_lookup",
                      f"'{invoice['vendor']}' has no vendor master record on file; "
                      f"bank-detail and new-vendor checks skipped.")
        return findings

    vendor_id = record["vendor_id"]

    pending = vendor_master.pending_payment_changes(vendor_id, as_of=invoice["invoice_date"])
    is_new = vendor_master.is_new_vendor(vendor_id, as_of=invoice["invoice_date"])
    collision = vendor_master.employee_collision(vendor_id, employees or [])

    if trace is not None:
        bits = []
        if pending:
            fields = ", ".join(sorted({p["field"] for p in pending}))
            bits.append(f"{len(pending)} unverified payment-detail change(s) on file ({fields})")
        if is_new:
            bits.append(f"vendor onboarded {record.get('onboarded_date')}, within the new-vendor window")
        if collision:
            bits.append(f"{collision['match_field']} matches employee {collision['employee_name']}")
        detail = f"Vendor master check for '{record['canonical_name']}': " + (
            "; ".join(bits) if bits else "no holds, not a new vendor, no employee collision."
        )
        trace.add("vendor_master_check", detail)

    if pending:
        findings.append({
            "type": "vendor_bank_detail_change",
            "severity": SEVERITY_BY_TYPE["vendor_bank_detail_change"],
            "needs_judgement": False,
            "detail": {"vendor_id": vendor_id, "changes": pending},
        })

    if collision:
        findings.append({
            "type": "vendor_employee_collision",
            "severity": SEVERITY_BY_TYPE["vendor_employee_collision"],
            "needs_judgement": False,
            "detail": collision,
        })

    # A collision or a bank-detail hold already routes this to a human; do not
    # also raise "first invoice from a new vendor" on top of a vendor that's
    # already being held for a stronger reason -- it would just be noise ahead
    # of the actual problem in the packet.
    if is_new and not pending and not collision:
        findings.append({
            "type": "new_vendor_first_invoice",
            "severity": SEVERITY_BY_TYPE["new_vendor_first_invoice"],
            "needs_judgement": False,
            "detail": {"vendor_id": vendor_id, "onboarded_date": record.get("onboarded_date")},
        })

    return findings


def _approval_findings(invoice, ledger, employees_by_id, requester_id, trace=None):
    """Approval-threshold and split-invoice findings. Needs the ledger to look
    up the requester's other recent invoices to the same vendor."""
    findings = []
    requester = (employees_by_id or {}).get(requester_id)
    if requester is None and requester_id is None:
        # No requester recorded at all is a data-completeness question, not an
        # approval-policy one; nothing to check.
        return findings

    prior_entries = ledger.entries_for_vendor(
        invoice["vendor"], before_date=invoice["invoice_date"], day_window=approval.SPLIT_WINDOW_DAYS
    )
    raw = approval.check_invoice(invoice, requester, prior_entries)

    if trace is not None:
        if requester is None:
            trace.add("approval_check",
                      f"Invoice cites requester_id={requester_id!r}, which has no employee "
                      f"master record; treated as no accountable requester.")
        elif raw:
            f = raw[0]
            if f["type"] == "approval_threshold_exceeded":
                trace.add("approval_check",
                          f"Requester {requester['name']}'s approval limit is "
                          f"Rs {requester['approval_limit']:,.2f}; this invoice alone is "
                          f"Rs {invoice['total_amount']:,.2f}, over by Rs {f['detail']['over_by']:,.2f}.")
            else:
                members = ", ".join(m["invoice_id"] for m in f["detail"]["members"])
                trace.add("approval_check",
                          f"Requester {requester['name']}'s approval limit is "
                          f"Rs {requester['approval_limit']:,.2f}; invoices to {invoice['vendor']} within "
                          f"{approval.SPLIT_WINDOW_DAYS} days are each under that limit but sum to "
                          f"Rs {f['detail']['group_total']:,.2f} ({members}) -- Rs {f['detail']['over_by']:,.2f} over.")
        else:
            trace.add("approval_check",
                      f"Requester {requester['name']}'s approval limit is "
                      f"Rs {requester['approval_limit']:,.2f}; this invoice and its recent history to "
                      f"{invoice['vendor']} are within it.")

    for f in raw:
        findings.append({
            "type": f["type"],
            "severity": SEVERITY_BY_TYPE[f["type"]],
            "needs_judgement": False,
            "detail": f["detail"],
        })
    return findings


def classify(invoice, match_result, ledger, vendor_master=None, employees_by_id=None, trace=None):
    """Return a list of findings: [{type, severity, needs_judgement, detail}].

    `trace`, when supplied, is any object with .add(action, detail); findings
    that depend on a search (ledger duplicate search, vendor-master lookup,
    approval history) record what they found, including a negative result, as
    each search runs.
    """
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

    employee_list = list(employees_by_id.values()) if employees_by_id else []
    findings.extend(_vendor_master_findings(invoice, vendor_master, employee_list, trace))

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

    # Approval-policy checks are opt-in on the employee master being supplied,
    # the same way vendor-master checks are opt-in on vendor_master -- a
    # caller that never loaded an employee master (an older dataset, most
    # unit tests) gets no approval findings rather than every invoice being
    # treated as "no accountable requester on record."
    if ledger is not None and employees_by_id is not None:
        findings.extend(_approval_findings(
            invoice, ledger, employees_by_id, invoice.get("requester_id"), trace
        ))

    return findings
