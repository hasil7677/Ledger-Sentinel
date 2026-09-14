"""Main loop: ingest -> match -> resolve -> post or escalate.

Deterministic matching/classification decides *what's wrong*. The agent
tools decide judgement calls (same vendor? really a duplicate? how to
phrase it). Conservative by design: any deterministic finding that isn't
explicitly resolved by a judgement call falls through to escalation.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matcher import match_invoice
from core.anomalies import classify
from core.ledger import Ledger
from core.vendor_master import VendorMaster
from agent import tools
from agent.escalation import build_packet
from agent.trace import Trace, describe_variance, llm_source

VENDOR_NORMALIZE_CONFIDENCE = 0.6
DUPLICATE_CONFIDENCE = 0.5


def load_data(data_dir):
    with open(os.path.join(data_dir, "pos.json")) as f:
        pos = {p["id"]: p for p in json.load(f)}
    with open(os.path.join(data_dir, "grns.json")) as f:
        grns_list = json.load(f)
    grn_by_po = {g["po_id"]: g for g in grns_list}
    with open(os.path.join(data_dir, "invoices.json")) as f:
        invoices = json.load(f)
    return pos, grn_by_po, invoices


def load_vendor_master(data_dir):
    """Optional. A dataset generated before core/vendor_master.py existed has
    no vendors.json; every vendor-master check then simply finds no record and
    is skipped (see core/anomalies.py::_vendor_master_findings), so older
    datasets keep running rather than erroring out."""
    path = os.path.join(data_dir, "vendors.json")
    if not os.path.exists(path):
        return VendorMaster.empty()
    return VendorMaster.load(path)


def load_employees(data_dir):
    """Optional, same reasoning as load_vendor_master. Returns {employee_id: record}."""
    path = os.path.join(data_dir, "employees.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {e["employee_id"]: e for e in json.load(f)}


def process_invoice(invoice, pos, grn_by_po, ledger, vendor_master=None, employees_by_id=None):
    po = pos.get(invoice.get("po_id"))
    grn = grn_by_po.get(invoice.get("po_id"))

    # The trace is written as each check runs, from that check's own output, so
    # the packet's step list is a record of execution rather than a summary
    # composed once the answer is already known.
    trace = Trace()
    if po is None:
        trace.add("retrieve_documents",
                  f"Invoice {invoice['id']} cites no purchase order; no PO/GRN on file to match against.")
    elif grn is None:
        # PO on file but nothing received yet -- matcher treats received qty as
        # unknown rather than zero, so say so instead of implying a GRN exists.
        trace.add("retrieve_documents",
                  f"Retrieved PO {po['id']} for invoice {invoice['id']}; no goods receipt on file, "
                  f"so quantities cannot be confirmed against receiving. "
                  f"{len(invoice['lines'])} invoice line(s) to compare on price and tax only.")
    else:
        trace.add("retrieve_documents",
                  f"Retrieved PO {po['id']} and GRN {grn['id']} for invoice {invoice['id']}; "
                  f"{len(invoice['lines'])} invoice line(s) to compare.")

    match_result = match_invoice(invoice, po, grn)

    if match_result["status"] == "matched":
        variances = match_result["variances"]
        outside = [v for v in variances if not v["within_tolerance"]]
        within = [v for v in variances if v["within_tolerance"]]
        if not variances:
            trace.add("three_way_match",
                      "PO, GRN and invoice agree on every line: quantities, unit prices and GST rates all match."
                      if grn is not None else
                      "Invoice agrees with the PO on unit prices and GST rates. Quantities were not verified "
                      "against receiving because no goods receipt exists.")
        else:
            trace.add("three_way_match",
                      f"Compared {len(invoice['lines'])} line(s) across PO/GRN/invoice: "
                      f"{len(outside)} variance(s) outside tolerance, {len(within)} within tolerance.")
        for v in within:
            trace.add("tolerance_check", f"{describe_variance(v)} -- within tolerance, not escalated.")
        for v in outside:
            trace.add("tolerance_check", f"{describe_variance(v)} -- outside tolerance.")

        if match_result["vendor_exact_match"]:
            trace.add("vendor_check", f"Invoice vendor '{invoice['vendor']}' matches the PO vendor exactly.")
        else:
            trace.add("vendor_check",
                      f"Invoice vendor '{invoice['vendor']}' differs from PO vendor '{po['vendor']}'; "
                      f"referred for entity resolution.")

    findings = classify(invoice, match_result, ledger,
                        vendor_master=vendor_master, employees_by_id=employees_by_id, trace=trace)

    if not findings:
        trace.add("conclusion", "No exceptions raised. Auto-cleared and posted to the ledger.")
        ledger.post(invoice, resolution="auto_clear")
        return {"invoice_id": invoice["id"], "decision": "auto_clear", "findings": [],
                "packet": None, "steps": trace.steps()}

    judgements = {}
    escalate = False

    for f in findings:
        if f["type"] == "vendor_name_drift":
            result = tools.normalize_vendor(invoice["vendor"], po["vendor"])
            judgements["vendor_name_drift"] = result
            resolved = result.get("same_entity") and result.get("confidence", 0) >= VENDOR_NORMALIZE_CONFIDENCE
            trace.add(
                "judgement_vendor",
                f"Vendor resolution: '{invoice['vendor']}' vs '{po['vendor']}' -- "
                f"{'same entity' if result.get('same_entity') else 'not the same entity'} "
                f"at confidence {result.get('confidence', 0):.2f} "
                f"(threshold {VENDOR_NORMALIZE_CONFIDENCE:.2f}). "
                f"{result.get('reasoning', '')}".strip(),
                source=llm_source(result),
            )
            if not resolved:
                escalate = True
        elif f["type"] == "suspected_duplicate":
            # A ledger near-match on vendor + amount + date is sufficient on its own to
            # put this in front of a human. The model may sharpen severity, confidence
            # and wording, but it cannot suppress the escalation: one wrong "not a
            # duplicate" means paying the invoice twice, the costliest failure in AP.
            result = tools.reason_duplicate(invoice, f["detail"]["candidates"])
            judgements["suspected_duplicate"] = result
            model_agrees = bool(result.get("is_duplicate")) and result.get("confidence", 0) >= DUPLICATE_CONFIDENCE
            trace.add(
                "judgement_duplicate",
                f"Duplicate assessment against {len(f['detail']['candidates'])} ledger candidate(s): "
                f"model says {'duplicate' if result.get('is_duplicate') else 'not a duplicate'} "
                f"at confidence {result.get('confidence', 0):.2f} "
                f"(threshold {DUPLICATE_CONFIDENCE:.2f}), "
                f"related invoice {result.get('related_invoice_id') or 'none'}. "
                f"{result.get('reasoning', '')}".strip(),
                source=llm_source(result),
            )
            if not model_agrees:
                trace.add(
                    "rule_escalation",
                    "Ledger near-match on vendor, amount and date stands on its own; escalating "
                    "regardless of the model's assessment. Model judgement may adjust severity and "
                    "confidence on a duplicate but can never suppress the escalation.",
                )
            escalate = True
        else:
            # missing_po, price/qty/tax/line-item variance, and every vendor-master
            # and approval-policy finding: no judgement call, always escalate. None
            # of these ask "is this really a problem" -- they are already a fact
            # (a bank account changed, a requester's invoices sum past their limit)
            # and the only open question is what a human does about it.
            trace.add("rule_escalation",
                      f"{f['type'].replace('_', ' ').capitalize()} is a hard exception under the matching "
                      f"rules ({f['severity']} severity); routed to a human without model judgement.")
            escalate = True

    if not escalate:
        trace.add("conclusion",
                  "All findings resolved automatically; no human review required. Posted to the ledger.")
        ledger.post(invoice, resolution="auto_clear")
        return {"invoice_id": invoice["id"], "decision": "auto_clear", "findings": [f["type"] for f in findings],
                "packet": None, "steps": trace.steps()}

    summary_result = tools.summarize_exception(invoice, po, grn, findings)
    packet = build_packet(invoice, po, grn, findings, judgements, summary_result)
    trace.add(
        "conclusion",
        f"Escalated as {packet['exception_type'].replace('_', ' ')} "
        f"({packet['severity']} severity, confidence {packet['confidence']:.2f}). "
        f"Proposed: {packet['proposed_resolution']}",
        source=llm_source(summary_result),
    )
    packet["steps"] = trace.steps()
    return {"invoice_id": invoice["id"], "decision": "escalate", "findings": [f["type"] for f in findings],
            "packet": packet, "steps": trace.steps()}


def run(data_dir, out_dir, fresh_ledger=False):
    """`fresh_ledger=True` truncates runtime/ledger.json before the run.

    The default (False) matches the pre-existing CLI behaviour: `python -m
    agent.orchestrator` is meant to be runnable repeatedly against a live
    `runtime/` directory, appending newly-arrived invoices onto whatever is
    already posted -- that is what makes the duplicate detector meaningful in
    a demo. `eval/run_eval.py` passes True, because a batch score has to be
    reproducible from the dataset alone, not from whatever state a previous
    eval run left behind.
    """
    os.makedirs(out_dir, exist_ok=True)
    pos, grn_by_po, invoices = load_data(data_dir)
    vendor_master = load_vendor_master(data_dir)
    employees_by_id = load_employees(data_dir)
    ledger = Ledger(os.path.join(out_dir, "ledger.json"), fresh=fresh_ledger)

    tools.reset_usage()
    results = []
    exceptions = []
    for invoice in invoices:
        result = process_invoice(invoice, pos, grn_by_po, ledger,
                                 vendor_master=vendor_master, employees_by_id=employees_by_id)
        results.append(result)
        if result["packet"] is not None:
            exceptions.append(result["packet"])

    with open(os.path.join(out_dir, "exceptions.json"), "w") as f:
        json.dump(exceptions, f, indent=2)

    return results, tools.get_usage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="runtime")
    parser.add_argument("--fresh-ledger", action="store_true",
                        help="Truncate the ledger before this run instead of appending to it.")
    args = parser.parse_args()

    results, usage = run(args.data_dir, args.out_dir, fresh_ledger=args.fresh_ledger)

    auto_clear = sum(1 for r in results if r["decision"] == "auto_clear")
    escalate = sum(1 for r in results if r["decision"] == "escalate")
    print(f"Processed {len(results)} invoices -> {auto_clear} auto-cleared, {escalate} escalated")
    print(f"Ledger + exception queue written to {args.out_dir}/")
    print(f"LLM usage: {usage}")


if __name__ == "__main__":
    main()
