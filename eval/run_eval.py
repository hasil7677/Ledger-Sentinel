"""Batch eval: run the orchestrator over the labelled dataset and report
the four numbers the spec asks for (STARTER.md section 9).

Usage:
    python eval/run_eval.py [--data-dir data] [--out-dir eval/out]
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools
from agent.orchestrator import load_data, process_invoice
from core.ledger import Ledger


def run_eval(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pos, grn_by_po, invoices = load_data(data_dir)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = {l["invoice_id"]: l for l in json.load(f)}

    ledger = Ledger(os.path.join(out_dir, "ledger.json"))
    tools.reset_usage()

    results = []
    latencies = []
    exceptions = []
    for invoice in invoices:
        start = time.perf_counter()
        result = process_invoice(invoice, pos, grn_by_po, ledger)
        latencies.append(time.perf_counter() - start)
        results.append(result)
        if result["packet"] is not None:
            exceptions.append(result["packet"])

    with open(os.path.join(out_dir, "exceptions.json"), "w") as f:
        json.dump(exceptions, f, indent=2)

    usage = tools.get_usage()

    total = len(results)
    auto_clear = [r for r in results if r["decision"] == "auto_clear"]
    escalated = [r for r in results if r["decision"] == "escalate"]

    false_clears = [r for r in auto_clear if labels[r["invoice_id"]]["expected_outcome"] == "escalate"]
    genuine_escalations = [r for r in escalated if labels[r["invoice_id"]]["expected_outcome"] == "escalate"]

    auto_clear_rate = len(auto_clear) / total if total else 0
    false_clear_rate = len(false_clears) / len(auto_clear) if auto_clear else 0
    escalation_precision = len(genuine_escalations) / len(escalated) if escalated else 0

    total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    cost_inr = total_tokens / 1000 * tools.PRICE_PER_1K_TOKENS_INR
    cost_per_invoice = cost_inr / total if total else 0
    p50_latency_ms = statistics.median(latencies) * 1000 if latencies else 0

    metrics = {
        "total_invoices": total,
        "auto_clear_rate": round(auto_clear_rate, 4),
        "false_clear_rate": round(false_clear_rate, 4),
        "false_clear_count": len(false_clears),
        "false_clear_invoice_ids": [r["invoice_id"] for r in false_clears],
        "escalation_precision": round(escalation_precision, 4),
        "escalation_count": len(escalated),
        "llm_calls": usage["calls"],
        "llm_tokens": total_tokens,
        "cost_per_invoice_inr": round(cost_per_invoice, 4),
        "p50_latency_ms": round(p50_latency_ms, 3),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="ledger_sentinel_eval_")
    metrics = run_eval(args.data_dir, out_dir)

    print(f"{'metric':<24}{'value'}")
    for k, v in metrics.items():
        if k == "false_clear_invoice_ids":
            continue
        print(f"{k:<24}{v}")
    if metrics["false_clear_invoice_ids"]:
        print(f"false-clear invoices: {metrics['false_clear_invoice_ids']}")
    print(f"\n(eval artifacts written to {out_dir})")


if __name__ == "__main__":
    main()
