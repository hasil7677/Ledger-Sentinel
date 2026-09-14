"""Batch eval: run the orchestrator over the labelled dataset and quantify the
one thing this project claims to optimise -- the false-clear / false-escalate
asymmetry -- rather than a handful of aggregate rates.

The previous version of this file reported an overall auto-clear rate, an
overall false-clear rate, and escalation precision. Those numbers hide exactly
what a reviewer needs to see: WHICH anomaly types leak past the agent, and
whether the shortfall is the eval script or the detector. A 0% false-clear
rate over an aggregate is compatible with one anomaly type silently missing
100% of the time, as long as the other seven are easy -- averages launder
that. This version reports the anomaly-taxonomy confusion matrix and
per-type recall so a weak detector cannot hide behind a strong one.

It also fixes two correctness bugs in the old eval, both silent:

  1. It built `core.ledger.Ledger(out_dir/ledger.json)` without `fresh=True`,
     so a second run against the same --out-dir inherited the first run's
     postings. Every duplicate-detection metric was then contaminated by
     whatever the previous run happened to leave behind -- the eval was not
     reproducible against its own output directory.
  2. It called `process_invoice` without vendor_master or employees_by_id,
     so none of the vendor-master or approval-policy detectors ever ran
     inside the eval at all -- their entire contribution to auto-clear rate,
     false-clear rate and escalation precision was invisible.

Usage:
    python eval/run_eval.py --data-dir data --out-dir eval_out
    python eval/run_eval.py --regenerate --seed 42 --count 300 --anomaly-rate 0.2 \\
        --data-dir eval_data --out-dir eval_out
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools
from agent.orchestrator import load_data, load_vendor_master, load_employees, process_invoice
from core.ledger import Ledger

# The asymmetry the README's core thesis rests on, made an explicit,
# overridable number instead of an implicit "0% is good, full stop." A false
# clear (auto-paying something that should have gone to a human) is modelled
# as costing DEFAULT_FALSE_CLEAR_COST times as much as a false escalate
# (bothering a clerk with a clean invoice). 50x is a starting assumption, not
# a measured constant -- nothing in this project prices an AP clerk's time or
# a written-off duplicate payment, so treat the absolute score as illustrative
# and the *ranking* it produces across runs/config changes as the useful part.
DEFAULT_FALSE_CLEAR_COST = 50.0
DEFAULT_FALSE_ESCALATE_COST = 1.0


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _confusion_matrix(results, labels):
    """{anomaly_type: {"auto_clear": n, "escalate": n, "total": n,
                        "expected_outcome": "auto_clear"|"escalate"}}"""
    matrix = defaultdict(lambda: {"auto_clear": 0, "escalate": 0})
    expected_by_type = {}
    for r in results:
        label = labels[r["invoice_id"]]
        atype = label["anomaly_type"]
        matrix[atype][r["decision"]] += 1
        expected_by_type[atype] = label["expected_outcome"]

    out = {}
    for atype, counts in sorted(matrix.items()):
        total = counts["auto_clear"] + counts["escalate"]
        expected = expected_by_type[atype]
        correct = counts[expected]
        out[atype] = {
            "expected_outcome": expected,
            "auto_clear": counts["auto_clear"],
            "escalate": counts["escalate"],
            "total": total,
            "correct_rate": round(correct / total, 4) if total else None,
            # Named directly in terms of the asymmetry: a type whose correct
            # outcome is "escalate" but landed on "auto_clear" is a false
            # clear FOR THAT TYPE; the reverse for a type that should clear.
            "false_clear_count": counts["auto_clear"] if expected == "escalate" else 0,
            "false_escalate_count": counts["escalate"] if expected == "auto_clear" else 0,
        }
    return out


def _per_type_recall(confusion):
    """Recall for each type whose correct behaviour is to escalate: of the
    invoices seeded as that type, what fraction actually got escalated. This
    is the number that answers "does the split-invoice detector actually
    work" or "does it only work for duplicates" -- an aggregate can't."""
    return {
        atype: row["correct_rate"]
        for atype, row in confusion.items()
        if row["expected_outcome"] == "escalate"
    }


def run_eval(data_dir, out_dir, fc_cost=DEFAULT_FALSE_CLEAR_COST, fe_cost=DEFAULT_FALSE_ESCALATE_COST,
            regenerate=False, seed=None, count=None, anomaly_rate=None):
    os.makedirs(out_dir, exist_ok=True)

    if regenerate:
        from data.generator import generate
        generate(count or 300, anomaly_rate if anomaly_rate is not None else 0.2, seed or 0, data_dir)

    pos, grn_by_po, invoices = load_data(data_dir)
    vendor_master = load_vendor_master(data_dir)
    employees_by_id = load_employees(data_dir)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = {l["invoice_id"]: l for l in json.load(f)}

    # fresh=True: a batch score has to be reproducible from the dataset alone,
    # not from whatever a previous eval run against the same --out-dir left
    # posted. See the module docstring.
    ledger = Ledger(os.path.join(out_dir, "ledger.json"), fresh=True)
    tools.reset_usage()

    results = []
    latencies = []
    exceptions = []
    for invoice in invoices:
        start = time.perf_counter()
        result = process_invoice(invoice, pos, grn_by_po, ledger,
                                 vendor_master=vendor_master, employees_by_id=employees_by_id)
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
    false_escalates = [r for r in escalated if labels[r["invoice_id"]]["expected_outcome"] == "auto_clear"]
    genuine_escalations = [r for r in escalated if labels[r["invoice_id"]]["expected_outcome"] == "escalate"]
    true_anomalies = [inv_id for inv_id, l in labels.items() if l["expected_outcome"] == "escalate"]

    confusion = _confusion_matrix(results, labels)
    per_type_recall = _per_type_recall(confusion)

    auto_clear_rate = len(auto_clear) / total if total else 0
    # Two false-clear denominators, because they answer different questions
    # and conflating them is exactly the kind of average that hides a problem:
    #   - "of what we cleared, how much was wrong" (precision-flavoured)
    #   - "of the real problems in the dataset, how many did we miss" (recall-
    #     flavoured, and the one closer to what a CFO actually loses sleep over)
    false_clear_rate_of_cleared = len(false_clears) / len(auto_clear) if auto_clear else 0
    false_clear_rate_of_anomalies = len(false_clears) / len(true_anomalies) if true_anomalies else 0
    false_escalate_rate = len(false_escalates) / total if total else 0
    escalation_precision = len(genuine_escalations) / len(escalated) if escalated else 0

    cost_weighted_total = len(false_clears) * fc_cost + len(false_escalates) * fe_cost
    cost_weighted_per_invoice = cost_weighted_total / total if total else 0

    total_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
    cost_inr = total_tokens / 1000 * tools.price_per_1k_inr()
    cost_per_invoice = cost_inr / total if total else 0
    p50_latency_ms = statistics.median(latencies) * 1000 if latencies else 0

    metrics = {
        "run_stamp": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "data_dir": os.path.abspath(data_dir),
            "seed": seed,
            "regenerated": regenerate,
            "llm_mode": os.environ.get("LEDGER_SENTINEL_LLM", "auto"),
        },
        "asymmetry": {
            "false_clear_cost": fc_cost,
            "false_escalate_cost": fe_cost,
            "ratio": round(fc_cost / fe_cost, 4) if fe_cost else None,
        },
        "total_invoices": total,
        "true_anomaly_count": len(true_anomalies),
        "auto_clear_rate": round(auto_clear_rate, 4),
        "false_clear_count": len(false_clears),
        "false_clear_rate_of_cleared": round(false_clear_rate_of_cleared, 4),
        "false_clear_rate_of_anomalies": round(false_clear_rate_of_anomalies, 4),
        "false_clear_invoice_ids": [r["invoice_id"] for r in false_clears],
        "false_escalate_count": len(false_escalates),
        "false_escalate_rate": round(false_escalate_rate, 4),
        "escalation_precision": round(escalation_precision, 4),
        "escalation_count": len(escalated),
        "cost_weighted_score_total": round(cost_weighted_total, 2),
        "cost_weighted_score_per_invoice": round(cost_weighted_per_invoice, 4),
        "confusion_matrix": confusion,
        "per_anomaly_type_recall": per_type_recall,
        "llm_calls": usage["calls"],
        "llm_tokens": total_tokens,
        "llm_fallbacks": usage["fallbacks"],
        "cost_per_invoice_inr": round(cost_per_invoice, 4),
        "p50_latency_ms": round(p50_latency_ms, 3),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _print_report(metrics):
    stamp = metrics["run_stamp"]
    print(f"run:            {stamp['generated_at_utc']} "
          f"(commit {stamp['git_commit'] or 'unknown'}, llm_mode={stamp['llm_mode']})")
    print(f"data:           {stamp['data_dir']}"
          + (f" (regenerated, seed={stamp['seed']})" if stamp["regenerated"] else ""))
    print()
    print(f"{'metric':<32}{'value'}")
    top_level = [
        "total_invoices", "true_anomaly_count", "auto_clear_rate",
        "false_clear_count", "false_clear_rate_of_cleared", "false_clear_rate_of_anomalies",
        "false_escalate_count", "false_escalate_rate",
        "escalation_precision", "escalation_count",
        "cost_weighted_score_total", "cost_weighted_score_per_invoice",
        "llm_calls", "llm_tokens", "llm_fallbacks", "cost_per_invoice_inr", "p50_latency_ms",
    ]
    for k in top_level:
        print(f"{k:<32}{metrics[k]}")
    print(f"\nasymmetry: false-clear costs {metrics['asymmetry']['ratio']}x a false-escalate "
          f"(fc={metrics['asymmetry']['false_clear_cost']}, fe={metrics['asymmetry']['false_escalate_cost']})")

    print(f"\n{'anomaly_type':<28}{'expected':<12}{'auto_clear':<12}{'escalate':<10}{'correct_rate'}")
    for atype, row in metrics["confusion_matrix"].items():
        rate = "n/a" if row["correct_rate"] is None else f"{row['correct_rate']:.0%}"
        print(f"{atype:<28}{row['expected_outcome']:<12}{row['auto_clear']:<12}{row['escalate']:<10}{rate}")

    if metrics["false_clear_invoice_ids"]:
        print(f"\nfalse-clear invoices ({len(metrics['false_clear_invoice_ids'])}): "
              f"{metrics['false_clear_invoice_ids']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fc-cost", type=float, default=DEFAULT_FALSE_CLEAR_COST,
                        help="Cost of one false-clear, in arbitrary units (default %(default)s).")
    parser.add_argument("--fe-cost", type=float, default=DEFAULT_FALSE_ESCALATE_COST,
                        help="Cost of one false-escalate, in the same units (default %(default)s).")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate the dataset at --data-dir before scoring it, from --seed/--count/--anomaly-rate.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--anomaly-rate", type=float, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="ledger_sentinel_eval_")
    metrics = run_eval(args.data_dir, out_dir, fc_cost=args.fc_cost, fe_cost=args.fe_cost,
                       regenerate=args.regenerate, seed=args.seed, count=args.count,
                       anomaly_rate=args.anomaly_rate)

    _print_report(metrics)
    print(f"\n(eval artifacts written to {out_dir})")


if __name__ == "__main__":
    main()
