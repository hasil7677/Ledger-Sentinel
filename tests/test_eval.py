"""eval/run_eval.py: the rebuilt batch eval. Covers the shape of the new
metrics (confusion matrix, per-type recall, the cost-weighted asymmetry
score), reproducibility from a fixed seed, and the two bugs this rebuild
fixed in the old eval -- a stale ledger contaminating a second run against
the same --out-dir, and vendor-master/approval detectors never running at
all because they weren't wired into the eval's process_invoice calls.
"""

import json

import pytest

from eval.run_eval import run_eval


@pytest.fixture(scope="module")
def eval_result(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("eval_data")
    out_dir = tmp_path_factory.mktemp("eval_out")
    metrics = run_eval(str(data_dir), str(out_dir), regenerate=True, seed=42, count=300, anomaly_rate=0.2)
    return {"metrics": metrics, "data_dir": data_dir, "out_dir": out_dir}


def test_total_invoices_matches_dataset_size(eval_result):
    assert eval_result["metrics"]["total_invoices"] == 300


def test_confusion_matrix_covers_every_seeded_anomaly_type(eval_result):
    with open(eval_result["data_dir"] / "labels.json") as f:
        labels = json.load(f)
    seeded_types = {l["anomaly_type"] for l in labels}
    assert seeded_types == set(eval_result["metrics"]["confusion_matrix"])


def test_confusion_matrix_rows_sum_to_their_total(eval_result):
    for atype, row in eval_result["metrics"]["confusion_matrix"].items():
        assert row["auto_clear"] + row["escalate"] == row["total"]


def test_per_type_recall_only_covers_escalate_expected_types(eval_result):
    confusion = eval_result["metrics"]["confusion_matrix"]
    recall = eval_result["metrics"]["per_anomaly_type_recall"]
    escalate_types = {t for t, row in confusion.items() if row["expected_outcome"] == "escalate"}
    assert set(recall) == escalate_types
    assert "clean" not in recall  # clean's correct outcome is auto_clear, not escalate


def test_escalation_precision_is_100_percent(eval_result):
    """Every escalation in this dataset traces back to a genuinely seeded
    anomaly (the generator never seeds an anomaly-free invoice that also
    triggers a hard rule) -- if this drops, something is escalating on a
    clean invoice for the wrong reason."""
    assert eval_result["metrics"]["escalation_precision"] == 1.0


def test_false_escalate_rate_is_zero_on_this_dataset(eval_result):
    assert eval_result["metrics"]["false_escalate_count"] == 0


def test_reported_false_clears_are_the_known_documented_limitations_only(eval_result):
    """As of this generator, the only true false clears are the first half of
    each threshold_splitting pair (detection is inherently retrospective) and
    a rare price_variance_out_tolerance line whose rupee impact lands under
    the Rs 500 floor even at the maximum seeded bump. If a false clear shows
    up in any OTHER anomaly type, that's a real regression, not a known
    limitation, and this test should fail."""
    confusion = eval_result["metrics"]["confusion_matrix"]
    unexpected = {
        atype: row["false_clear_count"]
        for atype, row in confusion.items()
        if row["false_clear_count"] > 0 and atype not in ("threshold_splitting", "price_variance_out_tolerance")
    }
    assert unexpected == {}


def test_cost_weighted_score_reflects_the_configured_asymmetry(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    out_a = tmp_path_factory.mktemp("out_a")
    out_b = tmp_path_factory.mktemp("out_b")
    metrics_default = run_eval(str(data_dir), str(out_a), regenerate=True, seed=5, count=150,
                               anomaly_rate=0.2, fc_cost=50.0, fe_cost=1.0)
    # Re-score the SAME already-generated dataset (regenerate=False) with a
    # different asymmetry to isolate the cost formula from dataset variation.
    metrics_symmetric = run_eval(str(data_dir), str(out_b), regenerate=False, fc_cost=1.0, fe_cost=1.0)

    assert metrics_default["asymmetry"]["ratio"] == 50.0
    assert metrics_symmetric["asymmetry"]["ratio"] == 1.0
    if metrics_default["false_clear_count"] > 0:
        assert metrics_default["cost_weighted_score_total"] > metrics_symmetric["cost_weighted_score_total"]


def test_reproducible_from_fixed_seed(tmp_path_factory):
    data_a = tmp_path_factory.mktemp("data_a")
    data_b = tmp_path_factory.mktemp("data_b")
    out_a = tmp_path_factory.mktemp("out_a")
    out_b = tmp_path_factory.mktemp("out_b")

    metrics_a = run_eval(str(data_a), str(out_a), regenerate=True, seed=99, count=150, anomaly_rate=0.2)
    metrics_b = run_eval(str(data_b), str(out_b), regenerate=True, seed=99, count=150, anomaly_rate=0.2)

    # run_stamp holds an absolute tmp-dir path and a wall-clock timestamp;
    # p50_latency_ms is a wall-clock measurement -- neither is expected to be
    # bit-identical across two separate runs, only everything the pipeline
    # actually computes from the (identical, seed-derived) data.
    comparable_keys = [k for k in metrics_a if k not in ("run_stamp", "p50_latency_ms")]
    for key in comparable_keys:
        assert metrics_a[key] == metrics_b[key], f"metric {key!r} differs between two seed=99 runs"


def test_eval_does_not_leak_state_across_out_dirs(tmp_path_factory):
    """The bug this rebuild fixed: the old eval built its Ledger without
    fresh=True, so a second run against a stale --out-dir inherited the first
    run's postings. Running twice against the SAME out_dir must produce
    identical results to running once."""
    data_dir = tmp_path_factory.mktemp("data")
    out_dir = tmp_path_factory.mktemp("out")

    first = run_eval(str(data_dir), str(out_dir), regenerate=True, seed=11, count=150, anomaly_rate=0.2)
    second = run_eval(str(data_dir), str(out_dir), regenerate=False)  # same out_dir, same data, again

    assert first["escalation_count"] == second["escalation_count"]
    assert first["false_clear_count"] == second["false_clear_count"]
    assert first["confusion_matrix"] == second["confusion_matrix"]


def test_stamped_artifact_is_written(eval_result):
    metrics_path = eval_result["out_dir"] / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        written = json.load(f)
    stamp = written["run_stamp"]
    assert stamp["generated_at_utc"]
    assert stamp["seed"] == 42
    assert stamp["regenerated"] is True


def test_run_stamp_records_llm_mode(eval_result):
    assert eval_result["metrics"]["run_stamp"]["llm_mode"] == "stub"
