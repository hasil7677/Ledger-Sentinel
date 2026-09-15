"""Two claims about the eval that tests/test_eval.py does not check directly:

1. **Byte-identical reproducibility.** test_eval.py::test_reproducible_from_fixed_seed
   already checks that two seed=99 runs produce identical *metrics*. It does
   not check the dataset generation step in isolation, and "the metrics
   matched" is compatible with two different datasets that happen to score
   the same. This file regenerates the dataset itself twice, from the same
   seed, to two different directories, and diffs the files byte for byte --
   the strongest form of the reproducibility claim, and the one a skeptical
   reviewer would ask for first.

2. **Seed stability of the headline number.** The README reports one number
   -- an 8.2% false-clear rate -- from one seed (42). A single seed is not
   evidence that the number is a property of the system rather than a
   property of that seed's particular draw of invoices. This file reruns the
   full eval across 8 different seeds and asserts the false-clear rate stays
   within a documented range, so a future change that pushes it outside that
   range fails a test instead of silently drifting past what
   EVAL_METHODOLOGY.md claims.

Marked `eval_full`: each seed regenerates and scores a fresh 300-invoice
dataset, so this file is measurably slower (~2-4s) than the rest of the unit
suite. Still included in a plain `pytest` run; run on its own with
`pytest -m eval_full -v`.
"""

import json

import pytest

from data.generator import generate
from eval.run_eval import run_eval

pytestmark = pytest.mark.eval_full

DATASET_FILES = ("pos.json", "grns.json", "invoices.json", "labels.json", "vendors.json", "employees.json")

# Seeds used for the stability check. Deliberately not just "42 plus small
# perturbations" -- includes the seed the README's headline table is built
# from (42) alongside seven others with no special relationship to it.
STABILITY_SEEDS = [1, 2, 3, 42, 99, 123, 777, 2026]


# --- 1. byte-identical dataset regeneration ----------------------------------

def test_dataset_generation_is_byte_identical_across_two_separate_runs(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    generate(count=300, anomaly_rate=0.2, seed=42, out_dir=str(dir_a))
    generate(count=300, anomaly_rate=0.2, seed=42, out_dir=str(dir_b))

    for fname in DATASET_FILES:
        bytes_a = (dir_a / fname).read_bytes()
        bytes_b = (dir_b / fname).read_bytes()
        assert bytes_a == bytes_b, f"{fname} differs between two seed=42 generation runs"


def test_dataset_generation_is_byte_identical_across_several_seeds_and_sizes(tmp_path):
    """Not just seed=42 -- several (seed, count) combinations, so "the
    generator is reproducible" is a property of the generator, not a fact
    about one seed that happened to avoid whatever nondeterminism exists."""
    for i, (seed, count) in enumerate([(1, 60), (7, 150), (42, 300), (2026, 90)]):
        dir_a = tmp_path / f"case{i}_a"
        dir_b = tmp_path / f"case{i}_b"
        generate(count=count, anomaly_rate=0.2, seed=seed, out_dir=str(dir_a))
        generate(count=count, anomaly_rate=0.2, seed=seed, out_dir=str(dir_b))
        for fname in DATASET_FILES:
            assert (dir_a / fname).read_bytes() == (dir_b / fname).read_bytes(), (
                f"seed={seed} count={count}: {fname} differs between two runs"
            )


def test_dataset_content_is_not_trivially_identical_across_different_seeds(tmp_path):
    """Guard against a vacuous pass: if the generator ignored `seed` entirely,
    every test above would also pass. Confirm two different seeds actually
    produce different invoice data."""
    dir_a = tmp_path / "seed_a"
    dir_b = tmp_path / "seed_b"
    generate(count=150, anomaly_rate=0.2, seed=1, out_dir=str(dir_a))
    generate(count=150, anomaly_rate=0.2, seed=2, out_dir=str(dir_b))
    assert (dir_a / "invoices.json").read_bytes() != (dir_b / "invoices.json").read_bytes()


# --- 2. seed-stability of the headline false-clear number --------------------

@pytest.fixture(scope="module")
def multi_seed_metrics(tmp_path_factory):
    """Run the full eval (regenerate + score) once per seed in STABILITY_SEEDS
    and cache the result for the whole module -- these are expensive enough
    (full 300-invoice generate + process_invoice pass) that re-running them
    per test would make this file noticeably slower for no extra coverage."""
    results = {}
    for seed in STABILITY_SEEDS:
        data_dir = tmp_path_factory.mktemp(f"data_{seed}")
        out_dir = tmp_path_factory.mktemp(f"out_{seed}")
        results[seed] = run_eval(str(data_dir), str(out_dir), regenerate=True,
                                 seed=seed, count=300, anomaly_rate=0.2)
    return results


def test_false_clear_rate_of_anomalies_stays_within_the_documented_range(multi_seed_metrics):
    """EVAL_METHODOLOGY.md reports the range this asserts. If a change to the
    matcher, the orchestrator, or the generator's anomaly mix moves the range,
    that document goes stale -- this test is what notices.

    The range is empirical, not a target: it was measured by running exactly
    this fixture and copied into EVAL_METHODOLOGY.md, not chosen first and
    then made to pass. Widen it only alongside updating that document, with
    the same honesty the README already applies to the base 8.2% figure --
    never narrow it to make a change look more stable than it is.
    """
    rates = {seed: m["false_clear_rate_of_anomalies"] for seed, m in multi_seed_metrics.items()}
    assert min(rates.values()) >= 0.04, rates
    assert max(rates.values()) <= 0.11, rates
    # seed=42 is the one number the README puts in its headline table --
    # pin it exactly so a regression there is caught precisely, not just
    # detected as "somewhere in range."
    assert multi_seed_metrics[42]["false_clear_rate_of_anomalies"] == 0.0816


def test_false_clears_across_all_seeds_are_only_the_two_documented_structural_types(multi_seed_metrics):
    """Same check as test_eval.py's single-seed version, run across every
    stability seed: every false clear must trace back to threshold_splitting
    (retrospective-by-construction) or price_variance_out_tolerance (the
    Rs 500 floor edge case). If any OTHER type produces a false clear at ANY
    seed, that is a real regression a single-seed run could miss entirely."""
    known_types = {"threshold_splitting", "price_variance_out_tolerance"}
    for seed, metrics in multi_seed_metrics.items():
        unexpected = {
            atype: row["false_clear_count"]
            for atype, row in metrics["confusion_matrix"].items()
            if row["false_clear_count"] > 0 and atype not in known_types
        }
        assert unexpected == {}, f"seed={seed}: unexpected false-clear type(s): {unexpected}"


def test_escalation_precision_is_not_actually_100_percent_across_seeds(multi_seed_metrics):
    """The honest, inconvenient counterpart to tests/test_eval.py::
    test_escalation_precision_is_100_percent, which only holds at seed=42.

    Running the eval across 8 seeds found escalation_precision drops below
    100% at 2 of them (seed=3, seed=99): one `vendor_name_drift` invoice
    escalates instead of auto-clearing, because the seeded alias pair "Acme
    Corp" / "Acme Corporation Pvt Ltd" has a difflib.SequenceMatcher ratio of
    0.545 (see agent/heuristics.py::VENDOR_SIMILARITY_THRESHOLD = 0.6) --
    below the stub heuristic's own resolution threshold. When that specific
    pair is drawn for a vendor_name_drift anomaly, the heuristic correctly
    reports low confidence that the names are the same entity, and
    agent/orchestrator.py correctly escalates a low-confidence vendor match
    for human review -- which is the *right* behaviour for the heuristic
    given what it was actually asked, but contradicts the dataset's ground
    truth label of "should always auto-clear" for this anomaly type.

    This is not a detector bug (core/anomalies.py, core/matcher.py): the
    escalation is correct given the confidence. It is a labelling gap in the
    eval, same shape as the two already-documented false-clear limitations --
    found here specifically because this file checks more than one seed,
    which is the entire reason EVAL_METHODOLOGY.md asks for a seed range
    instead of a single number.
    """
    imprecise_seeds = {
        seed: m["escalation_precision"]
        for seed, m in multi_seed_metrics.items()
        if m["escalation_precision"] < 1.0
    }
    # Document the exact, current shape of the finding rather than merely its
    # existence: assert on precisely which seeds and how many false escalates,
    # so a future run that finds MORE of them (a real regression, e.g. the
    # heuristic threshold or a different vendor pair drifting under it) fails
    # loudly instead of being masked by a vague ">= 0" assertion.
    assert imprecise_seeds == {3: 0.9787, 99: 0.9787}
    for seed in imprecise_seeds:
        confusion = multi_seed_metrics[seed]["confusion_matrix"]
        offending = {
            atype: row["false_escalate_count"]
            for atype, row in confusion.items()
            if row["false_escalate_count"] > 0
        }
        assert offending == {"vendor_name_drift": 1}, f"seed={seed}: {offending}"


def test_true_anomaly_count_is_stable_across_seeds(multi_seed_metrics):
    """scale_anomaly_plan's largest-remainder allocation is deterministic
    given (count, anomaly_rate) and does not depend on the seed -- only WHICH
    invoices get which anomaly does. 49 true anomalies at every seed is the
    baseline the false-clear-rate-of-anomalies denominator rests on."""
    counts = {seed: m["true_anomaly_count"] for seed, m in multi_seed_metrics.items()}
    assert set(counts.values()) == {49}, counts
