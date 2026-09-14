"""data/generator.py: the synthetic dataset generator. Covers the invariants
the rest of the suite and the eval depend on -- exact invoice count, every
anomaly type actually landing on the correct side of the matcher's own
tolerance rules, no fraud-vendor name collisions (a real bug found via the
eval, see the comment above fraud_vendor_names), reproducibility, and that
the default requester can never accidentally trip an approval-policy check.
"""

from collections import Counter

import pytest

from core.approval import SPLIT_WINDOW_DAYS
from core.matcher import DEFAULT_TOLERANCES, match_invoice
from core.vendor_master import VendorMaster
from data.generator import (
    DEFAULT_REQUESTER_ID,
    EMPLOYEES,
    VENDORS,
    generate,
)


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("ls_gen")
    pos, grns, invoices, labels = generate(count=300, anomaly_rate=0.2, seed=42, out_dir=str(out_dir))
    return {"out_dir": out_dir, "pos": pos, "grns": grns, "invoices": invoices, "labels": labels}


def test_total_invoice_count_matches_requested_count(small_dataset):
    assert len(small_dataset["invoices"]) == 300
    assert len(small_dataset["labels"]) == 300


def test_every_invoice_has_a_label(small_dataset):
    label_ids = {l["invoice_id"] for l in small_dataset["labels"]}
    invoice_ids = {i["id"] for i in small_dataset["invoices"]}
    assert label_ids == invoice_ids


def test_taxonomy_is_proportionally_represented(small_dataset):
    by_type = Counter(l["anomaly_type"] for l in small_dataset["labels"])
    # At count=300/rate=0.2 (60 anomalies over 13 weighted types), every type
    # should have at least one instance -- this is the taxonomy the eval's
    # confusion matrix depends on being populated.
    expected_types = {
        "clean", "quantity_variance", "price_variance_in_tolerance", "price_variance_out_tolerance",
        "duplicate", "missing_po", "vendor_name_drift", "tax_mismatch", "line_item_mismatch",
        "vendor_bank_detail_change", "new_vendor_first_invoice", "vendor_employee_collision",
        "approval_threshold_exceeded", "threshold_splitting",
    }
    assert expected_types <= set(by_type)
    for atype, count in by_type.items():
        assert count > 0


def test_reproducible_with_same_seed(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("a")
    dir_b = tmp_path_factory.mktemp("b")
    _, _, inv_a, labels_a = generate(count=120, anomaly_rate=0.2, seed=7, out_dir=str(dir_a))
    _, _, inv_b, labels_b = generate(count=120, anomaly_rate=0.2, seed=7, out_dir=str(dir_b))
    assert inv_a == inv_b
    assert labels_a == labels_b


def test_different_seeds_produce_different_datasets(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("a")
    dir_b = tmp_path_factory.mktemp("b")
    _, _, inv_a, _ = generate(count=120, anomaly_rate=0.2, seed=1, out_dir=str(dir_a))
    _, _, inv_b, _ = generate(count=120, anomaly_rate=0.2, seed=2, out_dir=str(dir_b))
    assert inv_a != inv_b


def test_vendors_json_and_employees_json_are_written(small_dataset):
    out_dir = small_dataset["out_dir"]
    assert (out_dir / "vendors.json").exists()
    assert (out_dir / "employees.json").exists()


def test_vendor_master_hash_chain_verifies_immediately_after_generation(small_dataset):
    vm = VendorMaster.load(str(small_dataset["out_dir"] / "vendors.json"))
    report = vm.verify_chain()
    assert report["ok"] is True


def test_no_two_fraud_vendors_share_a_name(small_dataset):
    """Regression test for a real bug: two independently-shuffled name pools
    could draw the same combo, and the second add_vendor() would silently
    overwrite the first in VendorMaster._by_name, misrouting every invoice
    for the first vendor onto the second vendor's state."""
    vm = VendorMaster.load(str(small_dataset["out_dir"] / "vendors.json"))
    names = [r["canonical_name"] for r in vm.records]
    counts = Counter(names)
    dupes = {name: n for name, n in counts.items() if n > 1}
    assert dupes == {}


def test_fraud_vendor_names_never_collide_with_canonical_vendors(small_dataset):
    vm = VendorMaster.load(str(small_dataset["out_dir"] / "vendors.json"))
    canonical_names = {v[0] for v in VENDORS} | {v[1] for v in VENDORS}
    fraud_names = {r["canonical_name"] for r in vm.records if r["vendor_id"].split("-")[1].isdigit()
                  and int(r["vendor_id"].split("-")[1]) > 100}
    assert canonical_names.isdisjoint(fraud_names)


def test_every_requester_id_is_a_known_employee(small_dataset):
    known = {e["employee_id"] for e in EMPLOYEES}
    for inv in small_dataset["invoices"]:
        assert inv.get("requester_id") in known


def test_default_requester_limit_exceeds_the_highest_possible_invoice_total():
    """The generator's own safety margin: EMP-001's limit must stay well above
    anything three lines at the maximum product price and quantity can produce,
    or an ordinary (non-approval-anomaly) invoice could accidentally trip the
    approval-threshold check and silently corrupt the "clean" label."""
    from data.generator import PRODUCTS

    max_line = max(price for _, _, price, _ in PRODUCTS) * 200 * 1.18  # max unit_price * max qty * max GST
    worst_case_total = max_line * 3  # up to 3 lines per invoice
    default_limit = next(e["approval_limit"] for e in EMPLOYEES if e["employee_id"] == DEFAULT_REQUESTER_ID)
    assert default_limit > worst_case_total


def test_quantity_variance_anomalies_are_always_outside_the_one_unit_tolerance(small_dataset):
    po_by_id = {p["id"]: p for p in small_dataset["pos"]}
    grn_by_po = {g["po_id"]: g for g in small_dataset["grns"]}
    inv_by_id = {i["id"]: i for i in small_dataset["invoices"]}
    labels_by_id = {l["invoice_id"]: l for l in small_dataset["labels"]}

    checked = 0
    for label in small_dataset["labels"]:
        if label["anomaly_type"] != "quantity_variance":
            continue
        invoice = inv_by_id[label["invoice_id"]]
        po = po_by_id[invoice["po_id"]]
        grn = grn_by_po[invoice["po_id"]]
        result = match_invoice(invoice, po, grn)
        qty_variances = [v for v in result["variances"] if v["type"] == "quantity_variance"]
        assert qty_variances, f"{invoice['id']} labelled quantity_variance but matcher found no such variance"
        assert all(not v["within_tolerance"] for v in qty_variances)
        checked += 1
    assert checked > 0


def test_price_variance_in_tolerance_anomalies_actually_clear_the_matcher(small_dataset):
    po_by_id = {p["id"]: p for p in small_dataset["pos"]}
    grn_by_po = {g["po_id"]: g for g in small_dataset["grns"]}
    inv_by_id = {i["id"]: i for i in small_dataset["invoices"]}

    checked = 0
    for label in small_dataset["labels"]:
        if label["anomaly_type"] != "price_variance_in_tolerance":
            continue
        invoice = inv_by_id[label["invoice_id"]]
        po = po_by_id[invoice["po_id"]]
        grn = grn_by_po.get(invoice["po_id"])
        result = match_invoice(invoice, po, grn)
        assert result["clean"] is True
        checked += 1
    assert checked > 0
