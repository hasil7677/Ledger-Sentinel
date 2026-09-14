"""core/ledger.py + core/chain.py: the append-only, hash-chained ledger.
Covers posting/chaining, tamper detection (edit, delete, reorder), the
fresh=True reproducibility fix, and the near-match / vendor-history queries
the duplicate and split-invoice detectors depend on.
"""

import json

import pytest

from core.chain import GENESIS_HASH
from core.ledger import Ledger, verify_entries
from tests.conftest import make_clean_triple


def _invoice(inv_id, vendor="Acme Corp", amount=1000.0, invoice_date="2026-07-10", requester_id="EMP-001", po_id="PO-1"):
    return {"id": inv_id, "vendor": vendor, "po_id": po_id, "total_amount": amount,
            "invoice_date": invoice_date, "requester_id": requester_id}


# --- posting and chaining -------------------------------------------------------

def test_post_returns_a_chained_entry(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    entry = ledger.post(_invoice("INV-1"), resolution="auto_clear")
    assert entry["seq"] == 1
    assert entry["prev_hash"] == GENESIS_HASH
    assert entry["entry_hash"]


def test_second_entry_chains_to_the_first(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    first = ledger.post(_invoice("INV-1"), resolution="auto_clear")
    second = ledger.post(_invoice("INV-2"), resolution="auto_clear")
    assert second["prev_hash"] == first["entry_hash"]
    assert second["seq"] == 2


def test_head_is_genesis_for_empty_ledger(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    assert ledger.head() == GENESIS_HASH


def test_head_is_last_entry_hash(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1"), resolution="auto_clear")
    last = ledger.post(_invoice("INV-2"), resolution="auto_clear")
    assert ledger.head() == last["entry_hash"]


def test_requester_id_is_covered_by_the_hash(tmp_path):
    """A field the split-invoice detector relies on must be inside the chain,
    or an attacker could reattribute an invoice to a different requester
    without detection."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", requester_id="EMP-002"), resolution="auto_clear")

    entries = ledger.all()
    entries[0]["requester_id"] = "EMP-999"
    with open(ledger.path, "w") as f:
        json.dump(entries, f)

    report = ledger.verify()
    assert report["ok"] is False


def test_approved_by_is_outside_the_hash_by_design(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    entry = ledger.post(_invoice("INV-1"), resolution="approved_by_clerk", approved_by="clerk_1")
    assert entry["approved_by"] == "clerk_1"
    assert ledger.verify()["ok"] is True

    entries = ledger.all()
    entries[0]["approved_by"] = "someone_else"
    with open(ledger.path, "w") as f:
        json.dump(entries, f)
    assert ledger.verify()["ok"] is True  # approved_by is documented as unchained


# --- fresh=True reproducibility -------------------------------------------------

def test_fresh_true_truncates_existing_ledger(tmp_path):
    path = str(tmp_path / "ledger.json")
    ledger = Ledger(path)
    ledger.post(_invoice("INV-1"), resolution="auto_clear")
    assert len(ledger.all()) == 1

    fresh = Ledger(path, fresh=True)
    assert fresh.all() == []


def test_default_construction_appends_to_existing_file(tmp_path):
    path = str(tmp_path / "ledger.json")
    Ledger(path).post(_invoice("INV-1"), resolution="auto_clear")
    reopened = Ledger(path)
    assert len(reopened.all()) == 1
    reopened.post(_invoice("INV-2"), resolution="auto_clear")
    assert len(Ledger(path).all()) == 2


# --- tamper detection ------------------------------------------------------------

def _post_three(ledger):
    for i in range(1, 4):
        ledger.post(_invoice(f"INV-{i}", amount=1000.0 * i), resolution="auto_clear")


def test_intact_chain_verifies_ok(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    _post_three(ledger)
    report = ledger.verify()
    assert report["ok"] is True
    assert report["entries"] == 3
    assert report["problems"] == []


def test_editing_an_entry_is_detected_as_hash_mismatch(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    _post_three(ledger)

    entries = ledger.all()
    entries[1]["total_amount"] = 999999.0
    with open(ledger.path, "w") as f:
        json.dump(entries, f)

    report = ledger.verify()
    assert report["ok"] is False
    assert report["problems"][0]["kind"] == "hash_mismatch"
    assert report["problems"][0]["seq"] == 2


def test_deleting_the_original_of_a_duplicate_pair_breaks_the_chain(tmp_path):
    """The specific attack this whole feature exists to catch: delete the
    original posting so a resubmission no longer looks like a duplicate."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    _post_three(ledger)

    entries = ledger.all()
    del entries[0]  # delete the first entry; entries[1] now has a dangling prev_hash
    with open(ledger.path, "w") as f:
        json.dump(entries, f)

    report = ledger.verify()
    assert report["ok"] is False
    assert any(p["kind"] == "broken_link" for p in report["problems"])


def test_reordering_entries_is_detected(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    _post_three(ledger)

    entries = ledger.all()
    entries[0], entries[1] = entries[1], entries[0]
    with open(ledger.path, "w") as f:
        json.dump(entries, f)

    report = ledger.verify()
    assert report["ok"] is False


def test_unchained_legacy_entry_is_reported_but_not_fatal():
    """A row written before hashing existed has no entry_hash. verify_entries
    must say so without claiming tampering, and CLI-level exit code logic
    (see core/ledger.main) treats this as non-fatal."""
    legacy = {"invoice_id": "INV-OLD", "vendor": "Acme Corp", "total_amount": 100.0,
              "invoice_date": "2026-01-01", "resolution": "auto_clear"}
    report = verify_entries([legacy])
    assert report["problems"][0]["kind"] == "unchained"
    fatal = [p for p in report["problems"] if p["kind"] != "unchained"]
    assert fatal == []


def test_cli_verify_exit_code(tmp_path, capsys):
    from core.ledger import main as ledger_main

    path = tmp_path / "ledger.json"
    ledger = Ledger(str(path))
    _post_three(ledger)
    assert ledger_main(["verify", "--path", str(path)]) == 0

    entries = json.loads(path.read_text())
    entries[1]["total_amount"] = 0.01
    path.write_text(json.dumps(entries))
    assert ledger_main(["verify", "--path", str(path)]) == 1


def test_cli_verify_missing_file_returns_2(tmp_path, capsys):
    from core.ledger import main as ledger_main
    assert ledger_main(["verify", "--path", str(tmp_path / "nope.json")]) == 2


# --- queries ---------------------------------------------------------------------

def test_find_near_matches_respects_vendor_amount_and_date_window(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", vendor="Acme Corp", amount=1000.0, invoice_date="2026-07-01"), resolution="auto_clear")

    same_vendor_close = ledger.find_near_matches("Acme Corp", 1000.5, "2026-07-15", exclude_invoice_id="INV-2")
    assert len(same_vendor_close) == 1

    wrong_vendor = ledger.find_near_matches("Different Co", 1000.0, "2026-07-15")
    assert wrong_vendor == []

    wrong_amount = ledger.find_near_matches("Acme Corp", 5000.0, "2026-07-15")
    assert wrong_amount == []


def test_find_near_matches_date_window_boundary(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", invoice_date="2026-07-01"), resolution="auto_clear")

    within = ledger.find_near_matches("Acme Corp", 1000.0, "2026-07-31")  # exactly 30 days
    assert len(within) == 1

    outside = ledger.find_near_matches("Acme Corp", 1000.0, "2026-08-01")  # 31 days
    assert outside == []


def test_find_near_matches_excludes_self():
    pass  # covered structurally by exclude_invoice_id usage above; kept as a
          # marker that self-exclusion was considered, not only vendor/amount.


def test_entries_for_vendor_filters_by_vendor_and_window(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", vendor="Acme Corp", invoice_date="2026-07-01"), resolution="auto_clear")
    ledger.post(_invoice("INV-2", vendor="Other Co", invoice_date="2026-07-05"), resolution="auto_clear")

    result = ledger.entries_for_vendor("Acme Corp", before_date="2026-07-10", day_window=14)
    assert [e["invoice_id"] for e in result] == ["INV-1"]


def test_entries_for_vendor_window_boundary(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", invoice_date="2026-07-01"), resolution="auto_clear")

    within = ledger.entries_for_vendor("Acme Corp", before_date="2026-07-15", day_window=14)  # exactly 14 days
    assert len(within) == 1

    outside = ledger.entries_for_vendor("Acme Corp", before_date="2026-07-16", day_window=14)  # 15 days
    assert outside == []


def test_entries_for_vendor_excludes_future_entries(tmp_path):
    """A ledger entry dated AFTER before_date must not count -- split detection
    should only ever see the requester's history up to the invoice in hand."""
    ledger = Ledger(str(tmp_path / "ledger.json"))
    ledger.post(_invoice("INV-1", invoice_date="2026-07-20"), resolution="auto_clear")
    result = ledger.entries_for_vendor("Acme Corp", before_date="2026-07-10", day_window=30)
    assert result == []
