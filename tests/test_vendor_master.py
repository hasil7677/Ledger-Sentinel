"""core/vendor_master.py: vendor records, the hash-chained change log, the
bank-detail-change hold, first-invoice-from-new-vendor handling, and
vendor/employee collision detection.
"""

import pytest

from core.vendor_master import VendorMaster, mask_account


def _vendor(vm, vendor_id="VEND-001", onboarded="2020-01-01"):
    return vm.add_vendor(
        vendor_id=vendor_id, canonical_name="Acme Corp", bank_account="111122223333",
        bank_ifsc="HDFC0001001", remit_to_name="Acme Corp",
        contact_email="accounts@acme.example", onboarded_date=onboarded,
        aliases=["Acme Corporation Pvt Ltd"],
    )


# --- lookup --------------------------------------------------------------------

def test_by_name_finds_canonical_and_alias():
    vm = VendorMaster.empty()
    _vendor(vm)
    assert vm.by_name("Acme Corp")["vendor_id"] == "VEND-001"
    assert vm.by_name("Acme Corporation Pvt Ltd")["vendor_id"] == "VEND-001"


def test_by_name_is_case_and_punctuation_insensitive():
    vm = VendorMaster.empty()
    _vendor(vm)
    assert vm.by_name("  ACME, CORP.  ") is not None


def test_by_name_miss_returns_none():
    vm = VendorMaster.empty()
    _vendor(vm)
    assert vm.by_name("Completely Different Vendor") is None


def test_empty_master_never_finds_anything():
    vm = VendorMaster.empty()
    assert vm.by_name("Acme Corp") is None


# --- change log / current_details ----------------------------------------------

def test_current_details_starts_at_onboarded_values():
    vm = VendorMaster.empty()
    _vendor(vm)
    details = vm.current_details("VEND-001")
    assert details["bank_account"] == "111122223333"


def test_record_change_updates_current_details_immediately():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999988887777", effective_at="2026-07-01",
                     actor="email_request", channel="email")
    assert vm.current_details("VEND-001")["bank_account"] == "999988887777"


def test_record_change_captures_old_value_from_folded_state_not_caller():
    vm = VendorMaster.empty()
    _vendor(vm)
    event = vm.record_change("VEND-001", "bank_account", "999988887777", effective_at="2026-07-01")
    assert event["old_value"] == "111122223333"


# --- bank-detail-change hold -----------------------------------------------------

def test_unverified_bank_change_is_pending():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999988887777", effective_at="2026-07-01",
                     actor="email_request", channel="email")
    pending = vm.pending_payment_changes("VEND-001", as_of="2026-07-05")
    assert len(pending) == 1
    assert pending[0]["field"] == "bank_account"


def test_verified_bank_change_is_no_longer_pending():
    vm = VendorMaster.empty()
    _vendor(vm)
    change = vm.record_change("VEND-001", "bank_account", "999988887777", effective_at="2026-07-01",
                              actor="email_request", channel="email")
    vm.record_verification("VEND-001", verifies_seq=change["seq"], actor="ap_clerk_1",
                           effective_at="2026-07-02", channel="phone_callback")
    assert vm.pending_payment_changes("VEND-001", as_of="2026-07-05") == []


def test_verification_is_scoped_to_the_specific_change_it_verifies():
    """Two separate bank-account changes; verifying the first must not clear
    the second."""
    vm = VendorMaster.empty()
    _vendor(vm)
    first = vm.record_change("VEND-001", "bank_account", "111", effective_at="2026-07-01")
    vm.record_verification("VEND-001", verifies_seq=first["seq"], actor="ap_clerk_1", effective_at="2026-07-02")
    second = vm.record_change("VEND-001", "bank_account", "222", effective_at="2026-07-10")

    pending = vm.pending_payment_changes("VEND-001", as_of="2026-07-15")
    assert len(pending) == 1
    assert pending[0]["seq"] == second["seq"]


def test_record_verification_unknown_seq_raises():
    vm = VendorMaster.empty()
    _vendor(vm)
    with pytest.raises(KeyError):
        vm.record_verification("VEND-001", verifies_seq=999, actor="x", effective_at="2026-07-01")


def test_non_payment_field_change_does_not_create_a_pending_hold():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "contact_email", "new@acme.example", effective_at="2026-07-01")
    assert vm.pending_payment_changes("VEND-001", as_of="2026-07-05") == []


def test_pending_changes_respects_as_of_cutoff():
    """A change effective in the future relative to `as_of` should not yet
    count as pending against an earlier invoice."""
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999", effective_at="2026-07-20")
    assert vm.pending_payment_changes("VEND-001", as_of="2026-07-05") == []
    assert len(vm.pending_payment_changes("VEND-001", as_of="2026-07-25")) == 1


# --- new-vendor window -----------------------------------------------------------

def test_is_new_vendor_true_within_window():
    vm = VendorMaster.empty()
    _vendor(vm, onboarded="2026-06-01")
    assert vm.is_new_vendor("VEND-001", as_of="2026-06-15", window_days=60) is True


def test_is_new_vendor_false_after_window():
    vm = VendorMaster.empty()
    _vendor(vm, onboarded="2026-01-01")
    assert vm.is_new_vendor("VEND-001", as_of="2026-06-15", window_days=60) is False


def test_is_new_vendor_boundary_at_exactly_window_days():
    vm = VendorMaster.empty()
    _vendor(vm, onboarded="2026-01-01")
    assert vm.is_new_vendor("VEND-001", as_of="2026-03-02", window_days=60) is True  # exactly 60 days
    assert vm.is_new_vendor("VEND-001", as_of="2026-03-03", window_days=60) is False  # 61 days


def test_is_new_vendor_false_for_a_date_before_onboarding():
    vm = VendorMaster.empty()
    _vendor(vm, onboarded="2026-06-01")
    assert vm.is_new_vendor("VEND-001", as_of="2026-05-01") is False


# --- employee collision -----------------------------------------------------------

def test_employee_collision_on_bank_account():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "500100010007", effective_at="2020-01-01")
    employees = [{"employee_id": "EMP-007", "name": "Priya Nair", "bank_account": "500100010007", "email": "priya@x"}]
    collision = vm.employee_collision("VEND-001", employees)
    assert collision["match_field"] == "bank_account"
    assert collision["employee_id"] == "EMP-007"


def test_employee_collision_on_email_when_no_bank_match():
    vm = VendorMaster.empty()
    vm.add_vendor(vendor_id="VEND-002", canonical_name="Shell Co", bank_account="000",
                  bank_ifsc="X", remit_to_name="Shell Co", contact_email="priya@x",
                  onboarded_date="2020-01-01")
    employees = [{"employee_id": "EMP-007", "name": "Priya Nair", "bank_account": "999", "email": "priya@x"}]
    collision = vm.employee_collision("VEND-002", employees)
    assert collision["match_field"] == "contact_email"


def test_no_collision_returns_none():
    vm = VendorMaster.empty()
    _vendor(vm)
    employees = [{"employee_id": "EMP-999", "name": "Unrelated Person", "bank_account": "0", "email": "x@x"}]
    assert vm.employee_collision("VEND-001", employees) is None


def test_no_collision_with_empty_employee_list():
    vm = VendorMaster.empty()
    _vendor(vm)
    assert vm.employee_collision("VEND-001", []) is None


# --- masking -----------------------------------------------------------------------

@pytest.mark.parametrize("account,expected", [
    ("123456789012", "********9012"),
    ("1234", "****"),
    ("12", "**"),
    ("", "(none)"),
    (None, "(none)"),
])
def test_mask_account(account, expected):
    assert mask_account(account) == expected


# --- hash-chain integrity ------------------------------------------------------------

def test_fresh_master_verifies_ok():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999", effective_at="2026-07-01")
    report = vm.verify_chain()
    assert report["ok"] is True


def test_tampering_with_a_change_event_is_detected():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999", effective_at="2026-07-01")
    record = vm.by_id("VEND-001")
    record["changes"][-1]["new_value"] = "TAMPERED"  # bypass the API, edit the raw dict directly

    report = vm.verify_chain()
    assert report["ok"] is False
    kinds = {p["kind"] for p in report["problems"]}
    assert "hash_mismatch" in kinds
    assert report["problems"][0]["vendor_id"] == "VEND-001"


def test_deleting_a_change_event_breaks_the_chain():
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "111", effective_at="2026-07-01")
    vm.record_change("VEND-001", "bank_account", "222", effective_at="2026-07-05")
    record = vm.by_id("VEND-001")
    del record["changes"][1]  # remove the middle "111" change event

    report = vm.verify_chain()
    assert report["ok"] is False
    assert any(p["kind"] == "broken_link" for p in report["problems"])


def test_save_and_load_round_trip(tmp_path):
    vm = VendorMaster.empty()
    _vendor(vm)
    vm.record_change("VEND-001", "bank_account", "999", effective_at="2026-07-01")
    path = str(tmp_path / "vendors.json")
    vm.save(path)

    reloaded = VendorMaster.load(path)
    assert reloaded.by_name("Acme Corp")["vendor_id"] == "VEND-001"
    assert reloaded.verify_chain()["ok"] is True
