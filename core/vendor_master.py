"""Vendor master records and their change history.

The project had no vendor master at all, which meant it could not see the
single most expensive attack in accounts payable. It is not the duplicate
invoice -- a duplicate is a known amount to a known account, and it is
recoverable. It is the **bank-account change**: an attacker gets into an email
thread, sends a plausible note on vendor letterhead saying "our banking has
moved", AP updates the payment details, and the next genuine invoice for a
genuine delivery is paid to the attacker. Every document in the three-way
match is correct. The PO matches, the goods arrived, the invoice is right.
Three-way match cannot see it, because nothing about it is a mismatch.

What catches it is comparing the payment details on the invoice against the
vendor's details of record, and refusing to pay on a change that has not been
confirmed out of band -- a phone call to the number already on file, not a
number supplied by whoever requested the change.

The change log is an append-only event log, hash-chained with the same
primitive as the ledger (`core/chain.py`). Three kinds of event:

  onboarded     the vendor's opening details
  change        a field changed, with old and new values and who asked
  verification  a human confirmed a specific earlier change out of band

Verification is a separate appended *event*, not a mutable flag on the change
record. That matters: if "verified" were a boolean on the change, flipping it
would be one character of edit, and flipping it is exactly what an attacker who
has reached the file would do. As an appended event it is inside the chain, so
flipping it means rewriting the chain from that point on.
"""

import json
import os
from datetime import date

from core import chain

CHANGE_FIELDS = (
    "seq",
    "vendor_id",
    "kind",
    "field",
    "old_value",
    "new_value",
    "effective_at",
    "actor",
    "channel",
    "verifies_seq",
    "note",
    "prev_hash",
)

# Payment fields. A change to any of these stops payment until it is verified
# out of band; a change to anything else (address, contact name) does not.
PAYMENT_FIELDS = ("bank_account", "bank_ifsc", "remit_to_name")

# How long a vendor counts as new. First payment to a vendor onboarded inside
# this window goes to a human regardless of how clean the invoice is.
NEW_VENDOR_DAYS = 60


def _norm(name):
    return " ".join(str(name).lower().replace(".", " ").replace(",", " ").split())


class VendorMaster:
    """A collection of vendor records, loaded from and saved to JSON."""

    def __init__(self, records, path=None):
        self.path = path
        self.records = [dict(r) for r in records]
        self._by_id = {r["vendor_id"]: r for r in self.records}
        self._by_name = {}
        for r in self.records:
            self._by_name[_norm(r["canonical_name"])] = r
            for alias in r.get("aliases", []):
                self._by_name.setdefault(_norm(alias), r)

    # --- loading / saving ----------------------------------------------------

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return cls([], path=path)
        with open(path) as f:
            return cls(json.load(f), path=path)

    @classmethod
    def empty(cls):
        """A master with no vendors in it. Every lookup misses, so every
        vendor-master control declines to fire rather than firing on nothing --
        which is what keeps the three-way-match pipeline runnable without one."""
        return cls([])

    def save(self, path=None):
        target = path or self.path
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w") as f:
            json.dump(self.records, f, indent=2)

    def __len__(self):
        return len(self.records)

    # --- lookup --------------------------------------------------------------

    def by_id(self, vendor_id):
        return self._by_id.get(vendor_id)

    def by_name(self, vendor_name):
        """Look up by canonical name or a known alias.

        Deliberately exact-after-normalisation. Fuzzy vendor matching is a
        judgement call and it lives in the agent layer; a fraud control that
        guessed which vendor it was looking at would be worse than no control.
        """
        return self._by_name.get(_norm(vendor_name))

    # --- derived state -------------------------------------------------------

    def current_details(self, vendor_id):
        """Fold the change log into the vendor's details as they stand now.

        Includes unverified changes: the master reflects what someone actually
        changed. Whether the change is safe to pay against is a separate
        question, answered by `pending_payment_changes`.
        """
        record = self._by_id[vendor_id]
        details = {f: record.get(f) for f in PAYMENT_FIELDS}
        details["contact_email"] = record.get("contact_email")
        for event in record.get("changes", []):
            if event["kind"] == "change" and event["field"] in details:
                details[event["field"]] = event["new_value"]
        return details

    def verified_seqs(self, vendor_id):
        record = self._by_id[vendor_id]
        return {e["verifies_seq"] for e in record.get("changes", [])
                if e["kind"] == "verification" and e.get("verifies_seq") is not None}

    def pending_payment_changes(self, vendor_id, as_of=None):
        """Payment-field changes on record with no matching verification event.

        `as_of` restricts to changes effective on or before a date, so an
        invoice is judged against the master as it stood when the invoice
        arrived rather than as it stands now.
        """
        record = self._by_id[vendor_id]
        verified = self.verified_seqs(vendor_id)
        cutoff = _parse_date(as_of) if as_of else None
        pending = []
        for event in record.get("changes", []):
            if event["kind"] != "change" or event["field"] not in PAYMENT_FIELDS:
                continue
            if event["seq"] in verified:
                continue
            if cutoff is not None:
                effective = _parse_date(event.get("effective_at"))
                if effective is not None and effective > cutoff:
                    continue
            pending.append(event)
        return pending

    def is_new_vendor(self, vendor_id, as_of, window_days=NEW_VENDOR_DAYS):
        record = self._by_id[vendor_id]
        onboarded = _parse_date(record.get("onboarded_date"))
        reference = _parse_date(as_of)
        if onboarded is None or reference is None:
            return False
        age = (reference - onboarded).days
        return 0 <= age <= window_days

    def employee_collision(self, vendor_id, employees):
        """Does this vendor's bank account or contact email match an employee's?

        The pattern this catches: an employee sets up a vendor record --
        themselves, or a shell company -- and routes payments to their own
        account or a personal inbox. Three-way match cannot see this either;
        the PO, GRN and invoice can all agree perfectly, because the fraud is
        in who is being paid, not in what was bought.

        `employees` is a list of employee records with 'bank_account' and
        'email'. Returns a finding detail dict or None. Matches on bank
        account first because it is the harder thing to fake by coincidence;
        an email match is reported too but is weaker evidence (a vendor
        contact could plausibly share a mail domain) and is worth a human
        eyeballing rather than being silently equally severe.
        """
        details = self.current_details(vendor_id)
        vendor_account = details.get("bank_account")
        vendor_email = (details.get("contact_email") or "").strip().lower()

        for emp in employees:
            if vendor_account and emp.get("bank_account") == vendor_account:
                return {
                    "match_field": "bank_account",
                    "employee_id": emp["employee_id"],
                    "employee_name": emp["name"],
                    "vendor_id": vendor_id,
                }
        for emp in employees:
            emp_email = (emp.get("email") or "").strip().lower()
            if vendor_email and emp_email and vendor_email == emp_email:
                return {
                    "match_field": "contact_email",
                    "employee_id": emp["employee_id"],
                    "employee_name": emp["name"],
                    "vendor_id": vendor_id,
                }
        return None

    # --- writing -------------------------------------------------------------

    def _append_event(self, vendor_id, record_fields):
        record = self._by_id[vendor_id]
        events = record.setdefault("changes", [])
        event = chain.append(events, {"vendor_id": vendor_id, **record_fields}, CHANGE_FIELDS)
        events.append(event)
        return event

    def record_change(self, vendor_id, field, new_value, effective_at,
                      actor="vendor_portal", channel="email", note=None):
        """Record that a vendor detail changed. Old value is read from the
        current folded state, not supplied by the caller, so the log cannot
        disagree with itself."""
        old_value = self.current_details(vendor_id).get(field)
        return self._append_event(vendor_id, {
            "kind": "change",
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "effective_at": effective_at,
            "actor": actor,
            "channel": channel,
            "verifies_seq": None,
            "note": note,
        })

    def record_verification(self, vendor_id, verifies_seq, actor, effective_at,
                            channel="phone_callback", note=None):
        """Record that a human confirmed an earlier change out of band.

        `channel` is recorded because it is the whole control: a verification
        by replying to the same email thread the change arrived on verifies
        nothing. The UI offers only channels that leave the thread.
        """
        record = self._by_id[vendor_id]
        target = next((e for e in record.get("changes", [])
                       if e["seq"] == verifies_seq and e["kind"] == "change"), None)
        if target is None:
            raise KeyError(f"no change event seq={verifies_seq} for vendor {vendor_id}")
        return self._append_event(vendor_id, {
            "kind": "verification",
            "field": target["field"],
            "old_value": None,
            "new_value": target["new_value"],
            "effective_at": effective_at,
            "actor": actor,
            "channel": channel,
            "verifies_seq": verifies_seq,
            "note": note,
        })

    def add_vendor(self, vendor_id, canonical_name, bank_account, bank_ifsc,
                   remit_to_name, contact_email, onboarded_date, aliases=None,
                   status="active"):
        record = {
            "vendor_id": vendor_id,
            "canonical_name": canonical_name,
            "aliases": list(aliases or []),
            "bank_account": bank_account,
            "bank_ifsc": bank_ifsc,
            "remit_to_name": remit_to_name,
            "contact_email": contact_email,
            "onboarded_date": onboarded_date,
            "status": status,
            "changes": [],
        }
        self.records.append(record)
        self._by_id[vendor_id] = record
        self._by_name[_norm(canonical_name)] = record
        for alias in record["aliases"]:
            self._by_name.setdefault(_norm(alias), record)
        self._append_event(vendor_id, {
            "kind": "onboarded",
            "field": None,
            "old_value": None,
            "new_value": bank_account,
            "effective_at": onboarded_date,
            "actor": "vendor_onboarding",
            "channel": "onboarding_form",
            "verifies_seq": None,
            "note": f"Vendor onboarded with account {mask_account(bank_account)}.",
        })
        return record

    # --- verification --------------------------------------------------------

    def verify_chain(self):
        """Verify every vendor's change log. Returns {"ok", "vendors", "problems"}
        where each problem carries the vendor it belongs to."""
        problems = []
        for record in self.records:
            report = chain.verify(record.get("changes", []), CHANGE_FIELDS,
                                  label="change event")
            for p in report["problems"]:
                problems.append({**p, "vendor_id": record["vendor_id"],
                                 "canonical_name": record["canonical_name"]})
        return {"ok": not problems, "vendors": len(self.records), "problems": problems}


def mask_account(account):
    """Show the last four digits only. Bank accounts end up in escalation
    packets, which end up on screens; the clerk needs enough to tell two
    accounts apart, not the whole number."""
    if not account:
        return "(none)"
    digits = str(account)
    if len(digits) <= 4:
        return "*" * len(digits)
    return "*" * (len(digits) - 4) + digits[-4:]


def _parse_date(value):
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None
