"""Append-only, hash-chained ledger store (JSON-backed).

The whole product of this system is evidence a human can act on. That evidence
was sitting in a mutable JSON array: anyone could open `runtime/ledger.json`,
change a posted amount or delete a row, and nothing anywhere would notice --
including the duplicate detector, whose entire job is to find an invoice that
is already in that file. Deleting the original posting is a one-line way to
make a duplicate invoice pay twice, and it would have left no mark.

So each entry now carries the hash of the entry before it (see `core/chain.py`
for the mechanism and for the honest limit of what a chain proves). Verify one
from the command line:

    python -m core.ledger verify --path runtime/ledger.json

Exit status is 0 when intact, 1 when the chain is broken. `unchained` entries
-- rows written before this existed -- are reported but do not fail the check,
because absence of a hash is not evidence of tampering.
"""

import argparse
import json
import os
from datetime import datetime, date

from core import chain
from core.chain import GENESIS_HASH

# Fields covered by the hash, in no particular order (the canonical form sorts
# them). Anything not in this list is NOT protected, so nothing
# decision-relevant may live outside it.
CHAINED_FIELDS = (
    "seq",
    "invoice_id",
    "vendor",
    "po_id",
    "requester_id",
    "total_amount",
    "invoice_date",
    "resolution",
    "posted_by",
    "posted_at",
    "prev_hash",
)


def _utcnow():
    return datetime.utcnow().isoformat() + "Z"


def compute_hash(entry):
    return chain.compute_hash(entry, CHAINED_FIELDS)


class Ledger:
    def __init__(self, path, fresh=False, clock=_utcnow):
        """`fresh=True` truncates an existing file.

        This is not a convenience. `eval/run_eval.py` appended into whatever
        ledger was already at its output path, so a second run saw the first
        run's postings and reported suspected duplicates the dataset never
        contained -- the eval was not reproducible against its own output
        directory. Any batch run that wants a clean starting state now has to
        say so.
        """
        self.path = path
        self._clock = clock
        if fresh or not os.path.exists(self.path):
            self._write([])

    # --- storage -------------------------------------------------------------

    def _read(self):
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries):
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)

    def all(self):
        return self._read()

    def head(self):
        """Hash of the last entry -- the value to record elsewhere if you want
        an anchor for the chain outside this file."""
        return chain.head(self._read())

    # --- writing -------------------------------------------------------------

    def post(self, invoice, resolution, posted_by="agent", approved_by=None):
        """Append a posted invoice, chained to the current head."""
        entries = self._read()
        entry = chain.append(entries, {
            "invoice_id": invoice["id"],
            "vendor": invoice["vendor"],
            "po_id": invoice.get("po_id"),
            "requester_id": invoice.get("requester_id"),
            "total_amount": invoice["total_amount"],
            "invoice_date": invoice["invoice_date"],
            "resolution": resolution,
            "posted_by": posted_by,
            "posted_at": self._clock(),
        }, CHAINED_FIELDS)
        if approved_by is not None:
            # Deliberately outside the hash: approver identity is attached by the
            # review UI after the agent's decision, and the chain attests to what
            # the agent posted. If approver identity ever becomes load-bearing for
            # a control, it has to move into CHAINED_FIELDS -- and doing so
            # invalidates every existing hash, which is the correct cost.
            entry["approved_by"] = approved_by
        entries.append(entry)
        self._write(entries)
        return entry

    # --- verification --------------------------------------------------------

    def verify(self):
        return verify_entries(self._read())

    # --- queries -------------------------------------------------------------

    def find_near_matches(self, vendor, total_amount, invoice_date,
                          exclude_invoice_id=None, amount_tol=1.0, day_window=30):
        """Already-posted invoices with the same vendor and a very close amount
        inside a date window. A candidate signal for suspected-duplicate
        detection; the agent layer decides how to describe it, never whether to
        act on it."""
        target_date = parse_date(invoice_date)

        matches = []
        for entry in self._read():
            if entry["invoice_id"] == exclude_invoice_id:
                continue
            if entry["vendor"] != vendor:
                continue
            if abs(entry["total_amount"] - total_amount) > amount_tol:
                continue
            if target_date is not None:
                entry_date = parse_date(entry["invoice_date"])
                if entry_date is not None and abs((entry_date - target_date).days) > day_window:
                    continue
            matches.append(entry)
        return matches

    def entries_for_vendor(self, vendor, before_date=None, day_window=None):
        """Posted entries for one vendor, optionally restricted to a backward
        window ending at before_date. Split-invoice detection needs a vendor's
        recent posting history rather than a single amount match."""
        target = parse_date(before_date) if before_date else None
        out = []
        for entry in self._read():
            if entry["vendor"] != vendor:
                continue
            if target is not None and day_window is not None:
                entry_date = parse_date(entry["invoice_date"])
                if entry_date is None:
                    continue
                delta = (target - entry_date).days
                if delta < 0 or delta > day_window:
                    continue
            out.append(entry)
        return out


def verify_entries(entries):
    return chain.verify(entries, CHAINED_FIELDS, label="entry")


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ledger integrity tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    verify_cmd = sub.add_parser("verify", help="Walk the hash chain and report any break.")
    verify_cmd.add_argument("--path", default=os.path.join("runtime", "ledger.json"))
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"No ledger at {args.path}")
        return 2

    with open(args.path) as f:
        report = verify_entries(json.load(f))

    print(f"ledger:  {args.path}")
    print(f"entries: {report['entries']}")
    print(f"head:    {report['head']}")
    if report["ok"]:
        print("status:  INTACT -- every entry hashes to its recorded value "
              "and links to the one before it.")
        return 0

    print(f"status:  {len(report['problems'])} problem(s)")
    for p in report["problems"]:
        print(f"  seq {p['seq']}: {p['kind']} -- {p['detail']}")
    fatal = [p for p in report["problems"] if p["kind"] != "unchained"]
    if fatal:
        return 1
    print("status:  no tampering detected; the entries above simply predate the chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
