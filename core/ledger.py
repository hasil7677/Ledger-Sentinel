"""Append-only ledger store (JSON-backed).

Simple on purpose: a list of posted entries. Real GL posting is out of
scope; this is the audit trail the demo needs.
"""

import json
import os
from datetime import datetime, date, timedelta


class Ledger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            self._write([])

    def _read(self):
        with open(self.path) as f:
            return json.load(f)

    def _write(self, entries):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)

    def all(self):
        return self._read()

    def post(self, invoice, resolution, posted_by="agent"):
        """Append a posted invoice. resolution: 'auto_clear' or 'approved_by_clerk'."""
        entries = self._read()
        entries.append({
            "invoice_id": invoice["id"],
            "vendor": invoice["vendor"],
            "po_id": invoice.get("po_id"),
            "total_amount": invoice["total_amount"],
            "invoice_date": invoice["invoice_date"],
            "resolution": resolution,
            "posted_by": posted_by,
            "posted_at": datetime.utcnow().isoformat() + "Z",
        })
        self._write(entries)

    def find_near_matches(self, vendor, total_amount, invoice_date, exclude_invoice_id=None, amount_tol=1.0, day_window=30):
        """Find already-posted invoices with the same vendor/amount within a date window.

        Used as a candidate signal for suspected-duplicate detection; the agent
        layer decides whether it's really a duplicate.
        """
        try:
            target_date = date.fromisoformat(invoice_date)
        except ValueError:
            target_date = None

        matches = []
        for entry in self._read():
            if entry["invoice_id"] == exclude_invoice_id:
                continue
            if entry["vendor"] != vendor:
                continue
            if abs(entry["total_amount"] - total_amount) > amount_tol:
                continue
            if target_date is not None:
                try:
                    entry_date = date.fromisoformat(entry["invoice_date"])
                except ValueError:
                    entry_date = None
                if entry_date is not None and abs((entry_date - target_date).days) > day_window:
                    continue
            matches.append(entry)
        return matches
