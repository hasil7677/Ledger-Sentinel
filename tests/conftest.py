"""Shared fixtures and builders for the Ledger Sentinel test suite.

Everything here runs offline by design: LEDGER_SENTINEL_LLM is pinned to
"stub" before any test module is collected, and the cached provider in
agent/llm.py is reset before and after every test so no test can leak a
provider override into the next one, and no stray TENSORMUX_API_KEY in a
developer's real environment can make a test silently start hitting the
network. If that ever happens it is a bug in this file.
"""

import os

os.environ["LEDGER_SENTINEL_LLM"] = "stub"

import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import llm as llm_module
from core.ledger import Ledger
from core.vendor_master import VendorMaster


@pytest.fixture(autouse=True)
def _isolated_llm_provider():
    llm_module.reset_provider()
    yield
    llm_module.reset_provider()


# --- document builders -------------------------------------------------------

def make_line(sku="SKU-1", qty=10, unit_price=100.0, gst_rate=18, description="Widget"):
    return {"sku": sku, "description": description, "qty": qty, "unit_price": unit_price, "gst_rate": gst_rate}


def line_total(qty, unit_price, gst_rate):
    subtotal = qty * unit_price
    return round(subtotal + subtotal * gst_rate / 100 + 1e-9, 2)


def make_po(vendor="Acme Corp", lines=None, po_id="PO-1", po_date="2026-07-01", requester_id="EMP-001"):
    lines = lines if lines is not None else [make_line()]
    return {"id": po_id, "vendor": vendor, "date": po_date, "requester_id": requester_id,
            "lines": [dict(l) for l in lines]}


def make_grn(po, grn_id="GRN-1", received_date="2026-07-05", qty_overrides=None):
    """qty_overrides: {sku: qty_received} to deviate from the PO's own qty."""
    qty_overrides = qty_overrides or {}
    lines = [{"sku": l["sku"], "qty_received": qty_overrides.get(l["sku"], l["qty"])} for l in po["lines"]]
    return {"id": grn_id, "po_id": po["id"], "vendor": po["vendor"],
            "received_date": received_date, "lines": lines}


def make_invoice(po, lines=None, inv_id="INV-1", invoice_date="2026-07-10", vendor=None, requester_id=None):
    lines = lines if lines is not None else [dict(l) for l in po["lines"]]
    total = round(sum(line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in lines), 2)
    return {
        "id": inv_id,
        "po_id": po["id"],
        "vendor": vendor if vendor is not None else po["vendor"],
        "requester_id": requester_id if requester_id is not None else po.get("requester_id", "EMP-001"),
        "invoice_date": invoice_date,
        "lines": lines,
        "total_amount": total,
    }


def make_clean_triple(vendor="Acme Corp", unit_price=100.0, qty=10, gst_rate=18,
                      po_id="PO-1", grn_id="GRN-1", inv_id="INV-1",
                      po_date="2026-07-01", grn_date="2026-07-05", inv_date="2026-07-10",
                      requester_id="EMP-001"):
    lines = [make_line(unit_price=unit_price, qty=qty, gst_rate=gst_rate)]
    po = make_po(vendor=vendor, lines=lines, po_id=po_id, po_date=po_date, requester_id=requester_id)
    grn = make_grn(po, grn_id=grn_id, received_date=grn_date)
    invoice = make_invoice(po, inv_id=inv_id, invoice_date=inv_date, requester_id=requester_id)
    return po, grn, invoice


@pytest.fixture
def fresh_ledger(tmp_path):
    return Ledger(str(tmp_path / "ledger.json"))


@pytest.fixture
def empty_vendor_master():
    return VendorMaster.empty()


@pytest.fixture
def sample_employees():
    return {
        "EMP-001": {"employee_id": "EMP-001", "name": "Meera Rao", "email": "meera.rao@ourcompany.example",
                    "bank_account": "500100010001", "department": "Procurement", "role": "director",
                    "approval_limit": 20_000_000.0},
        "EMP-002": {"employee_id": "EMP-002", "name": "Arjun Iyer", "email": "arjun.iyer@ourcompany.example",
                    "bank_account": "500100010002", "department": "Procurement", "role": "staff",
                    "approval_limit": 75_000.0},
        "EMP-007": {"employee_id": "EMP-007", "name": "Priya Nair", "email": "priya.nair@ourcompany.example",
                    "bank_account": "500100010007", "department": "Finance", "role": "staff",
                    "approval_limit": 75_000.0},
    }


def days_after(iso_date, n):
    return (date.fromisoformat(iso_date) + timedelta(days=n)).isoformat()
