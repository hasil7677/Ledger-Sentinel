"""Synthetic PO / GRN / Invoice / vendor-master / employee-master dataset
generator for Ledger Sentinel.

Produces a labelled dataset (POs, GRNs, invoices, vendor master, employee
master, and ground-truth labels) with a seeded anomaly taxonomy.

The taxonomy has two halves now. The original eight types (quantity/price/tax/
line-item variance, duplicate, missing PO, vendor name drift) are three-way-
match failures: something on the PO, GRN or invoice disagrees. The five added
here are not -- a vendor bank-account change, a brand-new vendor's first
invoice, a vendor colliding with an employee record, and an approval-limit
violation (single invoice or split across several) can all happen on an
invoice where the PO, GRN and invoice agree on every line. Three-way match
cannot see any of them, which is the point: they model the failure modes that
actually cost money in accounts payable and that this project didn't have a
detector for before this pass.

A fourteenth type, quantity_variance_no_grn (seed_quantity_variance_no_grn
below), is neither: it's a two-way-match case on purpose -- a PO and an
invoice with no goods receipt at all, price and tax matching, only quantity
disagreeing. Closes a gap this generator used to leave entirely untested: a
PO-without-GRN invoice used to be able to bill any quantity and still
auto-clear, because nothing produced this shape of case to exercise the
check against.

Usage:
    python data/generator.py [--count 60] [--anomaly-rate 0.2] [--seed 42] [--out-dir data]
"""

import argparse
import itertools
import json
import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vendor_master import VendorMaster

VENDORS = [
    ("Acme Corp", "Acme Corporation Pvt Ltd"),
    ("Bluewave Traders", "BlueWave Traders Pvt. Ltd."),
    ("Orion Industrial Supplies", "Orion Industrial Supplies Ltd"),
    ("Kestrel Packaging", "KESTREL PACKAGING PVT LTD"),
    ("Nimbus Logistics", "Nimbus Logistics Pvt Ltd"),
    ("Solstice Electricals", "Solstice Electricals Co."),
    ("Vantage Office Supplies", "Vantage Office Supplies Ltd"),
    ("Harbor Steel Works", "Harbor Steel Works Pvt Ltd"),
    ("Zenith Chemicals", "Zenith Chemicals Industries"),
    ("Marigold Stationery", "Marigold Stationery Mart"),
]

PRODUCTS = [
    ("SKU-1001", "A4 Copier Paper (ream)", 285.00, 12),
    ("SKU-1002", "Steel Angle Bracket 50mm", 42.50, 18),
    ("SKU-1003", "Industrial Gloves (pair)", 65.00, 12),
    ("SKU-1004", "LED Panel Light 40W", 1120.00, 18),
    ("SKU-1005", "Corrugated Box (large)", 38.00, 12),
    ("SKU-1006", "Safety Helmet", 310.00, 18),
    ("SKU-1007", "Packing Tape Roll", 55.00, 12),
    ("SKU-1008", "Hydraulic Hose 2m", 890.00, 18),
    ("SKU-1009", "Office Chair (standard)", 4200.00, 18),
    ("SKU-1010", "Cleaning Solvent 5L", 460.00, 12),
]

# Fixed org data, not seeded by rng: an employee master shouldn't reshuffle
# every time someone regenerates the dataset with a different --seed. Only
# which employee gets used where is randomised.
EMPLOYEES = [
    {"employee_id": "EMP-001", "name": "Meera Rao", "email": "meera.rao@ourcompany.example",
     "bank_account": "500100010001", "department": "Procurement", "role": "director",
     "approval_limit": 20_000_000.0},
    {"employee_id": "EMP-002", "name": "Arjun Iyer", "email": "arjun.iyer@ourcompany.example",
     "bank_account": "500100010002", "department": "Procurement", "role": "staff",
     "approval_limit": 75_000.0},
    {"employee_id": "EMP-003", "name": "Divya Shah", "email": "divya.shah@ourcompany.example",
     "bank_account": "500100010003", "department": "Facilities", "role": "staff",
     "approval_limit": 75_000.0},
    {"employee_id": "EMP-004", "name": "Karan Mehta", "email": "karan.mehta@ourcompany.example",
     "bank_account": "500100010004", "department": "Facilities", "role": "staff",
     "approval_limit": 75_000.0},
    {"employee_id": "EMP-005", "name": "Neha Kulkarni", "email": "neha.kulkarni@ourcompany.example",
     "bank_account": "500100010005", "department": "Warehouse", "role": "staff",
     "approval_limit": 75_000.0},
    {"employee_id": "EMP-006", "name": "Rohit Bhatt", "email": "rohit.bhatt@ourcompany.example",
     "bank_account": "500100010006", "department": "IT", "role": "manager",
     "approval_limit": 400_000.0},
    {"employee_id": "EMP-007", "name": "Priya Nair", "email": "priya.nair@ourcompany.example",
     "bank_account": "500100010007", "department": "Finance", "role": "staff",
     "approval_limit": 75_000.0},
]

# EMP-001's limit (Rs 2 crore) is comfortably above the highest total this
# generator can produce (three lines, max qty 200, max unit price Rs 4,200,
# 18% GST -- about Rs 29.7 lakh), so every invoice that doesn't deliberately
# seed an approval-policy anomaly is safe to attribute to them without
# accidentally tripping the approval-limit rule.
DEFAULT_REQUESTER_ID = "EMP-001"
# Dedicated pool for the split-invoice anomaly, one employee per group so two
# groups never share a requester and get mistaken for one larger group.
SPLIT_REQUESTER_POOL = ["EMP-003", "EMP-004", "EMP-005"]
SINGLE_OVER_LIMIT_REQUESTER_ID = "EMP-002"
# Never assigned as a requester -- exists solely to be the bank account /
# email a fraud vendor is made to collide with.
COLLISION_EMPLOYEE_ID = "EMP-007"

SPLIT_GROUP_SIZE = 2

FRAUD_VENDOR_ADJECTIVES = [
    "Falcon", "Silver", "Granite", "Cobalt", "Harvest", "Ironclad", "Meridian",
    "Lantern", "Copper", "Wraith", "Solace", "Pinnacle", "Anchor", "Cascade", "Beacon",
]
FRAUD_VENDOR_NOUNS = [
    "Engineering", "Trading", "Supplies", "Logistics", "Industrial",
    "Traders", "Works", "Enterprises", "Distributors", "Systems",
]

# Relative weights for the anomaly taxonomy. The original eight are the
# proportions from the original 60-invoice plan (2/2/1/2/1/2/1/1 out of 12);
# the five vendor-master / approval-policy types are weighted 1 each -- rarer
# than routine three-way-match noise, which matches how often each actually
# comes up in real AP, but present at any dataset size big enough to seed one.
ANOMALY_WEIGHTS = [
    ("quantity_variance", 2),
    ("price_variance_in_tolerance", 2),
    ("price_variance_out_tolerance", 1),
    ("duplicate", 2),
    ("missing_po", 1),
    ("vendor_name_drift", 2),
    ("tax_mismatch", 1),
    ("line_item_mismatch", 1),
    ("vendor_bank_detail_change", 1),
    ("new_vendor_first_invoice", 1),
    ("vendor_employee_collision", 1),
    ("approval_threshold_exceeded", 1),
    ("threshold_splitting", 1),
    # The two-way-match gap: PO on file, no goods receipt, invoice bills a
    # different quantity than the PO ordered. Weighted 1, same as the other
    # rarer types above -- see seed_quantity_variance_no_grn.
    ("quantity_variance_no_grn", 1),
]

# Types whose plan count is a count of INVOICES generated via the dedicated
# seeding functions below (mirroring duplicate/missing_po, which already work
# this way) rather than a count of in-place mutations of a "slots" triple.
# threshold_splitting is the one exception: its plan count is a count of
# GROUPS, and each group contributes SPLIT_GROUP_SIZE invoices -- see
# generate() for how that's reconciled against --count.
DEDICATED_SINGLE_TYPES = (
    "vendor_bank_detail_change", "new_vendor_first_invoice",
    "vendor_employee_collision", "approval_threshold_exceeded",
    "quantity_variance_no_grn",
)
DEDICATED_TYPES = DEDICATED_SINGLE_TYPES + ("duplicate", "missing_po", "threshold_splitting")


def scale_anomaly_plan(total_anomalies, weights=ANOMALY_WEIGHTS):
    """Distribute total_anomalies across the taxonomy by weight.

    Uses largest-remainder so the parts always sum exactly to total_anomalies,
    and every type keeps at least one instance as long as the budget allows.
    """
    weight_sum = sum(w for _, w in weights)
    exact = [(name, total_anomalies * w / weight_sum) for name, w in weights]
    plan = {name: int(value) for name, value in exact}

    # hand out the leftover by largest fractional remainder
    remaining = total_anomalies - sum(plan.values())
    for name, value in sorted(exact, key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if remaining <= 0:
            break
        plan[name] += 1
        remaining -= 1

    # never let a type vanish entirely while another has spare instances
    if total_anomalies >= len(weights):
        for name, _ in weights:
            if plan[name] == 0:
                donor = max(plan, key=lambda k: plan[k])
                plan[donor] -= 1
                plan[name] += 1

    return [(name, plan[name]) for name, _ in weights]

AUTO_CLEAR_TYPES = {"clean", "price_variance_in_tolerance", "vendor_name_drift"}


def money(x):
    return round(x + 1e-9, 2)


def line_total(qty, unit_price, gst_rate):
    subtotal = qty * unit_price
    return money(subtotal + subtotal * gst_rate / 100)


class IdGen:
    def __init__(self):
        self.po = 2000
        self.grn = 8000
        self.inv = 4000
        self.vendor = 100  # fraud-pool vendors start past the 10 canonical VEND-001..010

    def next_po(self):
        self.po += 1
        return f"PO-{self.po}"

    def next_grn(self):
        self.grn += 1
        return f"GRN-{self.grn}"

    def next_inv(self):
        self.inv += 1
        return f"INV-{self.inv}"

    def next_vendor_id(self):
        self.vendor += 1
        return f"VEND-{self.vendor}"


def make_lines(rng):
    n = rng.randint(1, 3)
    chosen = rng.sample(PRODUCTS, n)
    lines = []
    for sku, desc, price, gst in chosen:
        qty = rng.randint(5, 200)
        lines.append({
            "sku": sku,
            "description": desc,
            "qty": qty,
            "unit_price": price,
            "gst_rate": gst,
        })
    return lines


def controlled_lines(rng, target_total):
    """A single line sized so the post-tax total lands near target_total.

    Used by the approval-policy anomalies, which need to land deliberately on
    one side of a requester's limit rather than wherever a random qty*price
    happens to fall.
    """
    sku, desc, price, gst = rng.choice(PRODUCTS)
    qty = max(1, round(target_total / (price * (1 + gst / 100))))
    return [{"sku": sku, "description": desc, "qty": qty, "unit_price": price, "gst_rate": gst}]


def fraud_vendor_names(rng, n):
    """n distinct business-sounding names, disjoint from VENDORS and from each
    other, for the vendor-master anomalies that must not share state with the
    canonical 10 -- or with EACH OTHER.

    Sharing a vendor between a fraud anomaly and an ordinary triple would let
    the anomaly contaminate later invoices from the same vendor (every
    subsequent invoice from a vendor with an unverified bank change on file
    should, correctly, also be held -- which would then disagree with
    labels.json calling those later invoices clean). Dedicated names sidestep
    the problem entirely instead of trying to schedule around it.

    Call this ONCE per generation run for the whole fraud-vendor budget and
    hand out names from the returned list, rather than calling it once per
    anomaly instance. Two separate calls each reshuffle the full combo space
    independently, so nothing stops them drawing the same name -- which
    happened in practice (seed=42, two "Meridian Trading Pvt Ltd" vendors)
    and corrupted VendorMaster's name index: the second add_vendor() silently
    overwrote the first's entry in _by_name, so every invoice for the first
    vendor started resolving to the second vendor's (clean) state instead of
    its own pending change, producing a false clear no detector logic was
    actually wrong about.
    """
    combos = [f"{a} {b}" for a in FRAUD_VENDOR_ADJECTIVES for b in FRAUD_VENDOR_NOUNS]
    rng.shuffle(combos)
    if n > len(combos):
        combos = combos + [f"{c} Group {i}" for i, c in enumerate(itertools.cycle(combos), start=1)][:n - len(combos)]
    return [f"{name} Pvt Ltd" for name in combos[:n]]


def build_vendor_master(base_date):
    """The 10 canonical vendors, onboarded long before the dataset window so
    none of them can be mistaken for a new vendor, with unremarkable payment
    details and no pending changes. Fraud-pool vendors are added later by the
    anomaly seeding functions that need them."""
    vm = VendorMaster.empty()
    onboarded = (base_date - timedelta(days=500)).isoformat()
    for i, (canonical, alias) in enumerate(VENDORS, start=1):
        vendor_id = f"VEND-{i:03d}"
        domain = "".join(ch for ch in canonical.lower() if ch.isalnum())[:14]
        vm.add_vendor(
            vendor_id=vendor_id,
            canonical_name=canonical,
            bank_account=f"40010{i:07d}",
            bank_ifsc=f"HDFC000{1000 + i}",
            remit_to_name=canonical,
            contact_email=f"accounts@{domain}.example",
            onboarded_date=onboarded,
            aliases=[alias],
        )
    return vm


def build_clean_set(rng, ids, vendor, po_date, lines=None, requester_id=DEFAULT_REQUESTER_ID):
    """Build a matching PO / GRN / invoice triple with no variance."""
    lines = lines if lines is not None else make_lines(rng)
    po_id = ids.next_po()
    grn_id = ids.next_grn()
    inv_id = ids.next_inv()

    grn_date = po_date + timedelta(days=rng.randint(2, 7))
    inv_date = grn_date + timedelta(days=rng.randint(1, 5))

    po = {
        "id": po_id,
        "vendor": vendor,
        "date": po_date.isoformat(),
        "requester_id": requester_id,
        "lines": [dict(l) for l in lines],
    }
    grn = {
        "id": grn_id,
        "po_id": po_id,
        "vendor": vendor,
        "received_date": grn_date.isoformat(),
        "lines": [{"sku": l["sku"], "qty_received": l["qty"]} for l in lines],
    }
    inv_lines = [dict(l) for l in lines]
    total = money(sum(line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in inv_lines))
    invoice = {
        "id": inv_id,
        "po_id": po_id,
        "vendor": vendor,
        "requester_id": requester_id,
        "invoice_date": inv_date.isoformat(),
        "lines": inv_lines,
        "total_amount": total,
    }
    return po, grn, invoice


def apply_anomaly(anomaly_type, po, grn, invoice, rng, ids, vendor_variant):
    """Mutate a clean triple in place to seed the given anomaly. Returns (label_notes, related_id).

    Covers only the original eight three-way-match anomaly types. The five
    vendor-master / approval-policy types are seeded by dedicated functions
    below instead, since they need a different vendor identity or a
    controlled invoice total rather than an in-place field mutation.
    """
    notes = ""
    related_id = None

    if anomaly_type == "quantity_variance":
        # short is drawn from [2, 5], never [1, 5]: the matcher's own tolerance
        # is +-1 unit (core/matcher.py DEFAULT_TOLERANCES["qty_units"]), so a
        # short of exactly 1 is not an anomaly at all -- it's a legitimate
        # in-tolerance short-ship, and labelling it "quantity_variance /
        # escalate" would be a wrong label, not a detector miss. (Found via
        # eval/run_eval.py's per-type recall once it existed to show it: this
        # type was scoring under 100% for a labelling reason, not a bug in
        # core/matcher.py.)
        line = grn["lines"][0]
        short = min(rng.randint(2, 5), line["qty_received"] - 1)
        line["qty_received"] -= short
        notes = f"GRN received qty short by {short} vs invoice/PO qty"

    elif anomaly_type == "price_variance_in_tolerance":
        line = invoice["lines"][0]
        bump = round(rng.uniform(0.005, 0.018), 4)  # up to 1.8%, within 2% tolerance
        line["unit_price"] = money(line["unit_price"] * (1 + bump))
        invoice["total_amount"] = money(sum(
            line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in invoice["lines"]
        ))
        notes = f"Invoice unit price {bump*100:.2f}% over PO price, within tolerance"

    elif anomaly_type == "price_variance_out_tolerance":
        # The matcher clears a line if EITHER the percentage OR the Rs 500
        # total-impact floor is satisfied (core/matcher.py: "tolerance is
        # evaluated on total monetary impact, not per-unit price"). A 10-20%
        # bump comfortably clears the percentage clause, but on a low
        # qty*price line the rupee impact can still land under the floor --
        # a 15% bump on 5 units of a Rs 38 item is under Rs 3. That's not a
        # detector bug, it's the anomaly failing to be an anomaly under the
        # project's own documented rule. Pick the highest qty*price line so
        # the seeded bump reliably clears the floor as well.
        line = max(invoice["lines"], key=lambda l: l["qty"] * l["unit_price"])
        bump = round(rng.uniform(0.10, 0.20), 3)
        line["unit_price"] = money(line["unit_price"] * (1 + bump))
        invoice["total_amount"] = money(sum(
            line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in invoice["lines"]
        ))
        notes = f"Invoice unit price {bump*100:.1f}% over PO price, exceeds tolerance"

    elif anomaly_type == "vendor_name_drift":
        invoice["vendor"] = vendor_variant
        notes = f"Invoice vendor '{vendor_variant}' vs canonical '{po['vendor']}'"

    elif anomaly_type == "tax_mismatch":
        line = invoice["lines"][0]
        correct = line["gst_rate"]
        line["gst_rate"] = 18 if correct != 18 else 12
        invoice["total_amount"] = money(sum(
            line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in invoice["lines"]
        ))
        notes = f"Invoice applied {line['gst_rate']}% GST, correct rate is {correct}%"

    elif anomaly_type == "line_item_mismatch":
        existing_skus = {l["sku"] for l in po["lines"]}
        extra = rng.choice([p for p in PRODUCTS if p[0] not in existing_skus])
        sku, desc, price, gst = extra
        qty = rng.randint(5, 50)
        invoice["lines"].append({
            "sku": sku, "description": desc, "qty": qty,
            "unit_price": price, "gst_rate": gst,
        })
        invoice["total_amount"] = money(sum(
            line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in invoice["lines"]
        ))
        notes = f"Invoice has extra line {sku} not present on PO/GRN"

    return notes, related_id


# --- vendor-master / approval-policy anomaly seeding ------------------------
#
# Each function returns (po, grn, invoice, label) for a single-invoice
# anomaly, or a list of those for threshold_splitting's group. They all
# append to `vendor_master` / consult `employees_by_id` as a side effect where
# relevant, rather than only touching pos/grns/invoices/labels -- the whole
# point of these anomalies is that they live in vendor-master and approval
# state, not in the PO/GRN/invoice documents themselves.

def seed_vendor_bank_detail_change(rng, ids, vendor_master, base_date, day_spread, name):
    vendor_id = ids.next_vendor_id()
    onboarded = (base_date - timedelta(days=500)).isoformat()  # aged, so this isn't ALSO a new-vendor case
    domain = "".join(ch for ch in name.lower() if ch.isalnum())[:14]
    old_account = f"77070{rng.randint(1000000, 9999999)}"
    vendor_master.add_vendor(
        vendor_id=vendor_id, canonical_name=name, bank_account=old_account,
        bank_ifsc="ICIC0001234", remit_to_name=name,
        contact_email=f"accounts@{domain}.example", onboarded_date=onboarded,
    )

    po_date = base_date + timedelta(days=rng.randint(0, day_spread))
    po, grn, invoice = build_clean_set(rng, ids, name, po_date)

    new_account = f"99988{rng.randint(1000000, 9999999)}"
    change_date = (date.fromisoformat(invoice["invoice_date"]) - timedelta(days=rng.randint(1, 4))).isoformat()
    vendor_master.record_change(
        vendor_id, "bank_account", new_account, effective_at=change_date,
        actor="email_request", channel="email",
        note="Bank details update requested by email ahead of this invoice; not yet verified out of band.",
    )

    label = {
        "invoice_id": invoice["id"],
        "anomaly_type": "vendor_bank_detail_change",
        "expected_outcome": "escalate",
        "notes": f"Vendor '{name}' bank account changed by email on {change_date}, unverified, "
                 f"{rng.randint(1, 4)} day(s) before this otherwise-clean invoice.",
    }
    return po, grn, invoice, label


def seed_new_vendor_first_invoice(rng, ids, vendor_master, base_date, day_spread, name):
    vendor_id = ids.next_vendor_id()

    po_date = base_date + timedelta(days=rng.randint(0, day_spread))
    po, grn, invoice = build_clean_set(rng, ids, name, po_date)

    onboarded = (date.fromisoformat(invoice["invoice_date"]) - timedelta(days=rng.randint(5, 45))).isoformat()
    domain = "".join(ch for ch in name.lower() if ch.isalnum())[:14]
    vendor_master.add_vendor(
        vendor_id=vendor_id, canonical_name=name,
        bank_account=f"60060{rng.randint(1000000, 9999999)}",
        bank_ifsc="AXIS0009876", remit_to_name=name,
        contact_email=f"accounts@{domain}.example", onboarded_date=onboarded,
    )

    label = {
        "invoice_id": invoice["id"],
        "anomaly_type": "new_vendor_first_invoice",
        "expected_outcome": "escalate",
        "notes": f"Vendor '{name}' onboarded {onboarded}; this is their first invoice, "
                 f"otherwise a clean three-way match.",
    }
    return po, grn, invoice, label


def seed_vendor_employee_collision(rng, ids, vendor_master, employees_by_id, base_date, day_spread, name):
    vendor_id = ids.next_vendor_id()
    collider = employees_by_id[COLLISION_EMPLOYEE_ID]
    onboarded = (base_date - timedelta(days=500)).isoformat()
    vendor_master.add_vendor(
        vendor_id=vendor_id, canonical_name=name,
        bank_account=collider["bank_account"],  # the collision: paid to an employee's own account
        bank_ifsc="UTIB0005555", remit_to_name=name,
        contact_email=f"accounts@{name.lower().replace(' ', '')[:14]}.example",
        onboarded_date=onboarded,
    )

    po_date = base_date + timedelta(days=rng.randint(0, day_spread))
    po, grn, invoice = build_clean_set(rng, ids, name, po_date)

    label = {
        "invoice_id": invoice["id"],
        "anomaly_type": "vendor_employee_collision",
        "expected_outcome": "escalate",
        "notes": f"Vendor '{name}' bank account matches employee {collider['name']} ({collider['employee_id']}).",
    }
    return po, grn, invoice, label


def seed_approval_threshold_exceeded(rng, ids, employees_by_id, base_date, day_spread):
    employee = employees_by_id[SINGLE_OVER_LIMIT_REQUESTER_ID]
    canonical, _ = rng.choice(VENDORS)
    po_date = base_date + timedelta(days=rng.randint(0, day_spread))
    target = employee["approval_limit"] * rng.uniform(1.3, 1.8)
    lines = controlled_lines(rng, target)
    po, grn, invoice = build_clean_set(rng, ids, canonical, po_date, lines=lines,
                                       requester_id=employee["employee_id"])

    label = {
        "invoice_id": invoice["id"],
        "anomaly_type": "approval_threshold_exceeded",
        "expected_outcome": "escalate",
        "notes": f"Invoice Rs {invoice['total_amount']:,.2f} exceeds requester {employee['name']}'s "
                 f"approval limit of Rs {employee['approval_limit']:,.2f}; otherwise a clean match.",
    }
    return po, grn, invoice, label


def seed_threshold_splitting(rng, ids, employees_by_id, base_date, day_spread, group_index):
    employee = employees_by_id[SPLIT_REQUESTER_POOL[group_index % len(SPLIT_REQUESTER_POOL)]]
    canonical, _ = rng.choice(VENDORS)
    limit = employee["approval_limit"]

    # Reserve a slice of the timeline per group so two groups (even for the
    # same requester, if the pool is exhausted) never fall inside each
    # other's SPLIT_WINDOW_DAYS and get mistaken for one bigger group.
    usable_span = max(day_spread - 30, 30)
    slice_start = (group_index * 25) % usable_span
    po_date_base = base_date + timedelta(days=slice_start)

    per_invoice_target = limit * rng.uniform(0.55, 0.65)  # each member individually under the limit
    members = []
    for k in range(SPLIT_GROUP_SIZE):
        po_date = po_date_base + timedelta(days=k * 3)
        lines = controlled_lines(rng, per_invoice_target)
        members.append(build_clean_set(rng, ids, canonical, po_date, lines=lines,
                                       requester_id=employee["employee_id"]))

    member_ids = [invoice["id"] for _, _, invoice in members]
    labels = []
    for po, grn, invoice in members:
        others = [i for i in member_ids if i != invoice["id"]]
        labels.append({
            "invoice_id": invoice["id"],
            "anomaly_type": "threshold_splitting",
            "expected_outcome": "escalate",
            "notes": f"Requester {employee['name']} split a purchase from {canonical} across "
                     f"{SPLIT_GROUP_SIZE} invoices ({', '.join(member_ids)}), each individually under "
                     f"their Rs {limit:,.2f} limit; the group sums over it.",
            "related_invoice_id": ", ".join(others),
        })
    return members, labels


def seed_quantity_variance_no_grn(rng, ids, base_date, day_spread):
    """PO + invoice only -- deliberately no goods receipt at all, the two-way-
    match gap the README used to call out by name: 'PO + no GRN auto-clears
    if price/tax match, without quantity verification.' Price and GST rate
    match the PO exactly; only the billed quantity disagrees, and there is no
    GRN on file to check it against. core/matcher.py's quantity_variance_no_grn
    fallback (comparing against the PO's own ordered qty) is what's supposed
    to catch this now -- before it existed, this exact case auto-cleared."""
    canonical, _ = rng.choice(VENDORS)
    po_date = base_date + timedelta(days=rng.randint(0, day_spread))
    lines = make_lines(rng)
    po_id = ids.next_po()
    inv_id = ids.next_inv()

    po = {
        "id": po_id,
        "vendor": canonical,
        "date": po_date.isoformat(),
        "requester_id": DEFAULT_REQUESTER_ID,
        "lines": [dict(l) for l in lines],
    }

    inv_date = po_date + timedelta(days=rng.randint(3, 10))
    inv_lines = [dict(l) for l in lines]
    line = inv_lines[0]
    over = rng.randint(2, 10)  # outside the matcher's +-1 unit tolerance
    line["qty"] += over
    total = money(sum(line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in inv_lines))
    invoice = {
        "id": inv_id,
        "po_id": po_id,
        "vendor": canonical,
        "requester_id": DEFAULT_REQUESTER_ID,
        "invoice_date": inv_date.isoformat(),
        "lines": inv_lines,
        "total_amount": total,
    }

    label = {
        "invoice_id": inv_id,
        "anomaly_type": "quantity_variance_no_grn",
        "expected_outcome": "escalate",
        "notes": f"No goods receipt on file for {po_id}; invoice bills {over} more unit(s) of "
                 f"{line['sku']} than the PO ordered ({po['lines'][0]['qty']} -> {line['qty']}). "
                 f"Price and GST rate otherwise match the PO exactly -- the two-way-match gap.",
    }
    # No GRN is returned deliberately -- this anomaly IS the absence of one.
    return po, invoice, label


def generate(count, anomaly_rate, seed, out_dir):
    rng = random.Random(seed)
    ids = IdGen()
    base_date = date(2026, 7, 1)
    vendor_master = build_vendor_master(base_date)
    employees_by_id = {e["employee_id"]: e for e in EMPLOYEES}

    total_anomalies = round(count * anomaly_rate)
    anomaly_plan = scale_anomaly_plan(total_anomalies)
    plan_by_type = dict(anomaly_plan)

    # Spread invoices over a window proportional to the dataset size so invoices
    # per day stays constant as --count grows (60 invoices / 45 days). Holding the
    # window fixed would inflate invoice density and make the ledger's 30-day
    # near-match window fire far more often -- an artifact of the generator
    # rather than a real property of the matching pipeline.
    day_spread = max(45, round(45 * count / 60))

    n_duplicates = plan_by_type.get("duplicate", 0)
    n_missing_po = plan_by_type.get("missing_po", 0)
    n_bank_change = plan_by_type.get("vendor_bank_detail_change", 0)
    n_new_vendor = plan_by_type.get("new_vendor_first_invoice", 0)
    n_collision = plan_by_type.get("vendor_employee_collision", 0)
    n_approval_exceeded = plan_by_type.get("approval_threshold_exceeded", 0)
    n_splitting_groups = plan_by_type.get("threshold_splitting", 0)
    n_qty_no_grn = plan_by_type.get("quantity_variance_no_grn", 0)

    dedicated_invoice_budget = (
        n_duplicates + n_missing_po + n_bank_change + n_new_vendor
        + n_collision + n_approval_exceeded + (n_splitting_groups * SPLIT_GROUP_SIZE)
        + n_qty_no_grn
    )
    n_own_triple = max(0, count - dedicated_invoice_budget)

    anomaly_queue = []
    for kind, n in anomaly_plan:
        anomaly_queue.extend([kind] * n)
    rng.shuffle(anomaly_queue)

    triple_anomalies = [k for k in anomaly_queue if k not in DEDICATED_TYPES]
    slots = ["clean"] * max(0, n_own_triple - len(triple_anomalies)) + triple_anomalies
    rng.shuffle(slots)

    pos, grns, invoices, labels = [], [], [], []
    clean_pool_for_dup = []  # invoice_ids eligible to be duplicated

    for slot in slots:
        po_date = base_date + timedelta(days=rng.randint(0, day_spread))
        canonical, variant = rng.choice(VENDORS)
        po, grn, invoice = build_clean_set(rng, ids, canonical, po_date)

        if slot == "clean":
            label = {
                "invoice_id": invoice["id"],
                "anomaly_type": "clean",
                "expected_outcome": "auto_clear",
                "notes": "Clean match, no variance",
            }
            clean_pool_for_dup.append(invoice["id"])
        else:
            notes, _ = apply_anomaly(slot, po, grn, invoice, rng, ids, variant)
            label = {
                "invoice_id": invoice["id"],
                "anomaly_type": slot,
                "expected_outcome": "auto_clear" if slot in AUTO_CLEAR_TYPES else "escalate",
                "notes": notes,
            }
            if slot in AUTO_CLEAR_TYPES:
                clean_pool_for_dup.append(invoice["id"])

        pos.append(po)
        grns.append(grn)
        invoices.append(invoice)
        labels.append(label)

    # duplicates: reuse an already-generated invoice's PO/GRN, resubmit with a suffixed invoice number
    inv_by_id = {inv["id"]: inv for inv in invoices}
    dup_candidates = rng.sample(clean_pool_for_dup, min(n_duplicates, len(clean_pool_for_dup)))
    for original_id in dup_candidates:
        original = inv_by_id[original_id]
        dup_id = f"{original_id}-A"
        dup_date = date.fromisoformat(original["invoice_date"]) + timedelta(days=rng.randint(1, 10))
        duplicate = {
            "id": dup_id,
            "po_id": original["po_id"],
            "vendor": original["vendor"],
            "requester_id": original.get("requester_id", DEFAULT_REQUESTER_ID),
            "invoice_date": dup_date.isoformat(),
            "lines": [dict(l) for l in original["lines"]],
            "total_amount": original["total_amount"],
        }
        invoices.append(duplicate)
        labels.append({
            "invoice_id": dup_id,
            "anomaly_type": "duplicate",
            "expected_outcome": "escalate",
            "notes": f"Same vendor/amount, invoice number suffix vs already-posted {original_id}",
            "related_invoice_id": original_id,
        })

    # missing_po: invoice with no PO/GRN reference at all
    for _ in range(n_missing_po):
        canonical, _ = rng.choice(VENDORS)
        lines = make_lines(rng)
        inv_date = base_date + timedelta(days=rng.randint(0, day_spread + 10))
        total = money(sum(line_total(l["qty"], l["unit_price"], l["gst_rate"]) for l in lines))
        inv_id = ids.next_inv()
        invoices.append({
            "id": inv_id,
            "po_id": None,
            "vendor": canonical,
            "requester_id": DEFAULT_REQUESTER_ID,
            "invoice_date": inv_date.isoformat(),
            "lines": lines,
            "total_amount": total,
        })
        labels.append({
            "invoice_id": inv_id,
            "anomaly_type": "missing_po",
            "expected_outcome": "escalate",
            "notes": "No matching PO exists for this invoice",
        })

    # vendor-master and approval-policy anomalies. One shared name pool across
    # all three fraud-vendor-consuming types -- see fraud_vendor_names'
    # docstring for why per-call pools are not safe here.
    fraud_names = iter(fraud_vendor_names(rng, n_bank_change + n_new_vendor + n_collision))

    for _ in range(n_bank_change):
        po, grn, invoice, label = seed_vendor_bank_detail_change(
            rng, ids, vendor_master, base_date, day_spread, next(fraud_names)
        )
        pos.append(po); grns.append(grn); invoices.append(invoice); labels.append(label)

    for _ in range(n_new_vendor):
        po, grn, invoice, label = seed_new_vendor_first_invoice(
            rng, ids, vendor_master, base_date, day_spread, next(fraud_names)
        )
        pos.append(po); grns.append(grn); invoices.append(invoice); labels.append(label)

    for _ in range(n_collision):
        po, grn, invoice, label = seed_vendor_employee_collision(
            rng, ids, vendor_master, employees_by_id, base_date, day_spread, next(fraud_names)
        )
        pos.append(po); grns.append(grn); invoices.append(invoice); labels.append(label)

    for _ in range(n_approval_exceeded):
        po, grn, invoice, label = seed_approval_threshold_exceeded(rng, ids, employees_by_id, base_date, day_spread)
        pos.append(po); grns.append(grn); invoices.append(invoice); labels.append(label)

    for group_index in range(n_splitting_groups):
        members, group_labels = seed_threshold_splitting(
            rng, ids, employees_by_id, base_date, day_spread, group_index
        )
        for po, grn, invoice in members:
            pos.append(po); grns.append(grn); invoices.append(invoice)
        labels.extend(group_labels)

    for _ in range(n_qty_no_grn):
        po, invoice, label = seed_quantity_variance_no_grn(rng, ids, base_date, day_spread)
        pos.append(po); invoices.append(invoice); labels.append(label)  # deliberately no GRN appended

    invoices.sort(key=lambda i: i["invoice_date"])
    labels_by_id = {l["invoice_id"]: l for l in labels}
    labels = [labels_by_id[inv["id"]] for inv in invoices]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pos.json"), "w") as f:
        json.dump(pos, f, indent=2)
    with open(os.path.join(out_dir, "grns.json"), "w") as f:
        json.dump(grns, f, indent=2)
    with open(os.path.join(out_dir, "invoices.json"), "w") as f:
        json.dump(invoices, f, indent=2)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    vendor_master.save(os.path.join(out_dir, "vendors.json"))
    with open(os.path.join(out_dir, "employees.json"), "w") as f:
        json.dump(EMPLOYEES, f, indent=2)

    return pos, grns, invoices, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--anomaly-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    pos, grns, invoices, labels = generate(args.count, args.anomaly_rate, args.seed, args.out_dir)

    by_type = {}
    for l in labels:
        by_type[l["anomaly_type"]] = by_type.get(l["anomaly_type"], 0) + 1

    print(f"Generated {len(pos)} POs, {len(grns)} GRNs, {len(invoices)} invoices, "
          f"{len(VENDORS)} core vendors + fraud-pool vendors, {len(EMPLOYEES)} employees -> {args.out_dir}")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
