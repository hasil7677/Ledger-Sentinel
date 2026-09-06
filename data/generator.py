"""Synthetic PO / GRN / Invoice dataset generator for Ledger Sentinel.

Produces a labelled dataset (POs, GRNs, invoices, and ground-truth labels)
with a seeded anomaly taxonomy, matching the spec in STARTER.md section 6.

Usage:
    python data/generator.py [--count 60] [--anomaly-rate 0.2] [--seed 42] [--out-dir data]
"""

import argparse
import json
import os
import random
from datetime import date, timedelta

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

# Relative weights for the anomaly taxonomy. These are the proportions from the
# original 60-invoice plan (2/2/1/2/1/2/1/1 out of 12); they are scaled to
# whatever anomaly budget the requested --count/--anomaly-rate implies, so the
# taxonomy stays proportionally represented at any dataset size.
ANOMALY_WEIGHTS = [
    ("quantity_variance", 2),
    ("price_variance_in_tolerance", 2),
    ("price_variance_out_tolerance", 1),
    ("duplicate", 2),
    ("missing_po", 1),
    ("vendor_name_drift", 2),
    ("tax_mismatch", 1),
    ("line_item_mismatch", 1),
]


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

    def next_po(self):
        self.po += 1
        return f"PO-{self.po}"

    def next_grn(self):
        self.grn += 1
        return f"GRN-{self.grn}"

    def next_inv(self):
        self.inv += 1
        return f"INV-{self.inv}"


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


def build_clean_set(rng, ids, vendor, po_date):
    """Build a matching PO / GRN / invoice triple with no variance."""
    lines = make_lines(rng)
    po_id = ids.next_po()
    grn_id = ids.next_grn()
    inv_id = ids.next_inv()

    grn_date = po_date + timedelta(days=rng.randint(2, 7))
    inv_date = grn_date + timedelta(days=rng.randint(1, 5))

    po = {
        "id": po_id,
        "vendor": vendor,
        "date": po_date.isoformat(),
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
        "invoice_date": inv_date.isoformat(),
        "lines": inv_lines,
        "total_amount": total,
    }
    return po, grn, invoice


def apply_anomaly(anomaly_type, po, grn, invoice, rng, ids, vendor_variant):
    """Mutate a clean triple in place to seed the given anomaly. Returns (label_notes, related_id)."""
    notes = ""
    related_id = None

    if anomaly_type == "quantity_variance":
        line = grn["lines"][0]
        short = min(rng.randint(1, 5), line["qty_received"] - 1)
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
        line = invoice["lines"][0]
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


def generate(count, anomaly_rate, seed, out_dir):
    rng = random.Random(seed)
    ids = IdGen()

    total_anomalies = round(count * anomaly_rate)
    anomaly_plan = scale_anomaly_plan(total_anomalies)

    pos, grns, invoices, labels = [], [], [], []
    base_date = date(2026, 7, 1)

    # Spread invoices over a window proportional to the dataset size so invoices
    # per day stays constant as --count grows (60 invoices / 45 days). Holding the
    # window fixed would inflate invoice density and make the ledger's 30-day
    # near-match window fire far more often -- an artifact of the generator
    # rather than a real property of the matching pipeline.
    day_spread = max(45, round(45 * count / 60))

    anomaly_queue = []
    for kind, n in anomaly_plan:
        anomaly_queue.extend([kind] * n)
    rng.shuffle(anomaly_queue)

    n_duplicates = sum(1 for k in anomaly_queue if k == "duplicate")
    n_missing_po = sum(1 for k in anomaly_queue if k == "missing_po")
    n_own_triple = count - n_duplicates - n_missing_po  # clean + non-duplicate/non-missing anomalies

    clean_pool_for_dup = []  # invoice_ids eligible to be duplicated

    triple_anomalies = [k for k in anomaly_queue if k not in ("duplicate", "missing_po")]
    slots = ["clean"] * (n_own_triple - len(triple_anomalies)) + triple_anomalies
    rng.shuffle(slots)

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

    print(f"Generated {len(pos)} POs, {len(grns)} GRNs, {len(invoices)} invoices -> {args.out_dir}")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
