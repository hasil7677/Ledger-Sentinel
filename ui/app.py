"""Exception review queue. Run with: streamlit run ui/app.py

Dense, single-page, one-click approve/reject -- meant to look like a tool
an AP clerk uses 40 times a day, not a demo.
"""

import json
import os
import sys
from datetime import date, datetime

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ledger import Ledger

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
EXCEPTIONS_PATH = os.path.join(RUNTIME_DIR, "exceptions.json")
LEDGER_PATH = os.path.join(RUNTIME_DIR, "ledger.json")
REJECTIONS_PATH = os.path.join(RUNTIME_DIR, "rejections.json")

# Status palette (high/medium/low -> critical/serious/warning), plus the app's
# one accent color -- both from the validated dark-mode reference palette.
ACCENT = "#3987e5"
SEVERITY_COLOR = {"high": "#d03b3b", "medium": "#ec835a", "low": "#fab219"}
MUTED = "#898781"
SURFACE = "#1a1a19"
BORDER = "#2c2c2a"
MISMATCH_COLOR = SEVERITY_COLOR["high"]

# (label, exception_type slug). None = no filter ("All"). Client-side filter only --
# reads the same st.session_state.queue already loaded, no new data/fetch.
CATEGORIES = [
    ("All", None),
    ("Quantity Variance", "quantity_variance"),
    ("Suspected Duplicate", "suspected_duplicate"),
    ("Line Item Mismatch", "line_item_mismatch"),
]

st.set_page_config(page_title="Ledger Sentinel - Exception Queue", layout="wide")

CUSTOM_CSS = f"""
<style>
.stApp {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

/* every bordered st.container becomes a card: queue rows, PO/GRN/Invoice panels */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    background: {SURFACE} !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.45);
    transition: border-color 150ms ease, box-shadow 150ms ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.ls-doc-card) {{
    padding: 4px 2px;
    border-top: 2px solid {ACCENT} !important;
}}

/* queue card hover: accent border + pointer cursor */
[class*="st-key-card_"]:hover [data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: {ACCENT} !important;
    cursor: pointer;
}}

/* selected queue card: left accent stripe (key gets a "_selected" suffix in app.py) */
[class*="st-key-card_"][class*="_selected"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-left: 4px solid {ACCENT} !important;
}}

.ls-badge {{
    display: inline-block;
    padding: 3px 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 800;
    line-height: 1.4;
    letter-spacing: .02em;
    text-transform: uppercase;
    margin-right: 6px;
    white-space: nowrap;
}}

/* breathing room before the reasoning-trace expander */
[data-testid="stExpander"] {{
    margin-top: 20px;
}}

.ls-sku {{ color: {MUTED}; font-size: 0.78rem; }}
.ls-line {{ padding: 3px 0; border-bottom: 1px dashed {BORDER}; font-size: 0.86rem; }}
.ls-line:last-child {{ border-bottom: none; }}
.ls-doc-title {{ font-weight: 700; font-size: 0.95rem; margin-bottom: 1px; }}
.ls-doc-subtitle {{ color: {MUTED}; font-size: 0.75rem; margin-bottom: 10px; }}

.ls-step {{
    display: flex;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid {BORDER};
    font-size: 0.88rem;
    line-height: 1.4;
}}
.ls-step:last-child {{ border-bottom: none; }}
.ls-step-n {{
    flex: 0 0 auto;
    color: {MUTED};
    font-variant-numeric: tabular-nums;
    min-width: 1.4em;
}}
.ls-step-src {{
    display: inline-block;
    margin-left: 6px;
    font-size: 0.72rem;
    padding: 1px 7px;
    border-radius: 999px;
    white-space: nowrap;
}}

/* Approve / Reject: full-width, high-contrast, scoped to their own container keys */
div[class*="st-key-approve_wrap"] button {{
    background: #0ca30c !important;
    border-color: #0ca30c !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    padding-top: 0.6rem !important;
    padding-bottom: 0.6rem !important;
}}
div[class*="st-key-approve_wrap"] button:hover {{ background: #0a8c0a !important; }}
div[class*="st-key-reject_wrap"] button {{
    background: transparent !important;
    border: 1.5px solid {SEVERITY_COLOR['high']} !important;
    color: {SEVERITY_COLOR['high']} !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    padding-top: 0.6rem !important;
    padding-bottom: 0.6rem !important;
}}
div[class*="st-key-reject_wrap"] button:hover {{
    background: {SEVERITY_COLOR['high']} !important;
    color: #ffffff !important;
}}

/* Queue row select buttons: blend into their card rather than look like a button */
div[class*="st-key-row_"] button {{
    text-align: left !important;
    justify-content: flex-start !important;
}}

/* Category filter tabs: flat underline strip, not button chrome */
div[class*="st-key-tabbar_"] {{
    margin-bottom: 4px;
}}
div[class*="st-key-tabbar_"] button {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2.5px solid transparent !important;
    border-radius: 0 !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    padding: 6px 4px 10px 4px !important;
    box-shadow: none !important;
    transition: color 150ms ease, border-color 150ms ease !important;
}}
div[class*="st-key-tabbar_"] button:hover {{
    color: #e6e5e0 !important;
    border-bottom-color: {BORDER} !important;
}}
div[class*="st-key-tabbar_"][class*="_active"] button {{
    color: {ACCENT} !important;
    border-bottom-color: {ACCENT} !important;
    font-weight: 800 !important;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_exceptions():
    if not os.path.exists(EXCEPTIONS_PATH):
        return []
    with open(EXCEPTIONS_PATH) as f:
        return json.load(f)


def load_rejections():
    if not os.path.exists(REJECTIONS_PATH):
        return []
    with open(REJECTIONS_PATH) as f:
        return json.load(f)


def save_rejection(packet, reason):
    rejections = load_rejections()
    rejections.append({
        "invoice_id": packet["invoice_id"],
        "vendor": packet["vendor"],
        "amount": packet["amount"],
        "exception_type": packet["exception_type"],
        "reason": reason,
        "rejected_at": datetime.utcnow().isoformat() + "Z",
    })
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(REJECTIONS_PATH, "w") as f:
        json.dump(rejections, f, indent=2)


def packet_to_invoice_stub(packet):
    return {
        "id": packet["invoice_id"],
        "vendor": packet["vendor"],
        "po_id": packet.get("po_id"),
        "total_amount": packet["amount"],
        "invoice_date": packet["invoice_date"],
    }


def severity_badge(severity):
    color = SEVERITY_COLOR.get(severity, MUTED)
    return (f"<span class='ls-badge' style='background:{color}26;color:{color};"
            f"border:1px solid {color}66'>{severity.upper()}</span>")


def type_badge(exception_type):
    label = exception_type.replace("_", " ").title()
    return f"<span class='ls-badge' style='background:{BORDER};color:#e6e5e0'>{label}</span>"


def field_span(value, is_mismatch):
    if is_mismatch:
        return f"<b style='color:{MISMATCH_COLOR}'>{value}</b>"
    return str(value)


if "queue" not in st.session_state:
    all_exceptions = load_exceptions()
    resolved_ids = {r["invoice_id"] for r in load_rejections()}
    ledger_ids = {e["invoice_id"] for e in Ledger(LEDGER_PATH).all()}
    st.session_state.queue = [
        e for e in all_exceptions if e["invoice_id"] not in resolved_ids and e["invoice_id"] not in ledger_ids
    ]
    st.session_state.selected = st.session_state.queue[0]["invoice_id"] if st.session_state.queue else None
if "active_category" not in st.session_state:
    st.session_state.active_category = None  # None = "All"

queue = st.session_state.queue
reference_date = max((date.fromisoformat(e["invoice_date"]) for e in queue), default=date.today())

st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
    f"padding-bottom:12px;margin-bottom:20px;border-bottom:2px solid {ACCENT}'>"
    f"<span style='font-size:1.6rem;font-weight:700'>Ledger Sentinel — Exception Queue</span>"
    f"<span style='color:{MUTED};font-size:0.95rem'>{len(queue)} open exceptions</span>"
    f"</div>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2], gap="large")

with left:
    tab_cols = st.columns(len(CATEGORIES), gap="small")
    for (label, slug), tab_col in zip(CATEGORIES, tab_cols):
        count = sum(1 for p in queue if slug is None or p["exception_type"] == slug)
        is_active = st.session_state.active_category == slug
        tab_key = f"tabbar_{slug or 'all'}" + ("_active" if is_active else "")
        with tab_col:
            with st.container(key=tab_key):
                if st.button(f"{label} ({count})", key=f"tabbtn_{slug or 'all'}", use_container_width=True):
                    st.session_state.active_category = slug
                    st.rerun()
    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin-bottom:16px'></div>", unsafe_allow_html=True)

    filtered_queue = [
        p for p in queue
        if st.session_state.active_category is None or p["exception_type"] == st.session_state.active_category
    ]
    if not filtered_queue:
        st.caption("No exceptions in this category.")

    for packet in sorted(filtered_queue, key=lambda p: -p["confidence"]):
        aging = (reference_date - date.fromisoformat(packet["invoice_date"])).days
        is_selected = st.session_state.selected == packet["invoice_id"]
        card_key = f"card_{packet['invoice_id']}" + ("_selected" if is_selected else "")
        with st.container(key=card_key, border=True):
            head_l, head_r = st.columns([3, 2])
            with head_l:
                st.markdown(
                    f"<div style='font-weight:700'>{packet['vendor']}</div>"
                    f"<div style='color:{MUTED};font-size:0.78rem'>{packet['invoice_id']}</div>",
                    unsafe_allow_html=True,
                )
            with head_r:
                st.markdown(
                    f"<div style='text-align:right;font-weight:700'>₹{packet['amount']:,.0f}</div>"
                    f"<div style='text-align:right;color:{MUTED};font-size:0.78rem'>{aging}d aging</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='margin:8px 0 10px 0'>{severity_badge(packet['severity'])}"
                f"{type_badge(packet['exception_type'])}</div>",
                unsafe_allow_html=True,
            )
            if st.button("Review" if not is_selected else "Reviewing", key=f"row_{packet['invoice_id']}",
                         use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected = packet["invoice_id"]
                st.rerun()

with right:
    packet = next((p for p in queue if p["invoice_id"] == st.session_state.selected), None)

    if packet is None:
        st.info("Queue is empty. Nothing waiting on human review.")
    else:
        st.markdown(
            f"### {packet['invoice_id']} "
            f"<span style='margin-left:10px'>{severity_badge(packet['severity'])}"
            f"{type_badge(packet['exception_type'])}</span>",
            unsafe_allow_html=True,
        )

        st.markdown(f"**{packet['summary']}**")
        st.markdown(
            f"Proposed resolution: **{packet['proposed_resolution']}**  ·  "
            f"confidence <span style='color:{ACCENT};font-weight:700'>{packet['confidence']:.0%}</span>",
            unsafe_allow_html=True,
        )
        if packet.get("vendor_normalised_from"):
            st.caption(f"Vendor normalised from '{packet['vendor_normalised_from']}' -> '{packet['vendor']}'")

        steps = packet.get("steps") or []
        if steps:
            with st.expander(f"How the agent got here — {len(steps)} checks", expanded=True):
                for step in steps:
                    src = step.get("source", "deterministic")
                    if src.startswith("llm"):
                        tag_bg, tag_fg = f"{ACCENT}26", ACCENT
                    elif src == "heuristic-fallback":
                        tag_bg, tag_fg = f"{SEVERITY_COLOR['low']}26", SEVERITY_COLOR["low"]
                    else:
                        tag_bg, tag_fg = BORDER, "#c3c2b7"
                    st.markdown(
                        f"<div class='ls-step'>"
                        f"<span class='ls-step-n'>{step['n']}.</span>"
                        f"<span>{step['detail']}"
                        f"<span class='ls-step-src' style='background:{tag_bg};color:{tag_fg}'>{src}</span>"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )

        bad_skus = {f["detail"].get("sku") for f in packet["findings_detail"] if f["detail"].get("sku")}

        po = packet["evidence"]["po"]
        grn = packet["evidence"]["grn"]
        inv_lines = packet["evidence"]["invoice"]["qty_lines"]

        st.markdown("#### Three-way comparison")
        if po is None:
            st.warning("No matching PO found for this invoice.")
        else:
            po_by_sku = {l["sku"]: l for l in po["lines"]}
            grn_by_sku = {l["sku"]: l for l in (grn["lines"] if grn else [])}

            col_po, col_grn, col_inv = st.columns(3)

            with col_po:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='ls-doc-card'>"
                        f"<div class='ls-doc-title'>Purchase Order</div>"
                        f"<div class='ls-doc-subtitle'>{po['id']}</div></div>",
                        unsafe_allow_html=True,
                    )
                    rows = []
                    for line in inv_lines:
                        sku = line["sku"]
                        po_l = po_by_sku.get(sku)
                        if po_l is None:
                            rows.append(f"<div class='ls-line'><span class='ls-sku'>{sku}</span><br>"
                                        f"{field_span('not on PO', True)}</div>")
                            continue
                        price_bad = po_l["unit_price"] != line["unit_price"]
                        gst_bad = po_l["gst_rate"] != line["gst_rate"]
                        price_str = f"{po_l['unit_price']:.2f}"
                        gst_str = f"{po_l['gst_rate']}%"
                        rows.append(
                            f"<div class='ls-line'><span class='ls-sku'>{sku}</span><br>"
                            f"qty {po_l['qty']} · ₹{field_span(price_str, price_bad)} · "
                            f"GST {field_span(gst_str, gst_bad)}</div>"
                        )
                    st.markdown("".join(rows), unsafe_allow_html=True)

            with col_grn:
                with st.container(border=True):
                    if grn is not None:
                        st.markdown(
                            f"<div class='ls-doc-card'>"
                            f"<div class='ls-doc-title'>Goods Receipt</div>"
                            f"<div class='ls-doc-subtitle'>{grn['id']} · received {grn['received_date']}</div></div>",
                            unsafe_allow_html=True,
                        )
                        rows = []
                        for line in inv_lines:
                            sku = line["sku"]
                            grn_l = grn_by_sku.get(sku)
                            if grn_l is None:
                                rows.append(f"<div class='ls-line'><span class='ls-sku'>{sku}</span><br>"
                                            f"{field_span('not received', True)}</div>")
                                continue
                            qty_bad = grn_l["qty_received"] != line["qty"]
                            rows.append(
                                f"<div class='ls-line'><span class='ls-sku'>{sku}</span><br>"
                                f"received {field_span(grn_l['qty_received'], qty_bad)}</div>"
                            )
                        st.markdown("".join(rows), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<div class='ls-doc-card'><div class='ls-doc-title'>Goods Receipt</div>"
                            f"<div class='ls-doc-subtitle'>{field_span('no GRN on file', True)}</div></div>",
                            unsafe_allow_html=True,
                        )

            with col_inv:
                with st.container(border=True):
                    st.markdown(
                        f"<div class='ls-doc-card'>"
                        f"<div class='ls-doc-title'>Invoice</div>"
                        f"<div class='ls-doc-subtitle'>{packet['invoice_id']}</div></div>",
                        unsafe_allow_html=True,
                    )
                    rows = []
                    for line in inv_lines:
                        sku = line["sku"]
                        is_bad = sku in bad_skus
                        po_l = po_by_sku.get(sku)
                        grn_l = grn_by_sku.get(sku)
                        qty_bad = is_bad and bool(grn_l and grn_l["qty_received"] != line["qty"])
                        price_bad = is_bad and bool(po_l and po_l["unit_price"] != line["unit_price"])
                        gst_bad = is_bad and bool(po_l and po_l["gst_rate"] != line["gst_rate"])
                        extra_line = is_bad and po_l is None
                        price_str = f"{line['unit_price']:.2f}"
                        gst_str = f"{line['gst_rate']}%"
                        rows.append(
                            f"<div class='ls-line'><span class='ls-sku'>{field_span(sku, extra_line)}</span><br>"
                            f"qty {field_span(line['qty'], qty_bad)} · "
                            f"₹{field_span(price_str, price_bad)} · "
                            f"GST {field_span(gst_str, gst_bad)}</div>"
                        )
                    st.markdown("".join(rows), unsafe_allow_html=True)

        if packet["evidence"]["related"]:
            st.markdown("#### Related invoice(s)")
            st.json(packet["evidence"]["related"])

        st.markdown("&nbsp;")
        approve_col, reject_col = st.columns(2)
        with approve_col:
            with st.container(key="approve_wrap"):
                approve_clicked = st.button("✓ Approve", use_container_width=True, key="approve")
        with reject_col:
            with st.container(key="reject_wrap"):
                reject_clicked = st.button("✕ Reject", use_container_width=True, key="reject")

        if approve_clicked:
            ledger = Ledger(LEDGER_PATH)
            ledger.post(packet_to_invoice_stub(packet), resolution="approved_by_clerk")
            st.session_state.queue = [p for p in queue if p["invoice_id"] != packet["invoice_id"]]
            st.session_state.selected = st.session_state.queue[0]["invoice_id"] if st.session_state.queue else None
            st.rerun()
        if reject_clicked:
            save_rejection(packet, packet["proposed_resolution"])
            st.session_state.queue = [p for p in queue if p["invoice_id"] != packet["invoice_id"]]
            st.session_state.selected = st.session_state.queue[0]["invoice_id"] if st.session_state.queue else None
            st.rerun()
