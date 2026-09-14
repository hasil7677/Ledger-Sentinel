"""Decision trace: an ordered record of what the agent actually checked.

Steps are appended by the orchestrator at the moment each check runs, from
the real intermediate values that check produced. Nothing here reconstructs
a plausible-looking narrative after the decision is made -- if a step is in
the trace, that check genuinely executed, in that order.

Each step carries a `source` so a reviewer can see which conclusions came
from deterministic arithmetic and which came from model judgement.
"""

DETERMINISTIC = "deterministic"


def llm_source(judgement):
    """Label a step by the path that actually produced it.

    The judgement result carries its own `source`, set in agent/tools.py from
    whichever provider answered, so this reports rather than infers. The
    `llm_used` branch below only covers judgement dicts assembled by hand.
    """
    source = judgement.get("source")
    if source:
        return source
    model = judgement.get("model")
    if judgement.get("llm_used"):
        return f"llm:{model}" if model else "llm"
    return "heuristic-fallback"


class Trace:
    def __init__(self):
        self._steps = []

    def add(self, action, detail, source=DETERMINISTIC):
        self._steps.append({
            "n": len(self._steps) + 1,
            "action": action,
            "detail": detail,
            "source": source,
        })

    def steps(self):
        return list(self._steps)


def _rupees(value):
    return f"Rs {value:,.2f}"


def describe_variance(v):
    """Render one matcher variance using only the fields the matcher produced.

    Deliberately does not recompute anything (notably not the tolerance
    decision) -- it reports what core.matcher already decided.
    """
    sku = v.get("sku")
    kind = v["type"]

    if kind == "price_variance":
        return (
            f"Price variance on {sku}: PO {_rupees(v['po_value'])} vs invoice "
            f"{_rupees(v['invoice_value'])} (delta {v['delta_pct']:.2f}%)"
        )

    if kind == "quantity_variance":
        return (
            f"Quantity variance on {sku}: PO ordered {v['po_value']}, GRN received "
            f"{v['grn_value']}, invoice billed {v['invoice_value']} (billed {v['delta']:+d} vs received)"
        )

    if kind == "tax_mismatch":
        return (
            f"Tax mismatch on {sku}: PO rate {v['po_value']}% vs invoice rate "
            f"{v['invoice_value']}%"
        )

    if kind == "line_item_mismatch":
        billed = v.get("invoice_value") or {}
        return (
            f"Line {sku} appears on the invoice ({billed.get('qty')} @ "
            f"{_rupees(billed.get('unit_price', 0))}) but is not on the PO"
        )

    return f"{kind.replace('_', ' ')} on {sku}"
