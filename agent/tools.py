"""LLM judgement tools: vendor normalisation, duplicate reasoning, and
exception summarisation.

Talks to TensorMux (OpenAI-compatible) when TENSORMUX_API_KEY is set.
Falls back to a deterministic heuristic otherwise, so the pipeline still
runs end to end offline (eval, local dev, demo rehearsal). Every result
carries "llm_used" so callers/metrics can tell which path ran.
"""

import difflib
import json
import os
import time


def _load_dotenv():
    """Populate os.environ from a .env file at the repo root, without adding
    a python-dotenv dependency. Existing env vars always win."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

TENSORMUX_BASE_URL = "https://api.tensormux.com/v1"
TENSORMUX_MODEL = "glm-4-7-flash"
PRICE_PER_1K_TOKENS_INR = float(os.environ.get("TENSORMUX_PRICE_PER_1K_INR", "0"))

_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
_client = None
_client_checked = False


def get_usage():
    return dict(_usage)


def reset_usage():
    _usage.update({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0})


def _get_client():
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    api_key = os.environ.get("TENSORMUX_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(base_url=TENSORMUX_BASE_URL, api_key=api_key)
    except Exception:
        _client = None
    return _client


def _call_llm_json(system, user):
    """Return (parsed_json_or_None, llm_used_bool)."""
    client = _get_client()
    if client is None:
        return None, False

    start = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=TENSORMUX_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        return None, False
    elapsed = time.monotonic() - start

    _usage["calls"] += 1
    _usage["latency_s"] += elapsed
    if resp.usage:
        _usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
        _usage["completion_tokens"] += resp.usage.completion_tokens or 0

    try:
        return json.loads(resp.choices[0].message.content), True
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None, True


def normalize_vendor(invoice_vendor, po_vendor):
    """Decide whether invoice_vendor and po_vendor are the same legal entity."""
    system = (
        "You are an accounts-payable analyst. Decide whether two vendor name "
        "strings refer to the same legal entity (e.g. legal-suffix differences, "
        "punctuation, capitalisation, abbreviation). Respond as JSON: "
        '{"same_entity": bool, "confidence": float 0-1, "reasoning": str}.'
    )
    user = json.dumps({"invoice_vendor": invoice_vendor, "po_vendor": po_vendor})
    parsed, llm_used = _call_llm_json(system, user)
    if parsed is not None:
        parsed["llm_used"] = llm_used
        parsed["model"] = TENSORMUX_MODEL
        return parsed

    ratio = difflib.SequenceMatcher(None, invoice_vendor.lower(), po_vendor.lower()).ratio()
    same_entity = ratio >= 0.6
    return {
        "same_entity": same_entity,
        "confidence": round(ratio, 2),
        "reasoning": f"Heuristic string similarity {ratio:.2f} between '{invoice_vendor}' and '{po_vendor}'",
        "llm_used": False,
    }


def reason_duplicate(invoice, candidates):
    """Decide whether invoice is a duplicate of one of the candidate posted entries."""
    system = (
        "You are an accounts-payable analyst checking for duplicate invoice "
        "submissions. You are given a new invoice and one or more already-posted "
        "invoices with the same vendor and a very close amount. Decide if the new "
        "invoice is a duplicate of one of them (common pattern: same amount/vendor/date, "
        "invoice number modified with a suffix or transposed digits). Respond as JSON: "
        '{"is_duplicate": bool, "related_invoice_id": str or null, "confidence": float 0-1, '
        '"reasoning": str}.'
    )
    user = json.dumps({
        "invoice": {"id": invoice["id"], "vendor": invoice["vendor"],
                    "total_amount": invoice["total_amount"], "invoice_date": invoice["invoice_date"]},
        "candidates": candidates,
    })
    parsed, llm_used = _call_llm_json(system, user)
    if parsed is not None:
        parsed["llm_used"] = llm_used
        parsed["model"] = TENSORMUX_MODEL
        return parsed

    best = candidates[0]
    similarity = difflib.SequenceMatcher(None, invoice["id"], best["invoice_id"]).ratio()
    is_duplicate = similarity >= 0.7
    return {
        "is_duplicate": is_duplicate,
        "related_invoice_id": best["invoice_id"] if is_duplicate else None,
        "confidence": round(similarity, 2),
        "reasoning": (
            f"Heuristic: vendor+amount match posted invoice {best['invoice_id']}, "
            f"invoice-number similarity {similarity:.2f}"
        ),
        "llm_used": False,
    }


def summarize_exception(invoice, po, grn, findings):
    """Produce a plain-English summary and a proposed resolution for an exception packet."""
    system = (
        "You are an accounts-payable analyst writing an exception summary for a "
        "clerk who has 10 seconds to read it. Be concrete: name the amounts and "
        "fields involved, no jargon, no JSON-speak in the prose fields. Respond as JSON: "
        '{"summary": str, "proposed_resolution": str, "confidence": float 0-1}. '
        "Be conservative: only propose auto-approve-style resolutions when the evidence "
        "is unambiguous; otherwise propose the safer action (reject / route to approver)."
    )
    user = json.dumps({
        "invoice": invoice,
        "po": po,
        "grn": grn,
        "findings": findings,
    }, default=str)
    parsed, llm_used = _call_llm_json(system, user)
    if parsed is not None:
        parsed["llm_used"] = llm_used
        parsed["model"] = TENSORMUX_MODEL
        return parsed

    types = [f["type"] for f in findings]
    summary = f"Invoice {invoice['id']} from {invoice['vendor']} flagged for: {', '.join(types)}."
    resolution = "Hold for manual review; evidence insufficient for automated resolution."
    if "suspected_duplicate" in types:
        related = next((f["detail"].get("candidates", [{}])[0].get("invoice_id")
                         for f in findings if f["type"] == "suspected_duplicate"), None)
        resolution = f"Reject as duplicate. Related invoice already posted: {related}."
    elif "missing_po" in types:
        resolution = "Route to approver for non-PO invoice sign-off."
    elif types:
        resolution = f"Confirm {types[0].replace('_', ' ')} with vendor/receiving before posting."

    return {
        "summary": summary,
        "proposed_resolution": resolution,
        "confidence": 0.5,
        "llm_used": False,
    }
