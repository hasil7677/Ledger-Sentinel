"""Approval policy: per-requester authority limits, and detection of the
classic evasion -- splitting one purchase into several invoices that each
stay under the limit.

Nothing here is a workflow engine. There is no multi-step sign-off state
machine, no named second approver recorded against an invoice -- that would
be a materially bigger feature (approval routing, reviewer identity, a
pending/approved/rejected lifecycle per approval) and this project does not
have a UI for a second human to act in. What is genuinely buildable and
useful with a one-clerk review queue is the same shape as the rest of the
system: a deterministic rule that decides *whether* an invoice needed
authority its requester didn't have, and escalates when it did. The human
in the queue is the control; this is what puts the right invoices in front
of them.

Two findings:

  approval_threshold_exceeded  one invoice, on its own, exceeds the
                               requester's approval_limit.
  threshold_splitting          several invoices from the same requester to
                               the same vendor, each individually within
                               the requester's limit, whose sum within a
                               time window exceeds it. The individual-limit
                               check above would miss every one of them.
"""

from datetime import date

# How far back to look for a requester's other recent invoices to the same
# vendor when checking for structuring. 14 days: long enough to catch a
# "split it across two weeks so no single one draws attention" pattern,
# short enough that unrelated purchases months apart don't get pulled in.
SPLIT_WINDOW_DAYS = 14

# An invoice individually over the limit is unambiguous and needs no window;
# a group is only a group if it has at least this many members.
MIN_SPLIT_GROUP_SIZE = 2


def required_approval_ok(amount, approval_limit):
    """True if a requester with this limit could approve this amount alone."""
    return amount <= approval_limit


def check_single_invoice(invoice_amount, requester):
    """Does this one invoice, by itself, exceed what its requester may approve?

    Returns a finding dict or None. `requester` is an employee record
    (dict with 'approval_limit'); None means "no requester on record", which
    is treated as automatic escalation -- an invoice with no accountable
    requester is not a case this control can wave through.
    """
    if requester is None:
        return {
            "type": "approval_threshold_exceeded",
            "detail": {
                "reason": "no_requester_on_record",
                "amount": invoice_amount,
            },
        }
    limit = requester["approval_limit"]
    if invoice_amount <= limit:
        return None
    return {
        "type": "approval_threshold_exceeded",
        "detail": {
            "requester_id": requester["employee_id"],
            "requester_name": requester["name"],
            "approval_limit": limit,
            "amount": invoice_amount,
            "over_by": round(invoice_amount - limit, 2),
        },
    }


def check_split(invoice, requester, prior_entries, window_days=SPLIT_WINDOW_DAYS):
    """Does this invoice, combined with the requester's other recent invoices
    to the same vendor, look like a purchase deliberately broken up to stay
    under the approval limit?

    `prior_entries` are already-posted ledger entries for the same vendor,
    filtered to the window by the caller (core.ledger.Ledger.entries_for_vendor
    does this); each must carry `requester_id`. Only entries from the *same*
    requester count toward the group -- two different people buying from the
    same vendor in the same week is not evidence of anything.

    A group is only flagged if every member, including this invoice, is
    individually within the limit -- if one member already exceeds it alone,
    `check_single_invoice` already catches that member on its own, and calling
    it "splitting" as well would double-count the same fact under two names.
    """
    if requester is None:
        return None
    limit = requester["approval_limit"]
    if invoice["total_amount"] > limit:
        return None  # already flagged on its own; not a splitting case

    same_requester = [e for e in prior_entries if e.get("requester_id") == requester["employee_id"]]
    if not same_requester:
        return None

    group = same_requester + [{
        "invoice_id": invoice["id"],
        "total_amount": invoice["total_amount"],
        "invoice_date": invoice["invoice_date"],
    }]
    if len(group) < MIN_SPLIT_GROUP_SIZE:
        return None
    if any(m["total_amount"] > limit for m in group):
        return None  # a member individually over limit is its own finding, not a group pattern

    total = sum(m["total_amount"] for m in group)
    if total <= limit:
        return None

    return {
        "type": "threshold_splitting",
        "detail": {
            "requester_id": requester["employee_id"],
            "requester_name": requester["name"],
            "approval_limit": limit,
            "window_days": window_days,
            "group_total": round(total, 2),
            "over_by": round(total - limit, 2),
            "members": [{"invoice_id": m["invoice_id"], "amount": m["total_amount"],
                        "invoice_date": m["invoice_date"]} for m in group],
        },
    }


def check_invoice(invoice, requester, prior_entries, window_days=SPLIT_WINDOW_DAYS):
    """Run both approval checks for one invoice. Returns a list of 0-2 findings."""
    findings = []
    single = check_single_invoice(invoice["total_amount"], requester)
    if single:
        findings.append(single)
    else:
        split = check_split(invoice, requester, prior_entries, window_days=window_days)
        if split:
            findings.append(split)
    return findings
