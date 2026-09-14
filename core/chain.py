"""Hash-chain primitives shared by the ledger and the vendor-master change log.

Both stores need the same property: an append-only record where an edit, a
deletion or a reordering after the fact is detectable. They differ only in
which fields the hash covers, so that is the one thing a caller supplies.

The guarantee and its limit, stated once here:

  A chain makes tampering *evident*, not impossible. Anyone with write access
  to the file can recompute the whole chain and produce a self-consistent
  forgery. What the chain buys is that a casual or partial edit -- the realistic
  case, someone deleting the original posting so a duplicate clears -- cannot
  be done without leaving a break at a specific sequence number. Making it
  tamper-*proof* needs an anchor outside the file (a countersigned head, a
  WORM store, an external notary), which this project does not have and does
  not pretend to.
"""

import hashlib
import json

GENESIS_HASH = "0" * 64

HASH_FIELD = "entry_hash"
PREV_FIELD = "prev_hash"
SEQ_FIELD = "seq"


def canonical_payload(entry, fields):
    """The exact bytes the hash covers.

    sort_keys plus separators without spaces, so the digest cannot change
    because someone reformatted the file or a JSON writer changed its spacing.
    """
    payload = {k: entry.get(k) for k in fields}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(entry, fields):
    return hashlib.sha256(canonical_payload(entry, fields).encode("utf-8")).hexdigest()


def hash_of_unchained(entry, fields):
    """Hash a record written before chaining existed, so the chain can start on
    top of it. This attests to nothing about that record -- `verify` says so."""
    return compute_hash({**entry, PREV_FIELD: GENESIS_HASH}, fields)


def head(entries):
    if not entries:
        return GENESIS_HASH
    last = entries[-1]
    return last.get(HASH_FIELD) or GENESIS_HASH


def append(entries, record, fields, seq_field=SEQ_FIELD):
    """Return a new record, chained to the tail of `entries`. Does not mutate
    `entries`; the caller decides when to persist."""
    prev = entries[-1] if entries else None
    if prev is None:
        prev_hash = GENESIS_HASH
    else:
        prev_hash = prev.get(HASH_FIELD) or hash_of_unchained(prev, fields)

    entry = dict(record)
    entry[seq_field] = len(entries) + 1
    entry[PREV_FIELD] = prev_hash
    entry[HASH_FIELD] = compute_hash(entry, fields)
    return entry


def verify(entries, fields, seq_field=SEQ_FIELD, label="entry"):
    """Walk the chain and report every break, rather than raising on the first.

    The useful answer to "is this intact" is *where* it stopped being intact.

    Problem kinds:
      unchained      record predates hashing; its integrity is not attested
      hash_mismatch  contents do not match the record's own hash -- it was edited
      broken_link    prev_hash does not match the previous record -- something
                     was deleted, inserted or reordered at this point
      bad_sequence   seq is not 1..n in order
    """
    problems = []
    prev_hash = GENESIS_HASH
    chained_from = None

    for index, entry in enumerate(entries):
        seq = entry.get(seq_field)
        ident = {"seq": seq if seq is not None else index + 1, "label": label}

        if not entry.get(HASH_FIELD):
            problems.append({**ident, "kind": "unchained",
                             "detail": f"{label} has no hash; it predates the hash chain "
                                       f"and its integrity is not attested."})
            prev_hash = hash_of_unchained(entry, fields)
            continue

        if chained_from is None:
            chained_from = ident["seq"]

        if seq != index + 1:
            problems.append({**ident, "kind": "bad_sequence",
                             "detail": f"{label} is at position {index + 1} but carries seq {seq}."})

        if compute_hash(entry, fields) != entry[HASH_FIELD]:
            problems.append({**ident, "kind": "hash_mismatch",
                             "detail": f"{label} contents do not match its recorded hash; "
                                       f"it was edited after it was written."})

        if entry.get(PREV_FIELD) != prev_hash:
            problems.append({**ident, "kind": "broken_link",
                             "detail": f"prev_hash does not match the preceding {label}'s hash; "
                                       f"a record was deleted, inserted or reordered here."})

        prev_hash = entry[HASH_FIELD]

    return {
        "ok": not problems,
        "entries": len(entries),
        "chained_from": chained_from,
        "head": head(entries),
        "problems": problems,
    }
