"""What phrasings does the Synovitis extractor miss?

Synovitis is the weakest of the twelve: extractor AUC 0.621, recall 0.37, and
"never mentioned" in 88% of Latin / 90% of Greek / 98% of Cyrillic reports.
Topology alone scores 0.810 on it, so the signal is in the images and the
*teacher* is what fails. That makes the term bank, not the model, the thing to
fix.

Two views, because each is blind where the other sees:

  [A] The 58 gold studies. For every study where the radiologist marked
      Synovitis but our extractor found no mention, print the report. Small
      (27 positives) but it is the only ground truth in existence.

  [B] All 4407 reports. Rank vocabulary by lift: how much more often a token
      appears in reports our extractor calls Synovitis-positive than in the
      rest. Terms we already match will top the list, which is the control;
      what matters is the unmatched terms that ride alongside them, since those
      are candidate synonyms in languages whose vocabulary we never entered.

Usage:  .venv/bin/python src/probe_synovitis.py [label]
"""
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import extract_rules as R

LABEL = sys.argv[1] if len(sys.argv) > 1 else "Synovitis"
BANKS = {"Synovitis": R.SYNOVITIS, "Effusion": R.EFFUSION,
         "Contusion": R.CONTUSION, "Fracture": R.FRACTURE,
         "MCL": getattr(R, "MCL", None)}
BANK = BANKS.get(LABEL)

df = pd.read_csv("data/train.csv")
gold = df[df["ACL"].notna()]
print(f"reports {len(df)} | gold {len(gold)} | label {LABEL}")
if BANK:
    print(f"current term bank ({len(BANK)}): {', '.join(BANK)}\n")


def norm(t):
    """Same normalization the extractor uses, so 'matched' means the same thing."""
    return unicodedata.normalize("NFKD", str(t)).casefold()


def matches(text):
    n = norm(text)
    return [t for t in BANK if t in n] if BANK else []


# ---- [A] gold positives the extractor never mentions --------------------------
print("=" * 78)
print(f"[A] gold {LABEL}=1 studies where the extractor found NO mention")
print("=" * 78)
pos = gold[gold[LABEL] == 1]
missed = [r for _, r in pos.iterrows() if not matches(r["Report"])]
print(f"{len(missed)} of {len(pos)} gold positives are silent to our terms "
      f"({len(missed)/max(len(pos),1):.0%})\n")
for r in missed[:12]:
    txt = re.sub(r"\s+", " ", str(r["Report"])).strip()
    print(f"  - {txt[:320]}")
    print()

# a matched positive, for contrast: shows what a hit looks like
hit = [r for _, r in pos.iterrows() if matches(r["Report"])]
if hit:
    print(f"  (for contrast, {len(hit)} gold positives DO match, e.g. via "
          f"{matches(hit[0]['Report'])})")

# ---- [B] vocabulary lift over all 4407 -----------------------------------------
print("\n" + "=" * 78)
print(f"[B] tokens enriched in reports our extractor calls {LABEL}-positive")
print("=" * 78)
flag = df["Report"].apply(lambda t: bool(matches(t)))
print(f"extractor-positive: {flag.sum()} / {len(df)}\n")

TOKEN = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def toks(series):
    c = Counter()
    for t in series:
        c.update(set(TOKEN.findall(norm(t))))
    return c


inside, outside = toks(df.Report[flag]), toks(df.Report[~flag])
n_in, n_out = max(int(flag.sum()), 1), max(int((~flag).sum()), 1)
rows = []
for w, k in inside.items():
    if k < 4:
        continue
    p_in, p_out = k / n_in, outside.get(w, 0) / n_out
    lift = p_in / (p_out + 1e-6)
    if lift > 3:
        rows.append((lift, w, k, outside.get(w, 0)))
rows.sort(reverse=True)

known = set()
for t in (BANK or []):
    known.update(TOKEN.findall(norm(t)))
print(f"  {'token':<28}{'lift':>8}{'in':>7}{'out':>8}   already in bank?")
for lift, w, k, o in rows[:45]:
    seen = any(w.startswith(t) or t.startswith(w) for t in known)
    print(f"  {w:<28}{lift:>8.1f}{k:>7}{o:>8}   {'yes' if seen else '<-- NEW'}")

print(f"\n{len(rows)} tokens with lift > 3. 'NEW' ones co-occur with the "
      f"finding\nbut are not matched: candidate synonyms to verify by reading "
      f"context.")
