"""Scoring harness for report -> label extractors.

Scores any derived/labels_<name>.csv against the 58 hand-labeled studies using
the competition metric (macro AUC), and runs two sanity checks the gold set is
too small to provide:

  * derived prevalence across all 4407 studies -- the 58 are an enriched sample
    (every one has >=1 positive finding), so they cannot tell you base rates
  * "never mentioned" rate per script -- a direct read on which languages the
    term banks are missing, since a term-bank gap looks exactly like a negative

Usage:  python3 evaluate.py rules [other_method ...]
Stdlib only.
"""
import sys
import unicodedata

from rsna_labels import LABELS, load_train, gold_rows, read_derived

NOT_MENTIONED = 0.25  # sentinel emitted by extract_rules.score()


def auc(pairs):
    """Rank-based ROC AUC with tie correction. pairs: [(score, label01)]."""
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    ranked = sorted(pairs, key=lambda p: p[0])
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, y) in enumerate(ranked) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def script_of(text):
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            return "Cyrillic"
        if 0x0370 <= o <= 0x03FF:
            return "Greek"
    return "Latin"


def evaluate(name, rows, gold):
    derived = read_derived(name)
    print("=" * 78)
    print(f"METHOD: {name}")
    print("=" * 78)

    print(f"\n[1] Competition metric on the {len(gold)} hand-labeled studies")
    print(f"{'label':<18}{'n+':>4}{'AUC':>8}{'prec':>8}{'rec':>8}{'acc':>8}")
    aucs = []
    for l in LABELS:
        pairs = [(derived[r["StudyInstanceUID"]][l], int(r[l])) for r in gold]
        a = auc(pairs)
        npos = sum(y for _, y in pairs)
        tp = sum(1 for s, y in pairs if s >= 0.5 and y == 1)
        fp = sum(1 for s, y in pairs if s >= 0.5 and y == 0)
        fn = sum(1 for s, y in pairs if s < 0.5 and y == 1)
        tn = sum(1 for s, y in pairs if s < 0.5 and y == 0)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        acc = (tp + tn) / len(pairs)
        if a is not None:
            aucs.append(a)
        astr = f"{a:.3f}" if a is not None else "  n/a"
        print(f"{l:<18}{npos:>4}{astr:>8}{prec:>8.2f}{rec:>8.2f}{acc:>8.2f}")
    macro = sum(aucs) / len(aucs)
    print(f"{'MACRO AUC':<18}{'':>4}{macro:>8.3f}   <- competition metric")
    print(f"  (n=58 enriched sample; treat as a smoke test, not a leaderboard estimate)")

    print(f"\n[2] Derived prevalence across all {len(rows)} studies (threshold 0.5)")
    print(f"{'label':<18}{'derived':>10}{'gold-58':>10}   plausible clinical range")
    ranges = {
        "ACL": "10-25%", "MCL": "5-15%", "Medial Meniscus": "30-50%",
        "Lateral Meniscus": "10-25%", "Medial OA": "25-45%", "Lateral OA": "10-25%",
        "PF OA": "25-45%", "Effusion": "40-60%", "Synovitis": "10-30%",
        "Baker's": "15-30%", "Contusion": "10-25%", "Fracture": "3-10%",
    }
    for l in LABELS:
        d = sum(1 for r in rows if derived[r["StudyInstanceUID"]][l] >= 0.5) / len(rows)
        g = sum(int(r[l]) for r in gold) / len(gold)
        print(f"{l:<18}{d*100:>9.1f}%{g*100:>9.1f}%   {ranges[l]}")
    print("  ranges are rough priors for a symptomatic knee-MRI population,")
    print("  for spotting order-of-magnitude misses -- not ground truth.")

    print("\n[3] 'Never mentioned' rate by script (term-bank coverage)")
    by_script = {}
    for r in rows:
        by_script.setdefault(script_of(r["Report"]), []).append(r["StudyInstanceUID"])
    header = "".join(f"{s:>12}" for s in sorted(by_script))
    print(f"{'label':<18}{header}")
    for l in LABELS:
        cells = ""
        for s in sorted(by_script):
            uids = by_script[s]
            miss = sum(1 for u in uids if abs(derived[u][l] - NOT_MENTIONED) < 1e-9)
            cells += f"{miss/len(uids)*100:>11.0f}%"
        print(f"{l:<18}{cells}")
    counts = "  ".join(f"{s}={len(by_script[s])}" for s in sorted(by_script))
    print(f"  n: {counts}")
    print("  high % = the extractor never found the concept. For Greek/Cyrillic")
    print("  that means a missing term, not a healthy knee.")
    return macro


def main():
    names = sys.argv[1:] or ["rules"]
    rows = load_train()
    gold = gold_rows(rows)
    results = {n: evaluate(n, rows, gold) for n in names}
    if len(results) > 1:
        print("\n" + "=" * 78)
        print("COMPARISON (macro AUC on the 58)")
        for n, m in sorted(results.items(), key=lambda kv: -kv[1]):
            print(f"  {n:<24}{m:.3f}")


if __name__ == "__main__":
    main()
