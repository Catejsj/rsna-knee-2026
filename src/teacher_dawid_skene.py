"""Latent-label teacher: combine every public label source with Dawid-Skene. Local.

The true finding is treated as a hidden variable. Each source gets its own
per-label sensitivity and specificity, estimated by EM from how the sources
agree with EACH OTHER -- no gold, no image model. Gold-58 is used only to score
the result, so it stays a pure ruler.

Why this and not the alternatives: noise-transition matrices and gold-guided
meta-learning fit on the 58 and leave nothing to measure with; AUM and gradient
variance read the model's own training behaviour, and AUM turned out to flag
rare positives rather than report omissions (`rsna-finetune-aum`).

Honest limits, stated before the result:
  * the LLM teachers all read the SAME report, so their errors are correlated
    -- Dawid-Skene assumes conditional independence and will over-trust
    agreement;
  * every source is bounded by what the report says. No combination recovers a
    Synovitis the report never mentions;
  * the label lever has saturated (+0.023 teacher -> +0.004 model). The bar is
    leehans v4_blend at 0.8932, by a margin, not by noise.

**Gold contamination check.** Some public sources were built by people who also
had the 58 gold labels and may have tuned thresholds on them or copied them in.
Such a source scores inflated *on gold*. The `gold==` column reports how often a
source's binarised gold rows exactly equal gold; anything near 1.0 is not a
measurement.

Only StudyInstanceUID and the 12 label columns are ever read. Several of these
files also carry the report text; it is never loaded.

Usage:  .venv/bin/python src/teacher_dawid_skene.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# one sub-folder per public Kaggle label dataset (kaggle datasets download --unzip)
T = os.environ.get("TEACHER_DIR", os.path.join(R, "external", "teachers"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# name -> (path, column template). `{}` is the label name.
SOURCES = {
    "pilkwang":     (f"{R}/derived/report_labels_v2.csv", "{}"),
    "rules_v5":     (f"{R}/derived/labels_rules_v5.csv", "{}"),
    "leehans_v4":   (f"{T}/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", "{}"),
    "leehans_v2":   (f"{T}/rsna-knee-llm-report-labels/llm_labels_v2.csv", "{}"),
    "leehans_full": (f"{T}/rsna-knee-llm-report-labels/llm_labels_full.csv", "{}"),
    "lixin":        (f"{T}/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv", "{}"),
    "qwen":         (f"{T}/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv", "{}__label"),
    "yehezkiel":    (f"{T}/psuedo-labeling-knee/train.csv", "{}"),
    "flight":       (f"{T}/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv", "{}"),
    "jev_final2":   (f"{T}/jev-knee-labels/FINAL2_labels_scores.csv", "{}"),
}
# one source per author, so correlated siblings (leehans v2/full) are not
# counted as independent votes
# yehezkiel is excluded: it covers exactly the 58 gold studies and equals gold
# 100% -- a copy of the competition train.csv labels, not an annotator.
DS_SET = ["pilkwang", "rules_v5", "leehans_v4", "lixin", "qwen", "flight"]
JEV = "jev_final2"     # made by sending reports to an external API -- kept separate

train = pd.read_csv(f"{R}/data/train.csv", usecols=["StudyInstanceUID"] + LABELS)
gold = train[train["ACL"].notna()].set_index("StudyInstanceUID")
ids = train["StudyInstanceUID"].values
gold_ids = gold.index.values
Yg = gold[LABELS].values.astype(int)

S = {}
for name, (path, tmpl) in SOURCES.items():
    if not os.path.exists(path):
        print(f"  missing: {name}")
        continue
    cols = [tmpl.format(l) for l in LABELS]
    df = pd.read_csv(path, usecols=["StudyInstanceUID"] + cols)   # never the Report
    df = df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    df.columns = LABELS
    S[name] = df.apply(pd.to_numeric, errors="coerce")


def macro(P, rows=None):
    """Macro AUC on gold; P is [58,12] aligned to gold_ids, NaN = no prediction."""
    out = []
    for j in range(len(LABELS)):
        m = np.isfinite(P[:, j]) if rows is None else rows & np.isfinite(P[:, j])
        y = Yg[m, j]
        if len(np.unique(y)) == 2:
            r = rankdata(P[m, j])
            npos = y.sum(); nneg = len(y) - npos
            out.append((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(out))


def on_gold(df):
    return df.reindex(gold_ids)[LABELS].values.astype(float)


common = np.ones(len(gold_ids), dtype=bool)
for n, df in S.items():
    if n == "yehezkiel":
        continue
    common &= np.isfinite(on_gold(df)).all(1)
print(f"gold studies covered by every source: {common.sum()}/{len(gold_ids)}")

print(f"\n  {'source':<14}{'corpus':>8}{'gold AUC':>10}{'gold==':>8}   values")
for name, df in S.items():
    G = on_gold(df)
    cov = df.reindex(ids)[LABELS].notna().all(1).mean()
    exact = ((G[common] >= 0.5).astype(int) == Yg[common]).mean()
    vals = np.unique(df.values[np.isfinite(df.values)])
    vs = f"{len(vals)} levels" if len(vals) > 6 else ",".join(f"{v:g}" for v in vals)
    flag = "  <-- near-copy of gold?" if exact > 0.97 else ""
    print(f"  {name:<14}{cov:>8.1%}{macro(G, common):>10.4f}{exact:>8.3f}   {vs}{flag}")


def dawid_skene(names, iters=50):
    """Binary Dawid-Skene per label. Returns posterior P(true=1), [n_ids, 12]."""
    V = np.stack([S[n].reindex(ids)[LABELS].values.astype(float) for n in names])
    obs = np.isfinite(V)
    B = (V >= 0.5).astype(float)
    post = np.zeros((len(ids), len(LABELS)))
    for j in range(len(LABELS)):
        b, o = B[:, :, j], obs[:, :, j]                 # [sources, studies]
        # init: per-study mean vote (majority)
        q = np.where(o.sum(0) > 0, (b * o).sum(0) / np.maximum(o.sum(0), 1), 0.5)
        for _ in range(iters):
            pi = np.clip(q.mean(), 1e-3, 1 - 1e-3)
            w1, w0 = q, 1 - q
            sens = (((b * o * w1).sum(1) + 1) / ((o * w1).sum(1) + 2))[:, None]        # P(say 1 | 1)
            spec = ((((1 - b) * o * w0).sum(1) + 1) / ((o * w0).sum(1) + 2))[:, None]  # P(say 0 | 0)
            l1 = np.log(pi) + (o * (b * np.log(sens) + (1 - b) * np.log(1 - sens))).sum(0)
            l0 = np.log(1 - pi) + (o * ((1 - b) * np.log(spec) + b * np.log(1 - spec))).sum(0)
            q = 1 / (1 + np.exp(np.clip(l0 - l1, -50, 50)))
        post[:, j] = q
    return post


def mean_of(names):
    V = np.stack([S[n].reindex(ids)[LABELS].values.astype(float) for n in names])
    return np.nanmean(V, 0)


idx = pd.Index(ids)
gi = idx.get_indexer(gold_ids)
print("\ncombinations (gold AUC on the common subset):")
for title, names in [("D-S, 6 authors", DS_SET),
                     ("D-S, 6 authors + jev", DS_SET + [JEV]),
                     ("D-S, LLM authors only", [n for n in DS_SET if n != "rules_v5"])]:
    names = [n for n in names if n in S]
    ds = dawid_skene(names)[gi]
    mn = mean_of(names)[gi]
    # rank-mean too: sources use different scales (0/1, 5 levels, 200 levels),
    # so a plain mean lets the finest-grained source dominate the ordering
    rk = np.nanmean(np.stack([np.apply_along_axis(rankdata, 0,
                     np.nan_to_num(S[n].reindex(ids)[LABELS].values.astype(float), nan=0.5))
                     for n in names]), 0)[gi]
    print(f"  {title:<26} D-S {macro(ds, common):.4f}   mean {macro(mn, common):.4f}"
          f"   rank-mean {macro(rk, common):.4f}")
print(f"\n  bar: leehans_v4 {macro(on_gold(S['leehans_v4']), common):.4f}"
      "  (gold deltas under ~0.015 are noise)")
print("  NB: D-S posteriors are near-binary; AUC ties among confident studies are")
print("  expected and are why the plain mean can rank better at equal accuracy.")
