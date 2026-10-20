"""Best ensemble that actually FITS the 9 h submission cap. Local, seconds.

`ensemble_search.py` maximises gold with no notion of cost. That is how a
6-member combination estimated at 9.8 h got shipped, ran to the wall and scored
**nothing** -- the submission reads COMPLETE with an empty publicScore, which is
what a timeout looks like. There is no error message.

Runtime is described throughout the handoff as the binding constraint, so this
searches the same subsets under a cost model instead.

### The cost model, and what anchors it

Per-member CPU cost on ~1300 test studies, from the handoff's measured table,
plus a fixed overhead for the shared DICOM decode and kernel startup:

    fixed (decode + startup)   1.5 h
    DINOv2-small               1.0 h
    DINOv2-base                3.0 h     <-- one member for the price of three
    ResNet-50                  1.0 h
    EfficientNet-B3            1.0 h     (assumed; never timed on its own)
    DINOv2-small @336px        2.75 h    + 0.5 h for a SECOND decode pass

Calibrated against three observed runs, one of them a failure:

    small+base+leehans+radimg   6.0 + 1.5 = 7.5 h   observed 7.5 h, COMPLETED
    ens7 small+base+radA+fullB  6.0 + 1.5 = 7.5 h   observed ~7.5 h, COMPLETED
    the same 6 + radf0 + radf2  8.0 + 1.5 = 9.5 h   estimated 9.8 h, TIMED OUT

Two completions and one timeout bracket the model, which is the most that can be
claimed from three points. Treat it as calibrated, not exact.

### Caps

    7.5 h   proven -- a combination of this cost has finished twice
    8.5 h   safe margin under the rule
    9.0 h   the rule itself; 9.5 h is known to fail, so this is the cliff edge

**9 h is the competition rule for the scored rerun. 12 h is the CPU kernel limit
and does not apply to submissions.**

Usage:  .venv/bin/python src/ensemble_cost_search.py
"""
import itertools
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# NB `valpreds_full` was missing from ensemble_search.py's SOURCES, so that
# search could not see v19_fullA/v19_fullB -- and v19_fullB is a member of the
# SHIPPED ens7. Same class of bug as the search that once excluded its own
# incumbent. Every npz that exists is listed here.
SOURCES = [
    "kaggle/ensemble_search3/out/val_predictions_v3.npz",
    "kaggle/valpreds_336/out/val_predictions_336.npz",
    "kaggle/valpreds_radimg/out/val_predictions_radimg.npz",
    "kaggle/valpreds_effnet/out/val_predictions_effnet.npz",
    "kaggle/finetune_radimg_folds/out/val_predictions_radfolds.npz",
    "kaggle/valpreds_radv2/out/val_predictions_radv2.npz",
    "kaggle/valpreds_25d/out/val_predictions_25d.npz",
    "kaggle/valpreds_full/out/val_predictions_full.npz",
]

FIXED_H = 1.5
DECODE_336_H = 0.5
COST = {                       # member -> (hours, family)
    "v9_small":    (1.0, "dinov2-small"),
    "v10_small":   (1.0, "dinov2-small"),
    "v12_leehans": (1.0, "dinov2-small"),
    "v11_base":    (3.0, "dinov2-base"),
    "v13_336":     (2.75, "dinov2-small@336"),
    "v14_radimg":  (1.0, "resnet50"),
    "v15_effnet":  (1.0, "effnet-b3"),
    "v17_radA":    (1.0, "resnet50"),
    "v17_radB":    (1.0, "resnet50"),
    "v18_25d":     (1.0, "resnet50"),
    "v19_fullA":   (1.0, "resnet50"),
    "v19_fullB":   (1.0, "resnet50"),
    # 5-fold RadImageNet. Averaged they equal a single model (0.8506 vs 0.8508)
    # because folds correlate 0.938-0.971 -- different data does not decorrelate,
    # different inductive bias does. Individually each is more decorrelated from
    # the ViTs than the full model, so two of them were worth +0.0036.
    "v16_radf0":   (1.0, "resnet50-fold"),
    "v16_radf1":   (1.0, "resnet50-fold"),
    "v16_radf2":   (1.0, "resnet50-fold"),
    "v16_radf3":   (1.0, "resnet50-fold"),
    "v16_radf4":   (1.0, "resnet50-fold"),
}
SHIPPED = ("v10_small", "v11_base", "v17_radA", "v19_fullB")   # ens7, LB 0.908
SHIPPED_LB = 0.908
CAPS = [(7.5, "proven"), (8.5, "safe margin"), (9.0, "the rule / cliff edge")]

P, ref = {}, None
for rel in SOURCES:
    path = os.path.join(R, rel)
    if not os.path.exists(path):
        print(f"  skip (missing): {rel}")
        continue
    d = np.load(path, allow_pickle=True)
    if ref is None:
        ref = d["val_index"]
    assert np.array_equal(ref, d["val_index"]), f"val split differs in {rel}"
    for i, n in enumerate(d["names"]):
        P[str(n)] = d["plain"][i]
print(f"val rows aligned: {len(ref)} | members: {len(P)}")

unpriced = [m for m in P if m not in COST]
assert not unpriced, f"no cost for {unpriced} -- add it rather than guessing"

g = np.load(os.path.join(R, "kaggle/ensemble_search/out/val_predictions.npz"),
            allow_pickle=True)
gm, Yg = g["gold_mask"].astype(bool), g["Y_gold"]
print(f"gold studies in val: {gm.sum()}")


# macro() only ever reads the gold rows, so keep just those: 58 rows instead of
# 451 makes the 100k-subset sweep tractable without changing any number.
Pg = {m: P[m][gm] for m in P}
Yg_g = Yg[gm]
USABLE = [j for j in range(len(LABELS)) if len(np.unique(Yg_g[:, j])) == 2]


_POS = Yg_g[:, USABLE].astype(bool)          # [58, n_usable]
_NPOS = _POS.sum(0).astype(float)
_NNEG = (~_POS).sum(0).astype(float)


def macro_g(pg):
    """Macro AUC over the usable labels, all 12 at once.

    Identical to averaging `roc_auc_score` per label -- asserted against sklearn
    in `_selftest()` below, ties included -- but one `rankdata` call instead of
    twelve sklearn calls. The sweep is ~131k subsets, where sklearn's per-call
    validation overhead dominates everything else.

    Mann-Whitney: AUC = (sum of positive ranks - npos(npos+1)/2) / (npos*nneg).
    """
    r = rankdata(pg[:, USABLE], axis=0)       # average ranks -> ties handled
    s = (r * _POS).sum(0)
    return float(np.mean((s - _NPOS * (_NPOS + 1) / 2) / (_NPOS * _NNEG)))


def _selftest():
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = rng.random((len(Yg_g), len(LABELS)))
        p[rng.random(p.shape) < 0.15] = 0.5    # force ties
        mine = macro_g(p)
        ref = float(np.mean([roc_auc_score(Yg_g[:, j], p[:, j]) for j in USABLE]))
        assert abs(mine - ref) < 1e-12, f"fast AUC differs: {mine} vs {ref}"
    print("  fast macro-AUC matches sklearn (incl. ties)")


_selftest()


def macro(p):
    return macro_g(p[gm])


def cost(combo):
    h = FIXED_H + sum(COST[m][0] for m in combo)
    if any(COST[m][1].endswith("@336") for m in combo):
        h += DECODE_336_H            # 336 needs its own decode pass
    return h


def avg(combo):
    # probability average -- these arrays are already sigmoid outputs, and the
    # submission kernel does the same (`probs += torch.sigmoid(...)`)
    return np.mean([P[m] for m in combo], axis=0)


shipped_gold = macro(avg(SHIPPED))
shipped_cost = cost(SHIPPED)
print(f"\nshipped ens7 = {'+'.join(SHIPPED)}")
print(f"  gold {shipped_gold:.4f} | cost {shipped_cost:.2f} h | LB {SHIPPED_LB}")

print(f"\n  {'member':<14}{'gold':>8}{'cost':>7}{'corr':>7}   family")
base_avg = avg(SHIPPED)
for n in sorted(P, key=lambda k: -macro(P[k])):
    # per-label correlation, not pooled: pooling also measures shared per-label
    # offsets and inflates apparent decorrelation for anything not BCE-calibrated
    c = np.mean([np.corrcoef(P[n][gm][:, j], base_avg[gm][:, j])[0, 1]
                 for j in range(len(LABELS))])
    print(f"  {n:<14}{macro(P[n]):>8.4f}{COST[n][0]:>6.2f}h{c:>7.3f}   {COST[n][1]}")

MAX_CAP = max(c for c, _ in CAPS)
# Every member costs >=1 h on top of 1.5 h fixed, so nothing above 7 members can
# fit any cap. Bound the sweep by that rather than enumerating all 2^17.
MAX_K = int(MAX_CAP - FIXED_H)
print(f"\nsearching all subsets up to {MAX_K} members "
      f"(nothing larger can fit {MAX_CAP} h) ...")
rows, over = [], []
for k in range(1, MAX_K + 1):
    for c in itertools.combinations(sorted(P), k):
        rows.append((macro_g(np.mean([Pg[m] for m in c], axis=0)), cost(c), c))
rows.sort(key=lambda r: -r[0])
# separately, what the cost-blind search would return, so the two are comparable
for k in range(MAX_K + 1, len(P) + 1):
    for c in itertools.combinations(sorted(P), k):
        over.append((macro_g(np.mean([Pg[m] for m in c], axis=0)), cost(c), c))
over.sort(key=lambda r: -r[0])

print(f"\n  best OVERALL, ignoring cost (this is what the old search returns)")
s, h, c = max(rows + over, key=lambda r: r[0])
print(f"    {'+'.join(c)}")
print(f"    gold {s:.4f} | cost {h:.2f} h"
      + ("  <-- WOULD TIME OUT, scores NOTHING" if h > 9.0 else ""))

for cap, why in CAPS:
    fit = [r for r in rows if r[1] <= cap + 1e-9]
    print(f"\n  === cap {cap} h ({why}) === {len(fit)} of {len(rows)} fit")
    for s, h, c in fit[:5]:
        mark = "  <-- shipped" if set(c) == set(SHIPPED) else ""
        print(f"    {s:>7.4f}  {h:>5.2f}h  {'+'.join(c)}{mark}")
    b_s, b_h, b_c = fit[0]
    print(f"    margin over shipped: {b_s - shipped_gold:+.4f}"
          f"  ({b_h - shipped_cost:+.2f} h)")

# The specific question the cost model exists to answer.
print("\n  Is DINOv2-base worth 3 of the 9 hours?")
for cap, _ in CAPS:
    fit = [r for r in rows if r[1] <= cap + 1e-9]
    with_b = max((r for r in fit if "v11_base" in r[2]), default=None)
    without = max((r for r in fit if "v11_base" not in r[2]), default=None)
    if with_b and without:
        print(f"    cap {cap}h:  with base {with_b[0]:.4f} ({with_b[1]:.2f}h)  |  "
              f"without {without[0]:.4f} ({without[1]:.2f}h)  |  "
              f"base buys {with_b[0]-without[0]:+.4f}")

print("\n  Gold has tracked ensemble margins well (+0.0107 predicted, +0.010")
print("  delivered; +0.0108 predicted, +0.016 delivered) but got an architecture")
print("  swap backwards. Under ~0.005 here, treat as unresolved -- and prefer the")
print("  cheaper combination when two are inside that band, because runtime is")
print("  the constraint that has actually cost a submission.")
