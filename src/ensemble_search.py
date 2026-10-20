"""Ensemble search over cached validation predictions. Local, seconds.

Members are trained in separate Kaggle kernels (they use different caches and
different backbones, and the two caches cannot be attached to one kernel because
both are written as `slices_224_12x3.npy`). Each kernel saves its val predictions
to an npz; this merges them and searches every subset.

**The number that matters is correlation, not accuracy.** Measured on this
project:

    v13 336px       gold 0.8533   corr 0.934   contributed nothing (+0.0004)
    v11 base        gold 0.8541   corr 0.934   contributed +0.010 LB
    v14 RadImageNet gold 0.8508   corr 0.856   contributed +0.016 LB

The member with the *lowest* gold gave the largest gain. Being different beats
being good, so this prints the correlation of every member with the shipped
average alongside the subset scores.

Row alignment across files is asserted, never assumed: every training kernel
builds `val` from the same rng(0) permutation over the same study order, and
silently misaligned rows would produce a plausible but meaningless number.

**This search ignores runtime.** It once returned a 6-member combination that
ran past the 9 h submission cap and scored nothing. Use
`src/ensemble_cost_search.py` for anything that will actually be submitted;
this one is for studying member behaviour.

Usage:  .venv/bin/python src/ensemble_search.py
"""
import itertools
import os

import numpy as np
from sklearn.metrics import roc_auc_score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# every kernel that saved val predictions; missing files are skipped so this
# keeps working as members are added or retired
SOURCES = [
    "kaggle/ensemble_search3/out/val_predictions_v3.npz",
    "kaggle/valpreds_336/out/val_predictions_336.npz",
    "kaggle/valpreds_radimg/out/val_predictions_radimg.npz",
    "kaggle/valpreds_effnet/out/val_predictions_effnet.npz",
    "kaggle/finetune_radimg_folds/out/val_predictions_radfolds.npz",
    "kaggle/valpreds_radv2/out/val_predictions_radv2.npz",
    "kaggle/valpreds_25d/out/val_predictions_25d.npz",
    # was MISSING until 2026-08-27, so this search could not see v19_fullA/fullB --
    # and v19_fullB is a member of the shipped ens7. A search blind to its own
    # incumbent has bitten this project twice now.
    "kaggle/valpreds_full/out/val_predictions_full.npz",
]
# the combination currently on the leaderboard, and what it scored there
SHIPPED = ("v10_small", "v11_base", "v17_radA", "v19_fullB")   # ens7
SHIPPED_LB = 0.908

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

g = np.load(os.path.join(R, "kaggle/ensemble_search/out/val_predictions.npz"),
            allow_pickle=True)
gm, Yg = g["gold_mask"].astype(bool), g["Y_gold"]


def macro(p):
    out = []
    for j in range(len(LABELS)):
        y = Yg[gm, j]
        if len(np.unique(y)) == 2:
            out.append(roc_auc_score(y, p[gm, j]))
    return float(np.mean(out))


missing = [m for m in SHIPPED if m not in P]
assert not missing, f"shipped members missing: {missing}"
shipped = macro(np.mean([P[x] for x in SHIPPED], axis=0))
base_avg = np.mean([P[x] for x in SHIPPED], axis=0)

print(f"\n  {'member':<16}{'gold':>9}{'corr w/ shipped':>18}")
for n in sorted(P, key=lambda k: -macro(P[k])):
    c = np.corrcoef(P[n][gm].ravel(), base_avg[gm].ravel())[0, 1]
    flag = "  <-- decorrelated" if c < 0.90 else ""
    print(f"  {n:<16}{macro(P[n]):>9.4f}{c:>18.3f}{flag}")

rows = sorted(((macro(np.mean([P[x] for x in c], axis=0)), c)
               for k in range(1, len(P) + 1)
               for c in itertools.combinations(P, k)), reverse=True)

print(f"\n  {'combination':<58}{'gold':>9}")
for s, c in rows[:12]:
    mark = f"  <-- shipped, LB {SHIPPED_LB}" if set(c) == set(SHIPPED) else ""
    print(f"  {'+'.join(c):<58}{s:>9.4f}{mark}")

best_s, best_c = rows[0]
print(f"\n  shipped {'+'.join(SHIPPED)}")
print(f"    {shipped:.4f}  (LB {SHIPPED_LB})")
print(f"  best    {'+'.join(best_c)}")
print(f"    {best_s:.4f}")
print(f"  margin over shipped: {best_s - shipped:+.4f}")

for cand in [n for n in P if n not in SHIPPED]:
    with_c = max(s for s, c in rows if cand in c)
    print(f"  best containing {cand:<16}{with_c:.4f}  "
          f"({with_c - shipped:+.4f} vs shipped)")

print("\n  Gold has tracked ensemble margins well (+0.0107 predicted, +0.010")
print("  delivered; +0.0108 predicted, +0.016 delivered) but got an")
print("  architecture swap backwards. Under ~0.005 here, treat as unresolved.")
