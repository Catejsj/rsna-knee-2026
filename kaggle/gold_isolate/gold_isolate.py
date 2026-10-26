"""Which Tier-1 change actually improved *true* skill: data, or planes?

The label-ceiling run refuted the "we are only fitting our own extractor"
story: with more data, derived-label AUC rose +0.033 AND gold-label AUC rose
+0.027. Improvements do transfer.

But the leaderboard moved only +0.005 for a +0.061 CV move, and that is still
unexplained. The submission changed two things at once (1500 -> 4407 studies
AND 1 plane -> 3 planes); the ceiling run isolated only the first, holding
planes at 3. So the plane contribution has never been checked against gold.

This isolates it. Features are read from the cached matrix, so this costs
minutes instead of the 90-minute decode.

Also reports a per-label gold breakdown and a bootstrap interval, because the
gold set is 58 enriched studies with as few as 9 positives for some labels:
the previous run's seed-to-seed spread understated the real uncertainty, and
that matters more than the point estimates.
"""
import glob
import os

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANES = ("Sagittal", "Coronal", "Axial")
BLOCK = 64 * 64

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])
CACHE = glob.glob("/kaggle/input/**/features_64x24_3plane.npz", recursive=True)[0]

d = np.load(CACHE, allow_pickle=True)
X, keep = d["X"], d["study_ids"]
print(f"cached features {X.shape}", flush=True)

lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

Y_derived = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in keep])
Y_gold = np.zeros((len(keep), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[keep[gold_mask], LABELS].values.astype(int)
print(f"gold studies: {gold_mask.sum()}", flush=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def oof_scores(planes_used, pool_size, seed):
    cols = np.concatenate([np.arange(PLANES.index(p) * BLOCK,
                                     (PLANES.index(p) + 1) * BLOCK)
                           for p in planes_used])
    rng = np.random.default_rng(seed)
    gold_idx = np.where(gold_mask)[0]
    other = rng.permutation(np.where(~gold_mask)[0])
    rows = np.concatenate([gold_idx, other[:max(pool_size - len(gold_idx), 0)]])

    Xc = X[np.ix_(rows, cols)]
    mean, scale = Xc.mean(0), Xc.std(0) + 1e-6
    Z = PCA(n_components=100, random_state=seed,
            svd_solver="randomized").fit_transform((Xc - mean) / scale)

    is_gold = gold_mask[rows]
    derived, per_gold = [], {}
    for j, name in enumerate(LABELS):
        yd = Y_derived[rows, j]
        if yd.sum() < 25 or (1 - yd).sum() < 25:
            continue
        oof = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05),
                                Z, yd, cv=4, method="predict_proba")[:, 1]
        derived.append(roc_auc_score(yd, oof))
        yg = Y_gold[rows][is_gold, j]
        if len(np.unique(yg)) == 2:
            per_gold[name] = (yg, oof[is_gold])
    gold_auc = float(np.mean([roc_auc_score(y, p) for y, p in per_gold.values()]))
    return float(np.mean(derived)), gold_auc, per_gold


def bootstrap_ci(per_gold, n=400, seed=0):
    """Resample the 58 studies, not the seeds. This is the uncertainty that
    matters and the one the previous run did not report."""
    rng = np.random.default_rng(seed)
    n_studies = len(next(iter(per_gold.values()))[0])
    out = []
    for _ in range(n):
        idx = rng.integers(0, n_studies, n_studies)
        vals = []
        for y, p in per_gold.values():
            ys, ps = y[idx], p[idx]
            if len(np.unique(ys)) == 2:
                vals.append(roc_auc_score(ys, ps))
        if vals:
            out.append(np.mean(vals))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


print("\n" + "=" * 76)
print("ISOLATING PLANES (all 4407 studies): scored on derived AND on gold")
print("=" * 76)
print(f"{'planes':<16}{'vs DERIVED':>14}{'vs GOLD (58)':>16}{'gold 95% CI':>24}")
N = len(X)
for ps in [("Sagittal",), ("Sagittal", "Coronal"), PLANES]:
    ds, gs, pg = zip(*[oof_scores(ps, N, s) for s in SEEDS])
    lo, hi = bootstrap_ci(pg[0])
    name = "+".join(p[:3] for p in ps)
    print(f"{name:<16}{np.mean(ds):>14.4f}{np.mean(gs):>16.4f}"
          f"{f'[{lo:.3f}, {hi:.3f}]':>24}", flush=True)

print("\n" + "=" * 76)
print("PER-LABEL, all planes / all studies: derived vs gold")
print("=" * 76)
_, _, pg = oof_scores(PLANES, N, 0)
dv, _, _ = oof_scores(PLANES, N, 0)
print(f"{'label':<20}{'gold AUC':>11}{'n+ gold':>10}")
for name, (y, p) in sorted(pg.items(), key=lambda kv: -roc_auc_score(*kv[1])):
    print(f"{name:<20}{roc_auc_score(y, p):>11.3f}{int(y.sum()):>10}")

print("\nreference: LB 0.633 for this exact config.")
print("DONE", flush=True)
