"""Is our CV measuring disease, or measuring our own label generator?

Tier 1 raised CV by +0.067 and the leaderboard by +0.005. The suspicion: CV is
scored against `rules_v3` -- our own extractor -- so training on 3x more data
mostly taught the model to reproduce the extractor's systematic errors. If so,
true performance is capped by label quality and more image work is wasted.

The test. We hold 58 studies labeled by radiologists. For each training-pool
size, take out-of-fold predictions and score them twice:

    * against DERIVED labels on the whole pool  -- what CV has been reporting
    * against GOLD labels on those 58 studies   -- what actually matters

If derived-AUC climbs with data while gold-AUC stays flat, the ceiling is
confirmed and label quality outranks the encoder.

The 58 are enriched (every one has >=1 finding), so their absolute AUC is not
comparable to the leaderboard. That is fine: both pools are scored on the SAME
58 studies, so the comparison between them is valid even though the level is not.

Side effect, deliberately: writes the 4407 x 12288 feature matrix to
/kaggle/working so it can be published as a dataset. Every later experiment
then skips the 91-minute decode.
"""
import os
import time

import numpy as np
import pandas as pd

import glob
import sys

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

SIZE = 64
SLICES = 24
PLANES = ("Sagittal", "Coronal", "Axial")
SEEDS = (0, 1, 2)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")

gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
print(f"gold (radiologist-labeled) studies: {len(gold_ids)}", flush=True)

studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")

X, keep = [], []
t0 = time.time()
for i, uid in enumerate(studies):
    vec, found = F.study_features_multiplane(
        root, uid, by_study[uid], planes=PLANES, n_slices=SLICES, size=SIZE)
    if found == 0:
        continue
    X.append(vec)
    keep.append(uid)
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

X = np.array(X, dtype=np.float32)
keep = np.array(keep)
print(f"\nfeatures {X.shape} in {(time.time()-t0)/60:.1f} min", flush=True)

col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

np.savez_compressed("/kaggle/working/features_64x24_3plane.npz",
                    X=X, study_ids=keep, col_mean=col_mean,
                    size=SIZE, slices=SLICES, planes=np.array(PLANES))
print("cached features -> /kaggle/working/features_64x24_3plane.npz", flush=True)

Y_derived = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in keep])
Y_gold = np.full((len(keep), len(LABELS)), -1, dtype=int)
Y_gold[gold_mask] = gold_df.loc[keep[gold_mask], LABELS].values.astype(int)
print(f"gold studies present in feature matrix: {gold_mask.sum()}", flush=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def run(pool_size, seed):
    """Out-of-fold predictions for a pool of `pool_size` studies that always
    contains all 58 gold studies, so both pools are scored on the same 58."""
    rng = np.random.default_rng(seed)
    gold_idx = np.where(gold_mask)[0]
    other = rng.permutation(np.where(~gold_mask)[0])
    take = max(pool_size - len(gold_idx), 0)
    rows = np.concatenate([gold_idx, other[:take]])

    Xc = X[rows]
    mean, scale = Xc.mean(0), Xc.std(0) + 1e-6
    Z = PCA(n_components=100, random_state=seed,
            svd_solver="randomized").fit_transform((Xc - mean) / scale)

    derived_aucs, gold_aucs = [], []
    is_gold = gold_mask[rows]
    for j in range(len(LABELS)):
        yd = Y_derived[rows, j]
        if yd.sum() < 25 or (1 - yd).sum() < 25:
            continue
        oof = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05),
                                Z, yd, cv=4, method="predict_proba")[:, 1]
        derived_aucs.append(roc_auc_score(yd, oof))
        yg = Y_gold[rows][is_gold, j]
        if len(np.unique(yg)) == 2:
            gold_aucs.append(roc_auc_score(yg, oof[is_gold]))
    return float(np.mean(derived_aucs)), float(np.mean(gold_aucs))


print("\n" + "=" * 68)
print("Does more data improve the labels' story, or the truth?")
print("=" * 68)
print(f"{'pool':<10}{'vs DERIVED labels':>22}{'vs GOLD labels (58)':>24}")
results = {}
for pool in (1500, len(X)):
    d = [run(pool, s) for s in SEEDS]
    dm = np.mean([x[0] for x in d])
    gm = np.mean([x[1] for x in d])
    gs = np.std([x[1] for x in d])
    results[pool] = (dm, gm)
    print(f"{pool:<10}{dm:>22.4f}{gm:>18.4f} ±{gs:.3f}", flush=True)

d1500, g1500 = results[1500]
dall, gall = results[len(X)]
print(f"\n{'change':<10}{dall-d1500:>+22.4f}{gall-g1500:>+24.4f}")
print("\nIf DERIVED climbed and GOLD did not, CV has been measuring agreement")
print("with our own extractor, and label quality now outranks image work.")
print("DONE", flush=True)
