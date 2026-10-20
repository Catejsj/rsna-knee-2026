"""Tier 1: exploit what we already have, one dimension at a time.

Three assets the pipeline currently throws away:
  A. 2907 labeled studies -- we train on 1500 of 4407
  B. ~20000 series -- we read one plane per study, there are three
  C. the head -- a straight line through 100 PCA components

Nothing new is invented here. The point is to find out how much of the gap to
the 0.809 public baseline is just under-use of the data we were given, before
spending effort on an encoder.

Method, same as the step-1 sweep: change ONE thing at a time against a fixed
reference, everything else frozen (64px, 24 slices, rules_v3 labels, 4-fold CV,
macro AUC). DICOMs are read once at the richest setting -- all three planes --
and every cheaper config is a column slice of that one matrix.

Missing planes are NaN, then imputed with the column mean. Zeros would be a
real value (a black image) and the model would learn from it.

Runs on CPU: decode is CPU-bound and this does not touch the GPU quota.
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
BLOCK = SIZE * SIZE                      # one plane's feature width
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])
print(f"competition: {COMP}\nderived: {DERIV}", flush=True)

ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies available: {len(studies)}  (baseline trained on 1500)", flush=True)

X, keep, plane_counts = [], [], []
t0 = time.time()
for i, uid in enumerate(studies):
    vec, found = F.study_features_multiplane(
        root, uid, by_study[uid], planes=PLANES, n_slices=SLICES, size=SIZE)
    if found == 0:
        continue
    X.append(vec)
    keep.append(uid)
    plane_counts.append(found)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(X)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

X = np.array(X, dtype=np.float32)
Y = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
elapsed = time.time() - t0
print(f"\nfeature matrix {X.shape} in {elapsed/60:.1f} min "
      f"({elapsed/len(studies):.2f} s/study)", flush=True)
print(f"planes per study: {np.bincount(plane_counts, minlength=4)[1:]} (1,2,3)", flush=True)
print(f"=> 1300 test studies would take {elapsed/len(studies)*1300/3600:.2f} h "
      f"of the 9 h budget", flush=True)

# column-mean imputation for planes a study does not have
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def evaluate(rows, planes_used, head, n_pca=100, seed=0):
    """macro AUC for one config. `rows` selects studies, `planes_used` selects
    which plane blocks of the feature matrix to keep."""
    cols = np.concatenate([np.arange(PLANES.index(p) * BLOCK,
                                     (PLANES.index(p) + 1) * BLOCK)
                           for p in planes_used])
    Xc = X[np.ix_(rows, cols)]
    Yc = Y[rows]
    mean, scale = Xc.mean(0), Xc.std(0) + 1e-6
    Z = PCA(n_components=min(n_pca, Xc.shape[0] - 1, Xc.shape[1]),
            random_state=seed, svd_solver="randomized").fit_transform((Xc - mean) / scale)
    aucs = []
    for j in range(len(LABELS)):
        y = Yc[:, j]
        if y.sum() < 25 or (1 - y).sum() < 25:
            continue
        if head == "logistic":
            clf = LogisticRegression(max_iter=3000, C=0.05)
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_depth=4,
                early_stopping=True, random_state=seed)
        p = cross_val_predict(clf, Z, y, cv=4, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs))


rng = np.random.default_rng(0)
order = rng.permutation(len(X))
N = len(X)

print("\n" + "=" * 74)
print("A. TRAINING SET SIZE   (sagittal only, PCA-100 + logistic)")
print("=" * 74)
print(f"{'studies':<14}{'macro AUC':>12}")
best_n = None
for n in (1500, 3000, N):
    score = evaluate(order[:n], ("Sagittal",), "logistic")
    print(f"{n:<14}{score:>12.4f}", flush=True)
    best_n = n if best_n is None else best_n
rows_all = order[:N]

print("\n" + "=" * 74)
print("B. PLANES   (all studies, PCA-100 + logistic)")
print("=" * 74)
print(f"{'planes':<26}{'dims':>8}{'macro AUC':>12}")
plane_sets = [("Sagittal",), ("Sagittal", "Coronal"), PLANES]
plane_scores = {}
for ps in plane_sets:
    score = evaluate(rows_all, ps, "logistic")
    plane_scores[ps] = score
    print(f"{'+'.join(p[:3] for p in ps):<26}{len(ps)*BLOCK:>8}{score:>12.4f}", flush=True)
best_planes = max(plane_scores, key=plane_scores.get)

print("\n" + "=" * 74)
print(f"C. HEAD   (all studies, planes = {'+'.join(p[:3] for p in best_planes)})")
print("=" * 74)
print(f"{'head':<26}{'macro AUC':>12}")
for name, head, npca in (("logistic PCA-100", "logistic", 100),
                         ("logistic PCA-300", "logistic", 300),
                         ("hist gradient boost", "hgb", 100)):
    score = evaluate(rows_all, best_planes, head, n_pca=npca)
    print(f"{name:<26}{score:>12.4f}", flush=True)

print("\nreference: submitted model (1500 studies, sagittal, PCA-100 logistic) "
      "CV 0.6504 / LB 0.628", flush=True)
print("DONE", flush=True)
