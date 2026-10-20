"""Train the model. Features come from the shared rsna_features module.

v4 — v3 plus geometry normalization: slices ordered along the slice normal
(was filename order, i.e. arbitrary), centre-cropped to a fixed 160 mm of
anatomy (the acquired field of view ranges 70-320 mm), and every knee mirrored
to face the same way.

Laterality comes from the DICOM tag where present (49.7% of studies) and from
patient coordinates otherwise — the sign of the image-centre x in LPS space.
That inference agrees with the tag on 97.1% of the 2189 studies that carry one,
and is available on 100%. Flipping only the tagged half made things *worse*
(compartment gold 0.571 -> 0.534) because half-normalized is two conventions
rather than one; full coverage is what makes the fix coherent.

Everything else held at the v3 config so the leaderboard delta attributes to
geometry alone: all 4407 studies, three planes, 64px x 24 slices, PCA-100 +
per-label logistic regression. v3 scored CV 0.7116 / LB 0.633.

History: v1 (CV 0.6441, LB 0.605), v2 (CV 0.6504, LB 0.628),
v3 (CV 0.7116, LB 0.633).

NOTE: derived-label CV has been caught rewarding a change worth nothing on the
leaderboard (three planes: derived +0.032, gold +0.000). Judge this by the
submission, not by the CV number printed below.

Two sweep findings are load-bearing here and should not be "improved" without
re-measuring: PCA-300 and gradient boosting both scored *below* PCA-100 +
logistic, and 224px scored below 64px. This pipeline is data-limited, so extra
capacity overfits. Add information, not parameters.

`rsna_features` is the dataset catejsj/rsna-features-code, generated from
src/features.py by tools/sync_utility.py. The submission kernel imports the
identical module, so train and inference cannot drift.

Model saves as plain numpy arrays — no sklearn-version dependency downstream.
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
N_PCA = 100
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])
print(f"config: {SIZE}px x {SLICES} slices, planes={'+'.join(PLANES)}", flush=True)

ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies: {len(studies)}", flush=True)

X, keep = [], []
t0 = time.time()
for i, uid in enumerate(studies):
    vec, found = F.study_features_multiplane_geo(
        root, uid, by_study[uid], planes=PLANES, n_slices=SLICES, size=SIZE)
    if found == 0:
        continue
    X.append(vec)
    keep.append(uid)
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(X)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

elapsed = time.time() - t0
per_study = elapsed / max(len(studies), 1)
X = np.array(X, dtype=np.float32)
print(f"\nfeatures {X.shape} in {elapsed/60:.1f} min", flush=True)
print(f"THROUGHPUT {per_study:.2f} s/study => "
      f"{per_study*1300/3600:.2f} h for 1300 test studies (budget 9 h)", flush=True)

# A plane a study lacks is NaN; impute with the column mean. Inference must use
# THIS mean, not one recomputed on the test set, so it is saved with the model.
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

mean = X.mean(0)
scale = X.std(0) + 1e-6
pca = PCA(n_components=N_PCA, random_state=0,
          svd_solver="randomized").fit((X - mean) / scale)
Z = pca.transform((X - mean) / scale)
print(f"PCA {Z.shape}, explained var {pca.explained_variance_ratio_.sum():.2f}", flush=True)

Y = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
coefs, intercepts, prior, aucs = [], [], [], []
print(f"\n{'label':<18}{'n+':>6}{'cv AUC':>9}")
for j, l in enumerate(LABELS):
    y = Y[:, j]
    p = float(y.mean())
    prior.append(p)
    if y.sum() < 25 or (1 - y).sum() < 25:
        coefs.append(np.zeros(Z.shape[1]))
        intercepts.append(np.log((p + 1e-6) / (1 - p + 1e-6)))
        print(f"{l:<18}{y.sum():>6}{'skip':>9}")
        continue
    model = LogisticRegression(max_iter=3000, C=0.05).fit(Z, y)
    cv = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05), Z, y,
                           cv=4, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, cv)
    aucs.append(auc)
    coefs.append(model.coef_[0])
    intercepts.append(model.intercept_[0])
    print(f"{l:<18}{y.sum():>6}{auc:>9.3f}", flush=True)

print(f"\nMACRO CV AUC: {np.mean(aucs):.4f}   "
      f"(v3 same config without geometry: CV 0.7116 / LB 0.633)", flush=True)

np.savez_compressed(
    "/kaggle/working/baseline_model.npz",
    mean=mean, scale=scale, col_mean=col_mean,
    pca_mean=pca.mean_, pca_components=pca.components_,
    coefs=np.array(coefs), intercepts=np.array(intercepts),
    prior=np.array(prior), labels=np.array(LABELS),
    planes=np.array(PLANES),
    size=SIZE, slices=SLICES, per_study_seconds=per_study,
    crop_mm=F.CROP_MM, slice_order="physical", flip_to="R",
    cv_macro_auc=float(np.mean(aucs)),
)
print("saved /kaggle/working/baseline_model.npz\nDONE", flush=True)
