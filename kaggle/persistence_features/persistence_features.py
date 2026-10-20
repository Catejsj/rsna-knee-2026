"""Asaad-style persistent homology features: ULBP landmarks -> Rips filtration.

Implements the method from Aras Asaad's DPhil thesis (Buckingham, 2020) 5.1-5.2
and its mammography application in 6.3, which is what the project brief named:
LBP-code every pixel, keep uniform codes grouped into G_lambda geometries, take
their (x,y) positions as a landmark cloud, filter by increasing distance, and
record B0/B1 at each threshold.

This is a *different filtration* from `topology.py`, not a refinement of it:

    topology.py  filters over image INTENSITY  -> what shapes exist per brightness
    this file    filters over DISTANCE between texture landmarks
                 -> how texture-salient points cluster spatially

Asaad's hypothesis is that disease distorts the natural spatial texture of
tissue. That is a claim about landmark geometry, which intensity filtration
cannot see, so the earlier negative result does not carry over.

Two deliberate deviations, both documented in src/persistence.py: B0/B1 come
from the Delaunay complex (exact for B0, since the Euclidean MST is a Delaunay
subgraph) rather than full Rips, and the 8 rotations per geometry are merged
rather than kept separate -- the thesis uses 800 dims per geometry, and every
capacity increase measured on this project has been punished.

Validated locally: 58 uniform LBP codes exactly as the thesis states; a ring
gives b1=1, two rings b1=2, a solid disk b1~0.
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
import rsna_persistence as P

SIZE = 224
SLICES = 24
PLANES = ("Sagittal",)
N_THRESHOLDS = 24
BLOCK = 2 * 3 * 2 * N_THRESHOLDS   # 3 geometries x (B0,B1) x thresholds x (mean,max)

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies: {len(studies)}  features/plane: {BLOCK}  total: {BLOCK*len(PLANES)}",
      flush=True)

X, keep = [], []
t0 = time.time()
for i, uid in enumerate(studies):
    blocks, found = [], 0
    for plane in PLANES:
        row = F.pick_series_for_plane(by_study[uid], plane)
        stack = None
        if row is not None:
            slices = F.load_series_with_geometry(
                os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
            if len(slices) >= 3:
                stack = F.geometric_slice_stack(slices, n_slices=SLICES, size=SIZE)
        if stack is None:
            blocks.append(np.full(BLOCK, np.nan, dtype=np.float32))
        else:
            blocks.append(P.stack_features(stack, n_thresholds=N_THRESHOLDS))
            found += 1
    if found == 0:
        continue
    X.append(np.concatenate(blocks).astype(np.float32))
    keep.append(uid)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(X)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

X = np.array(X, dtype=np.float32)
keep = np.array(keep)
elapsed = time.time() - t0
per_study = elapsed / max(len(studies), 1)
print(f"\npersistence features {X.shape} in {elapsed/60:.1f} min "
      f"({per_study:.2f} s/study)", flush=True)
print(f"=> 1300 test studies would add {per_study*1300/3600:.2f} h at inference",
      flush=True)

col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

np.savez_compressed("/kaggle/working/persistence_features.npz",
                    X=X, study_ids=keep, col_mean=col_mean,
                    size=SIZE, slices=SLICES, planes=np.array(PLANES),
                    n_thresholds=N_THRESHOLDS)
print("saved /kaggle/working/persistence_features.npz", flush=True)

# quick sanity: do these features carry ANY signal on their own? The real
# ablation runs locally against the cached embeddings and the 58 gold studies.
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

LABELS = [c for c in lab.columns if c != "Report"]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
G = lab.loc[keep, LABELS].values.astype(np.float32)
B = (G >= 0.5).astype(int)
mean, scale = X.mean(0), X.std(0) + 1e-6
Z = PCA(n_components=64, random_state=0,
        svd_solver="randomized").fit_transform((X - mean) / scale)
aucs = []
print(f"\n{'label':<18}{'derived AUC (Asaad persistence alone)':>30}")
for j, l in enumerate(LABELS):
    if B[:, j].sum() < 25 or (1 - B[:, j]).sum() < 25:
        continue
    oof = cross_val_predict(Ridge(alpha=20.0), Z, G[:, j], cv=4)
    a = roc_auc_score(B[:, j], oof)
    aucs.append(a)
    print(f"{l:<18}{a:>30.3f}")
print(f"\nMACRO (Asaad persistence alone): {np.mean(aucs):.4f}   "
      f"(DINOv2 alone was 0.7643 on the same metric)", flush=True)
print("DONE", flush=True)
