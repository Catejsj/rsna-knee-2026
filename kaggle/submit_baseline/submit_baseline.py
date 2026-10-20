"""Tier-1 submission: hidden scans -> submission.csv. Internet OFF.

v3 — matches train_baseline v3: all three planes, 64px x 24 slices, PCA-100 +
per-label logistic regression. The model file carries its own config (size,
slices, planes, imputation means), so this script cannot disagree with how the
weights were fitted; nothing about the representation is restated here.

Feature code is imported from the shared rsna_features module — the identical
module the training kernel used. Both previously carried their own copy, which
is the failure this guards against: a drifted copy applies weights learned on
one representation to a different one, produces plausible numbers, and errors
nowhere.

Contract:
  * one row per StudyInstanceUID in test.csv, ALWAYS -- a study whose DICOMs
    fail to decode falls back to the training prior rather than vanishing
  * finish inside 9 h for ~1300 studies (measured: ~21 min)
  * no internet

The visible test set is 3 studies; at scoring time it is replaced with ~1300,
so nothing here is hard-coded to the sample.
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

COMP = os.path.dirname(glob.glob("/kaggle/input/**/test_series.csv", recursive=True)[0])
MODEL_PATH = glob.glob("/kaggle/input/**/baseline_model.npz", recursive=True)[0]
print(f"competition: {COMP}\nmodel: {MODEL_PATH}", flush=True)

M = np.load(MODEL_PATH, allow_pickle=True)
LABELS = [str(x) for x in M["labels"]]
SIZE = int(M["size"])
SLICES = int(M["slices"])
PLANES = tuple(str(p) for p in M["planes"])
PRIOR = M["prior"]
COL_MEAN = M["col_mean"]
print(f"model: {SIZE}px x {SLICES} slices, planes={'+'.join(PLANES)}, "
      f"train CV {float(M['cv_macro_auc']):.4f}", flush=True)


def predict(vec):
    # A plane this study lacks is NaN. Impute with the TRAINING column mean --
    # never a mean recomputed on the test set, which would leak and would also
    # differ from what the weights were fitted against.
    vec = np.where(np.isnan(vec), COL_MEAN, vec)
    z = (vec - M["mean"]) / M["scale"]
    z = (z - M["pca_mean"]) @ M["pca_components"].T
    logit = z @ M["coefs"].T + M["intercepts"]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


test = pd.read_csv(os.path.join(COMP, "test.csv"))
ser = pd.read_csv(os.path.join(COMP, "test_series.csv"))
root = os.path.join(COMP, "test_series")
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
print(f"test studies: {len(test)}", flush=True)

rows, n_fallback, plane_hits = [], 0, []
t0 = time.time()
for i, uid in enumerate(test.StudyInstanceUID):
    probs = None
    if uid in by_study:
        vec, found = F.study_features_multiplane_geo(
            root, uid, by_study[uid], planes=PLANES,
            n_slices=SLICES, size=SIZE)
        if found > 0:
            plane_hits.append(found)
            probs = predict(vec)
    if probs is None:
        probs = PRIOR
        n_fallback += 1
    rows.append([uid] + [float(v) for v in probs])
    if (i + 1) % 100 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(test)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(test)-i-1)/60:.1f} min", flush=True)

el = time.time() - t0
sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
sub.to_csv("/kaggle/working/submission.csv", index=False)

print(f"\nwrote submission.csv shape={sub.shape}", flush=True)
print(f"fallback-to-prior: {n_fallback}/{len(test)}", flush=True)
if plane_hits:
    print(f"planes found per study: mean {np.mean(plane_hits):.2f} of {len(PLANES)}",
          flush=True)
print(f"total {el/60:.1f} min = {el/3600:.2f} h of the 9 h budget "
      f"({el/max(len(test),1):.2f} s/study)", flush=True)
print(sub.head().to_string(), flush=True)
print("DONE", flush=True)
