"""Step 1, done properly: how much of the baseline's weakness was resolution,
and how much was slice count?

The baseline (LB 0.605) used 5 slices at 64x64 averaged into one vector. Those
two numbers were picked to answer a different question, never tuned. Before
adding an encoder we should know what the crude representation alone can do,
and where it saturates -- otherwise a later gain gets credited to the encoder
when it actually came from using more pixels.

Design: sweep ONE dimension at a time against a fixed baseline, not a grid.
  * resolution sweep : slices fixed at 24, size in {64, 96, 128, 160, 224}
  * slice sweep      : size fixed at 128, slices in {3, 5, 12, 24, 40}
  * aggregation      : mean vs mean+std across slices, at 128/24

Everything else is held identical to the submitted baseline: same 1500 studies,
same series choice, same PCA-100 + per-label logistic regression, same 4-fold
CV, same macro-AUC target. So the numbers are directly comparable to CV 0.635.

DICOMs are read ONCE at max resolution and max slice count; every config is
derived from that in memory. Re-reading per config would be 10x the IO.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import pydicom

N_STUDIES = 1500
MAX_SLICES = 40
MAX_SIZE = 224
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# (name, size, n_slices, aggregation)
CONFIGS = [
    ("res-064 x24", 64, 24, "mean"),
    ("res-096 x24", 96, 24, "mean"),
    ("res-128 x24", 128, 24, "mean"),
    ("res-160 x24", 160, 24, "mean"),
    ("res-224 x24", 224, 24, "mean"),
    ("slices-03 @128", 128, 3, "mean"),
    ("slices-05 @128", 128, 5, "mean"),
    ("slices-12 @128", 128, 12, "mean"),
    ("slices-40 @128", 128, 40, "mean"),
    ("meanstd @128x24", 128, 24, "meanstd"),
    ("BASELINE 64x5", 64, 5, "mean"),   # reproduces the submitted model
]

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])
print("competition:", COMP, flush=True)


def resize(a, n):
    h, w = a.shape
    yi = (np.arange(n) * h / n).astype(int)
    xi = (np.arange(n) * w / n).astype(int)
    return a[np.ix_(yi, xi)]


def load_slice(path):
    try:
        a = pydicom.dcmread(path).pixel_array.astype(np.float32)
    except Exception:
        return None
    if a.ndim != 2 or min(a.shape) < 16:
        return None
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return None
    return resize(np.clip((a - lo) / (hi - lo), 0, 1), MAX_SIZE)


def pick_series(df):
    for plane, fluid in (("Sagittal", 1), ("Coronal", 1), (None, None)):
        sub = df if plane is None else df[(df.Anatomical_Plane == plane)
                                          & (df.Fluid_Sensitive == fluid)]
        if len(sub):
            return sub.iloc[0]
    return None


def features(stack, size, n_slices, agg):
    """stack: list of MAX_SIZE x MAX_SIZE arrays, evenly spanning the series."""
    k = min(n_slices, len(stack))
    idx = np.linspace(0, len(stack) - 1, k).astype(int)
    sel = [resize(stack[i], size) if size != MAX_SIZE else stack[i] for i in idx]
    arr = np.stack(sel)
    if agg == "meanstd":
        return np.concatenate([arr.mean(0).ravel(), arr.std(0).ravel()])
    return arr.mean(0).ravel()


ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)][:N_STUDIES]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")

bank = {c[0]: [] for c in CONFIGS}
keep = []
t0 = time.time()
for i, s in enumerate(studies):
    row = pick_series(by_study[s])
    if row is None:
        continue
    d = os.path.join(root, s, row.SeriesInstanceUID)
    try:
        files = sorted(os.listdir(d))
    except Exception:
        continue
    if len(files) < 3:
        continue
    idx = np.linspace(0.15, 0.85, min(MAX_SLICES, len(files))) * (len(files) - 1)
    stack = [load_slice(os.path.join(d, files[int(k)])) for k in idx]
    stack = [x for x in stack if x is not None]
    if len(stack) < 3:
        continue
    for name, size, ns, agg in CONFIGS:
        bank[name].append(features(stack, size, ns, agg).astype(np.float32))
    keep.append(s)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(studies)} ok={len(keep)} "
              f"{(time.time()-t0)/(i+1):.2f} s/study", flush=True)

print(f"\nloaded {len(keep)} studies in {(time.time()-t0)/60:.1f} min", flush=True)

# ------------------------------------------------------------------ evaluate
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

Y = (lab.loc[keep, LABELS].values >= 0.5).astype(int)


def macro_auc(X):
    X = np.asarray(X, dtype=np.float32)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z = PCA(n_components=100, random_state=0,
            svd_solver="randomized").fit_transform((X - mean) / scale)
    aucs, per = [], {}
    for j, l in enumerate(LABELS):
        y = Y[:, j]
        if y.sum() < 25 or (1 - y).sum() < 25:
            continue
        p = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05), Z, y,
                              cv=4, method="predict_proba")[:, 1]
        a = roc_auc_score(y, p)
        aucs.append(a)
        per[l] = a
    return float(np.mean(aucs)), per


print("\n" + "=" * 78)
print(f"{'config':<20}{'dims':>9}{'macro AUC':>12}{'vs baseline':>14}")
print("=" * 78)
results = {}
base = None
for name, size, ns, agg in CONFIGS:
    X = np.array(bank[name], dtype=np.float32)
    m, per = macro_auc(X)
    results[name] = (m, per)
    if name.startswith("BASELINE"):
        base = m
    delta = "" if base is None else f"{m - base:+.4f}"
    print(f"{name:<20}{X.shape[1]:>9}{m:>12.4f}{delta:>14}", flush=True)
    del X

print("\nper-label for the best config:")
best = max(results, key=lambda k: results[k][0])
print(f"  winner: {best}  macro {results[best][0]:.4f}")
for l, a in sorted(results[best][1].items(), key=lambda kv: -kv[1]):
    b = results["BASELINE 64x5"][1].get(l, float("nan"))
    print(f"  {l:<20}{a:.3f}   baseline {b:.3f}   {a-b:+.3f}")
print("\nDONE", flush=True)
