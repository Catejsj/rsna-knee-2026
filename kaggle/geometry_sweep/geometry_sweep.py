"""Does fixing the geometry fix the compartment labels?

Read of the top public notebooks: all of them normalize geometry before the
model sees anything, and we do none of it.

  * laterality -- they flip every knee to one side. We feed left and right in
    mirrored, so "medial" (the inner compartment) sits on opposite sides of the
    image for different patients.
  * physical scale -- they centre-crop a fixed 160 mm of anatomy. The acquired
    field of view here runs ~70-320 mm, so our fixed pixel resize shows the
    same knee at wildly different scales.
  * slice order -- they sort by position along the slice normal. We sort by
    filename, which is SOP UID order, i.e. arbitrary.

The prediction that makes this falsifiable: laterality should lift the five
compartment-specific labels (both menisci, both tibiofemoral OA, PF OA) and
leave the whole-joint ones (Effusion, Synovitis, Baker's) roughly alone. Our
gold scores currently split exactly that way -- Effusion 0.796 at the top,
Medial Meniscus 0.407 at the bottom -- which is what suggested the hypothesis.
If the split does not move, the explanation is wrong.

Sagittal only: the plane sweep showed the extra planes bought no real skill, so
this costs a third of the decode and isolates the variable that matters.

Scored against BOTH derived labels and the 58 gold, because derived-CV has
already been caught rewarding a change worth nothing.
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
MAX_READ = 32
SEEDS = (0, 1, 2)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
COMPARTMENT = ["Medial Meniscus", "Lateral Meniscus", "Medial OA",
               "Lateral OA", "PF OA"]
WHOLE_JOINT = ["Effusion", "Synovitis", "Baker's", "ACL", "MCL",
               "Contusion", "Fracture"]

# name -> (order, crop_mm, flip_to)
VARIANTS = [
    ("baseline (as submitted)", "filename", None, None),
    ("+ laterality flip", "filename", None, "R"),
    ("+ physical crop", "filename", F.CROP_MM, None),
    ("+ physical order", "physical", None, None),
    ("all three", "physical", F.CROP_MM, "R"),
]

COMP_DIR = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

ser = pd.read_csv(os.path.join(COMP_DIR, "train_series.csv"))
train_csv = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP_DIR, "train_series")

bank = {v[0]: [] for v in VARIANTS}
keep = []
tag_stats = {"pos": 0, "spacing": 0, "side": 0, "side_L": 0, "total": 0}
t0 = time.time()
for i, uid in enumerate(studies):
    row = F.pick_series_for_plane(by_study[uid], "Sagittal")
    if row is None:
        continue
    slices = F.load_series_with_geometry(
        os.path.join(root, uid, row.SeriesInstanceUID), max_slices=MAX_READ)
    if len(slices) < 3:
        continue

    tag_stats["total"] += 1
    tag_stats["pos"] += all(s["pos"] is not None for s in slices)
    tag_stats["spacing"] += all(s["spacing"] is not None for s in slices)
    sides = [s["side"] for s in slices if s["side"]]
    tag_stats["side"] += bool(sides)
    tag_stats["side_L"] += bool(sides) and sides[0] == "L"

    vecs = {}
    ok = True
    for name, order, crop, flip in VARIANTS:
        v = F.geometric_features(slices, n_slices=SLICES, size=SIZE,
                                 crop_mm=crop, order=order, flip_to=flip)
        if v is None:
            ok = False
            break
        vecs[name] = v
    if not ok:
        continue
    for name, v in vecs.items():
        bank[name].append(v.astype(np.float32))
    keep.append(uid)

    if (i + 1) % 400 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(keep)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

keep = np.array(keep)
n = tag_stats["total"] or 1
print(f"\nloaded {len(keep)} studies in {(time.time()-t0)/60:.1f} min", flush=True)
print("DICOM tag coverage:", flush=True)
print(f"  ImagePositionPatient/Orientation : {tag_stats['pos']/n*100:5.1f}%")
print(f"  PixelSpacing                     : {tag_stats['spacing']/n*100:5.1f}%")
print(f"  Laterality                       : {tag_stats['side']/n*100:5.1f}%"
      f"   (left knees: {tag_stats['side_L']/n*100:.1f}%)", flush=True)

Y_derived = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in keep])
Y_gold = np.zeros((len(keep), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[keep[gold_mask], LABELS].values.astype(int)
print(f"gold studies present: {gold_mask.sum()}", flush=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def evaluate(X, seed):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z = PCA(n_components=100, random_state=seed,
            svd_solver="randomized").fit_transform((X - mean) / scale)
    derived, gold = [], {}
    for j, name in enumerate(LABELS):
        yd = Y_derived[:, j]
        if yd.sum() < 25 or (1 - yd).sum() < 25:
            continue
        oof = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05),
                                Z, yd, cv=4, method="predict_proba")[:, 1]
        derived.append(roc_auc_score(yd, oof))
        yg = Y_gold[gold_mask, j]
        if len(np.unique(yg)) == 2:
            gold[name] = roc_auc_score(yg, oof[gold_mask])
    return float(np.mean(derived)), gold


print("\n" + "=" * 86)
print(f"{'variant':<26}{'DERIVED':>10}{'GOLD all':>10}{'GOLD compartment':>19}"
      f"{'GOLD whole-joint':>19}")
print("=" * 86)
results = {}
for name, *_ in VARIANTS:
    X = np.array(bank[name], dtype=np.float32)
    ds, golds = zip(*[evaluate(X, s) for s in SEEDS])
    g_all = np.mean([np.mean(list(g.values())) for g in golds])
    g_cmp = np.mean([np.mean([g[k] for k in COMPARTMENT if k in g]) for g in golds])
    g_who = np.mean([np.mean([g[k] for k in WHOLE_JOINT if k in g]) for g in golds])
    results[name] = (np.mean(ds), g_all, g_cmp, g_who, golds[0])
    print(f"{name:<26}{np.mean(ds):>10.4f}{g_all:>10.4f}{g_cmp:>19.4f}{g_who:>19.4f}",
          flush=True)
    del X

print("\nprediction under test: laterality lifts COMPARTMENT, leaves WHOLE-JOINT flat.")
base = results["baseline (as submitted)"]
best = results["all three"]
print(f"\ncompartment {base[2]:.4f} -> {best[2]:.4f}  ({best[2]-base[2]:+.4f})")
print(f"whole-joint {base[3]:.4f} -> {best[3]:.4f}  ({best[3]-base[3]:+.4f})")

print("\nper-label gold, baseline vs all three:")
print(f"{'label':<20}{'baseline':>10}{'fixed':>10}{'delta':>9}")
for l in LABELS:
    b, f_ = base[4].get(l), best[4].get(l)
    if b is not None and f_ is not None:
        mark = "  <-- compartment" if l in COMPARTMENT else ""
        print(f"{l:<20}{b:>10.3f}{f_:>10.3f}{f_-b:>+9.3f}{mark}")
print("\nDONE", flush=True)
