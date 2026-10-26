"""Does topology add anything on top of DINOv2? The ablation, run locally.

Three arms per label: embeddings alone, topology alone, and both. Scored
against derived labels (a training diagnostic only) and against the 58
radiologist-labeled studies, which is the ruler that has actually tracked the
leaderboard: it called the graded-target change to within 0.001.

Two fusion strategies, because they are not equivalent:

  joint   -- standardize, concatenate (2304 + 384), one PCA over everything.
             Embeddings outnumber topology 6:1, so a joint PCA can simply
             ignore the topology directions.
  blocked -- PCA each block separately (200 + 64), then concatenate. This
             guarantees topology a fixed share of the representation.

If topology helps at all, blocked should show it more clearly. If blocked helps
and joint does not, the conclusion is about fusion, not about topology.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from rsna_labels import DATA, DERIVED, LABELS, ROOT

EMB = os.path.join(ROOT, "kaggle", "dinov2_encode", "out", "dinov2_embeddings.npz")
TOP = os.path.join(ROOT, "kaggle", "topology_features", "out", "topology_features.npz")
ALPHA = 20.0

e = np.load(EMB, allow_pickle=True)
t = np.load(TOP, allow_pickle=True)
emb = pd.DataFrame(e["X"], index=e["study_ids"])
top = pd.DataFrame(t["X"], index=t["study_ids"])
shared = emb.index.intersection(top.index)
emb, top = emb.loc[shared], top.loc[shared]
print(f"aligned {len(shared)} studies | embeddings {emb.shape[1]} "
      f"| topology {top.shape[1]}")

lab = pd.read_csv(os.path.join(DERIVED, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
train_csv = pd.read_csv(os.path.join(DATA, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

G = lab.loc[shared, LABELS].values.astype(np.float32)
B = (G >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in shared])
Y_gold = np.zeros((len(shared), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[np.asarray(shared)[gold_mask], LABELS].values.astype(int)
print(f"gold studies: {gold_mask.sum()}\n")


def std(a):
    a = np.asarray(a, dtype=np.float32)
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


def pca(a, n):
    return PCA(n_components=n, random_state=0,
               svd_solver="randomized").fit_transform(a)


E, T = std(emb.values), std(top.values)
ARMS = {
    "embeddings only (200)": pca(E, 200),
    "topology only (64)": pca(T, 64),
    "joint PCA-200": pca(np.hstack([E, T]), 200),
    "blocked 200+64": np.hstack([pca(E, 200), pca(T, 64)]),
}


def evaluate(Z):
    derived, gold, per_gold = [], [], {}
    for j, l in enumerate(LABELS):
        if B[:, j].sum() < 25 or (1 - B[:, j]).sum() < 25:
            continue
        oof = cross_val_predict(Ridge(alpha=ALPHA), Z, G[:, j], cv=4)
        derived.append(roc_auc_score(B[:, j], oof))
        yg = Y_gold[gold_mask, j]
        if len(np.unique(yg)) == 2:
            g = roc_auc_score(yg, oof[gold_mask])
            gold.append(g)
            per_gold[l] = g
    return float(np.mean(derived)), float(np.mean(gold)), per_gold


print("=" * 74)
print(f"{'arm':<26}{'derived':>11}{'gold (58)':>12}{'vs emb-only':>15}")
print("=" * 74)
results = {}
for name, Z in ARMS.items():
    d, g, per = evaluate(Z)
    results[name] = (d, g, per)
    base = results.get("embeddings only (200)", (0, 0, {}))[1]
    delta = "" if name == "embeddings only (200)" else f"{g - base:+.4f}"
    print(f"{name:<26}{d:>11.4f}{g:>12.4f}{delta:>15}")

print("\n" + "=" * 74)
print("PER LABEL, gold: where does topology actually help?")
print("=" * 74)
be = results["embeddings only (200)"][2]
bt = results["topology only (64)"][2]
bb = results["blocked 200+64"][2]
print(f"{'label':<20}{'emb':>8}{'topo':>8}{'both':>8}{'delta':>9}")
for l in LABELS:
    if l in be and l in bb:
        print(f"{l:<20}{be[l]:>8.3f}{bt.get(l, float('nan')):>8.3f}"
              f"{bb[l]:>8.3f}{bb[l] - be[l]:>+9.3f}")

print("\nreference: v6 submitted (embeddings, graded, PCA-200) -> gold 0.6815, LB 0.742")
print("note: gold is 58 enriched studies; per-label swings under ~0.10 are noise.")
