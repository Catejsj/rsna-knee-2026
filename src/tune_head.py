"""Head experiments on the cached DINOv2 embeddings. Runs locally in seconds.

Two questions, and they are not equally trustworthy.

**Targets (trustworthy).** `extract_rules.py` emits graded confidence: 0.90
strong evidence, 0.65 mentioned both ways, 0.25 never mentioned, 0.08 explicitly
denied, and training then does `>= 0.5` and throws the gradation away. Since
AUC reads only order, a graded target should carry more signal than a binary
one. Every top public notebook grades rather than thresholds. This is a change
to the *target*, not to model capacity, so derived-CV can be believed here.

**Capacity (NOT trustworthy).** PCA width and regularization strength are
exactly the kind of change derived-CV has been caught rewarding for nothing:
more capacity lets the model fit `rules_v3`'s systematic errors, which repeat
across CV folds. On pixels, CV preferred PCA-300 and the leaderboard disagreed.
So every capacity result is reported against the 58 gold studies as well, and
a capacity change is only worth submitting if BOTH move.

Usage:  .venv/bin/python src/tune_head.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from rsna_labels import DATA, DERIVED, LABELS, ROOT

EMB = os.path.join(ROOT, "kaggle", "dinov2_encode", "out",
                   "dinov2_embeddings.npz")
SEEDS = (0, 1, 2)

d = np.load(EMB, allow_pickle=True)
X_raw, ids = d["X"], d["study_ids"]

lab = pd.read_csv(os.path.join(DERIVED, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
train_csv = pd.read_csv(os.path.join(DATA, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

G = lab.loc[ids, LABELS].values.astype(np.float32)   # graded 0-1
B = (G >= 0.5).astype(int)                           # binary, as currently used
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

print(f"embeddings {X_raw.shape} | gold studies {gold_mask.sum()}")
print(f"graded target values in use: {sorted(np.unique(G))[:6]} ...\n")


def project(n_pca, seed):
    mean, scale = X_raw.mean(0), X_raw.std(0) + 1e-6
    Z = (X_raw - mean) / scale
    if n_pca:
        Z = PCA(n_components=n_pca, random_state=seed,
                svd_solver="randomized").fit_transform(Z)
    return Z


def score(Z, mode, C, seed):
    """Out-of-fold predictions -> (derived macro AUC, gold macro AUC)."""
    derived, gold = [], []
    for j in range(len(LABELS)):
        yb, yg_all = B[:, j], G[:, j]
        if yb.sum() < 25 or (1 - yb).sum() < 25:
            continue
        if mode == "binary":
            est = LogisticRegression(max_iter=3000, C=C)
            oof = cross_val_predict(est, Z, yb, cv=4,
                                    method="predict_proba")[:, 1]
        elif mode == "weighted":
            # binary label, but weight by how sure the extractor was; an
            # ambiguous 0.65/0.25 counts less than a confident 0.90/0.08
            w = np.abs(yg_all - 0.5) * 2
            est = LogisticRegression(max_iter=3000, C=C)
            oof = np.zeros(len(yb))
            from sklearn.model_selection import StratifiedKFold
            for tr, te in StratifiedKFold(4, shuffle=True,
                                          random_state=seed).split(Z, yb):
                m = LogisticRegression(max_iter=3000, C=C)
                m.fit(Z[tr], yb[tr], sample_weight=w[tr])
                oof[te] = m.predict_proba(Z[te])[:, 1]
        else:  # graded: regress the confidence directly, AUC only needs order
            est = Ridge(alpha=1.0 / max(C, 1e-6))
            oof = cross_val_predict(est, Z, yg_all, cv=4)
        derived.append(roc_auc_score(yb, oof))
        yg = Y_gold[gold_mask, j]
        if len(np.unique(yg)) == 2:
            gold.append(roc_auc_score(yg, oof[gold_mask]))
    return float(np.mean(derived)), float(np.mean(gold))


print("=" * 78)
print("A. TARGET ENCODING  (PCA-100, C=0.05: the submitted config)")
print("=" * 78)
print(f"{'target':<26}{'derived CV':>13}{'gold (58)':>12}")
Z100 = project(100, 0)
for mode in ("binary", "weighted", "graded"):
    ds, gs = zip(*[score(project(100, s), mode, 0.05, s) for s in SEEDS])
    tag = "  <-- current" if mode == "binary" else ""
    print(f"{mode:<26}{np.mean(ds):>13.4f}{np.mean(gs):>12.4f}{tag}")

print()
print("=" * 78)
print("B. CAPACITY  (best target from A): derived-CV is NOT reliable here")
print("=" * 78)
print(f"{'config':<26}{'derived CV':>13}{'gold (58)':>12}")
for n_pca in (50, 100, 200, 400, None):
    Z = project(n_pca, 0)
    ds, gs = zip(*[score(project(n_pca, s), "graded", 0.05, s) for s in SEEDS])
    name = f"PCA-{n_pca}" if n_pca else "no PCA (2304)"
    print(f"{name:<26}{np.mean(ds):>13.4f}{np.mean(gs):>12.4f}")

print()
print("=" * 78)
print("C. REGULARIZATION  (PCA-100, graded)")
print("=" * 78)
print(f"{'C':<26}{'derived CV':>13}{'gold (58)':>12}")
for C in (0.01, 0.05, 0.2, 1.0):
    ds, gs = zip(*[score(project(100, s), "graded", C, s) for s in SEEDS])
    print(f"{C:<26}{np.mean(ds):>13.4f}{np.mean(gs):>12.4f}")

print("\nreference: submitted v5 = binary / PCA-100 / C=0.05 -> CV 0.7892, LB 0.713")
print("rule: submit a capacity change only if derived AND gold both improve.")
