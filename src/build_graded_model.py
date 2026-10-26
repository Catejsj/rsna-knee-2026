"""Fit the graded-target head on cached embeddings and write a model .npz.

`tune_head.py` measured, on the 58 radiologist-labeled studies:

    target      derived CV    gold(58)
    binary        0.7892       0.6534   <- what v5 submitted
    weighted      0.7878       0.6634
    graded        0.7659       0.6640

Graded targets *lower* derived CV and *raise* gold. That is the expected shape,
not a contradiction: derived CV scores agreement with the binarized labels, so
training on those labels optimizes that metric directly. Gold is the honest
measure and prefers the graded target, as do all the top public notebooks,
on the reasoning that AUC reads only order, so a severity score carries more
information than a 0/1.

Ridge regresses the graded confidence directly. Its output is then passed
through the same linear->sigmoid path the submission kernel already uses;
sigmoid is monotonic, so it cannot change AUC and the existing inference code
needs no modification.

PCA width is set to 200 (gold's best, 0.6832) but honestly that choice sits
inside the noise of a 58-study set: the target change is the real hypothesis
being tested, and one submission is what settles it.

Usage:  .venv/bin/python src/build_graded_model.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from rsna_labels import DATA, DERIVED, LABELS, ROOT

EMB = os.path.join(ROOT, "kaggle", "dinov2_encode", "out",
                   "dinov2_embeddings.npz")
OUT_DIR = os.path.join(ROOT, "kaggle", "dinov2_model_upload")
N_PCA = 200
ALPHA = 20.0          # Ridge alpha; regularization was flat across 0.01-1.0 C

d = np.load(EMB, allow_pickle=True)
X, ids = d["X"].astype(np.float32), d["study_ids"]
lab = pd.read_csv(os.path.join(DERIVED, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
train_csv = pd.read_csv(os.path.join(DATA, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

G = lab.loc[ids, LABELS].values.astype(np.float32)
B = (G >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[ids[gold_mask], LABELS].values.astype(int)

mean, scale = X.mean(0), X.std(0) + 1e-6
pca = PCA(n_components=N_PCA, random_state=0,
          svd_solver="randomized").fit((X - mean) / scale)
Z = pca.transform((X - mean) / scale)
print(f"embeddings {X.shape} -> PCA {Z.shape} "
      f"(explained var {pca.explained_variance_ratio_.sum():.2f})")

coefs, intercepts, prior = [], [], []
d_aucs, g_aucs = [], []
print(f"\n{'label':<18}{'n+':>6}{'derived':>10}{'gold':>9}")
for j, l in enumerate(LABELS):
    yb, yg_t = B[:, j], G[:, j]
    prior.append(float(yb.mean()))
    if yb.sum() < 25 or (1 - yb).sum() < 25:
        coefs.append(np.zeros(Z.shape[1]))
        intercepts.append(0.0)
        print(f"{l:<18}{yb.sum():>6}{'skip':>10}")
        continue
    m = Ridge(alpha=ALPHA).fit(Z, yg_t)
    oof = cross_val_predict(Ridge(alpha=ALPHA), Z, yg_t, cv=4)
    da = roc_auc_score(yb, oof)
    d_aucs.append(da)
    coefs.append(m.coef_)
    intercepts.append(float(m.intercept_))
    yg = Y_gold[gold_mask, j]
    ga = roc_auc_score(yg, oof[gold_mask]) if len(np.unique(yg)) == 2 else float("nan")
    if ga == ga:
        g_aucs.append(ga)
    print(f"{l:<18}{yb.sum():>6}{da:>10.3f}{ga:>9.3f}")

print(f"\nMACRO derived {np.mean(d_aucs):.4f}   MACRO gold {np.mean(g_aucs):.4f}")
print("v5 submitted (binary/PCA-100): derived 0.7892, gold 0.6534, LB 0.713")

# col_mean carries over unchanged: it is the training-set imputation for a
# plane a study lacks, and the embeddings were built with it.
os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(
    os.path.join(OUT_DIR, "dinov2_model.npz"),
    mean=mean, scale=scale, col_mean=d["col_mean"],
    pca_mean=pca.mean_, pca_components=pca.components_,
    coefs=np.array(coefs), intercepts=np.array(intercepts),
    prior=np.array(prior), labels=np.array(LABELS), planes=d["planes"],
    size=int(d["size"]), slices=int(d["slices"]), dim=int(d["dim"]),
    per_study_seconds=6.81, cv_macro_auc=float(np.mean(d_aucs)),
    head="ridge_graded", n_pca=N_PCA, alpha=ALPHA,
)
print(f"\nwrote {OUT_DIR}/dinov2_model.npz")
