"""Retrieval-based classification: the training set as a searchable database.

The idea (2026-10-05), stated in IR terms: index every training study as
a vector, treat a test study as a QUERY, retrieve its nearest training studies
and predict from their labels -- a vector-space model over images, with a
probabilistic (kernel-weighted) vote. Persistent homology / topology serve as an
alternative or additional notion of "similar".

Gate test on cached descriptors, no GPU, gold used only to score:

    database = 4349 training studies, labels = pilkwang teacher (soft)
    queries  = the 58 gold studies

Vote: weights = exp(cosine_sim / tau) over the top-k, prediction = weighted mean
of the neighbours' graded teacher labels. Baseline: logistic regression on the
SAME features, so the comparison isolates retrieval vs a parametric readout.

What would make this worth a GPU run: kNN close to the parametric head on the
same features. Then the real version uses FINE-TUNED embeddings, and the
question becomes whether a non-parametric member is decorrelated enough from
the trained heads to help the ensemble.

Usage:  .venv/bin/python src/retrieval_knn.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def load(rel):
    d = np.load(os.path.join(R, rel), allow_pickle=True)
    return d["X"].astype(np.float64), d["study_ids"]


EMB, ids = load("kaggle/dinov2_encode/out/dinov2_embeddings.npz")
TOP, i2 = load("kaggle/topology_features/out/topology_features.npz")
PER, i3 = load("kaggle/persistence_features/out/persistence_features.npz")
assert np.array_equal(ids, i2) and np.array_equal(ids, i3), "study order differs"

train = pd.read_csv(os.path.join(R, "data/train.csv"), usecols=["StudyInstanceUID"] + LABELS)
gold = train[train["ACL"].notna()].set_index("StudyInstanceUID")
gm = np.array([u in set(gold.index) for u in ids])
Yg = gold.loc[ids[gm], LABELS].values.astype(int)
T = pd.read_csv(os.path.join(R, "derived/report_labels_v2.csv")).set_index("StudyInstanceUID")
fb = pd.read_csv(os.path.join(R, "derived/labels_rules_v5.csv")).set_index("StudyInstanceUID")
miss = [u for u in ids if u not in T.index]
T = pd.concat([T[LABELS], fb.loc[miss, LABELS]]) if miss else T[LABELS]
G = T.loc[ids].values.astype(float)
db, q = np.where(~gm)[0], np.where(gm)[0]
print(f"database {len(db)} training studies | queries {len(q)} gold studies")


def macro(P):
    out = []
    for j in range(len(LABELS)):
        y = Yg[:, j]
        if len(np.unique(y)) == 2:
            r = rankdata(P[:, j]); n1 = y.sum(); n0 = len(y) - n1
            out.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


def embed(X, n_pca=100):
    """Standardise -> PCA -> L2-normalise, fitted on the DATABASE only."""
    X = np.nan_to_num(X)
    sc = StandardScaler().fit(X[db])
    Z = sc.transform(X)
    p = PCA(n_components=min(n_pca, Z.shape[1]), random_state=0).fit(Z[db])
    Z = p.transform(Z)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def knn(Z, k, tau):
    S = Z[q] @ Z[db].T                                  # cosine similarity
    top = np.argsort(-S, axis=1)[:, :k]
    s = np.take_along_axis(S, top, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qkl->ql", w, G[db][top])


def logreg(Z):
    P = np.zeros((len(q), len(LABELS)))
    for j in range(len(LABELS)):
        y = (G[db, j] >= 0.5).astype(int)
        P[:, j] = LogisticRegression(max_iter=3000, C=1.0).fit(Z[db], y).predict_proba(Z[q])[:, 1]
    return P


FEATS = {
    "DINOv2 frozen": EMB,
    "topology (Betti)": TOP,
    "persistence (PH)": PER,
    "DINOv2 + topology": np.hstack([EMB, TOP]),
    "DINOv2 + persistence": np.hstack([EMB, PER]),
}
print(f"\n  {'features':<22}{'logreg':>8}" + "".join(f"{f'k={k}':>8}" for k in (10, 25, 50, 100, 200)))
best = {}
for name, X in FEATS.items():
    Z = embed(X)
    row = f"  {name:<22}{macro(logreg(Z)):>8.4f}"
    for k in (10, 25, 50, 100, 200):
        a = max(macro(knn(Z, k, tau)) for tau in (0.02, 0.05, 0.2))
        row += f"{a:>8.4f}"
        best[name] = max(best.get(name, 0), a)
    print(row, flush=True)

print("\n  kNN cells take the best of tau in {0.02, 0.05, 0.2} -- a mild optimism in")
print("  kNN's favour, stated so a near-tie is not over-read.")
print("  For scale: the fine-tuned members score 0.84-0.87 on the same 58 studies.")
