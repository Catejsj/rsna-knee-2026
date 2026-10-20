"""Re-test topology and persistence under the LLM labels. Local, ~1 min.

Both were measured in the **rules-era**, when the teacher was a regex extractor
scoring ~0.77 on gold. Two things have changed since:

  * the teacher is now the public LLM labels at 0.870 gold, and swapping to it
    was worth +0.043 LB on its own;
  * "add information, not parameters" turned out to be **conditional on a bad
    teacher** -- v17B's layer3 unfreeze is pure added capacity and won once the
    labels were good. Persistence was deliberately cut down under that rule
    (rotations merged, 600 landmarks, vs the thesis' ~800 dims per geometry),
    so the reduction that seemed obviously right may not be.

Nothing here needs recomputing: both descriptor sets are cached
(`kaggle/topology_features/out`, `kaggle/persistence_features/out`) alongside
the frozen DINOv2 embeddings. This just re-runs the head against a better
teacher and reports on gold-58.

Original numbers, for comparison:

    embeddings only              gold 0.6811
    + topology (blocked)         gold 0.6862   +0.005, inside noise
    topology alone               gold 0.6416
    persistence alone            gold 0.5627   worst of everything

Usage:  .venv/bin/python src/retest_tda.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def load(rel, key_hint):
    p = os.path.join(R, rel)
    if not os.path.exists(p):
        print(f"  missing: {rel}")
        return None, None
    d = np.load(p, allow_pickle=True)
    keys = list(d.keys())
    fk = next((k for k in keys if k not in ("study_ids", "ids")), keys[0])
    ik = next((k for k in keys if k in ("study_ids", "ids")), None)
    print(f"  {rel.split('/')[-1]:<28} {fk} {d[fk].shape}")
    return d[fk], (d[ik] if ik else None)


print("loading cached descriptors")
EMB, emb_ids = load("kaggle/dinov2_encode/out/dinov2_embeddings.npz", "emb")
TOP, top_ids = load("kaggle/topology_features/out/topology_features.npz", "topo")
PER, per_ids = load("kaggle/persistence_features/out/persistence_features.npz", "pers")

ids = emb_ids if emb_ids is not None else top_ids
assert ids is not None, "no study ids in any cache"
ids = np.asarray(ids)
for name, other in [("topology", top_ids), ("persistence", per_ids)]:
    if other is not None:
        assert np.array_equal(np.asarray(other), ids), \
            f"{name} study order differs -- features would be paired with the wrong knee"
print(f"  study order matches across caches ({len(ids)} studies)")

train = pd.read_csv(os.path.join(R, "data/train.csv"))
gold_df = train[train["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
gm = np.array([u in gold_ids for u in ids])
Yg = np.zeros((len(ids), len(LABELS)), dtype=int)
Yg[gm] = gold_df.loc[ids[gm], LABELS].values.astype(int)

# teachers: the rules-era one it was originally measured against, and the LLM
# labels that replaced it
TEACHERS = {}
rules = os.path.join(R, "derived/labels_rules_v3.csv")
if os.path.exists(rules):
    TEACHERS["rules_v3"] = pd.read_csv(rules).set_index("StudyInstanceUID")
for cand in (os.path.join(R, "derived/report_labels_v2.csv"),):
    if os.path.exists(cand):
        TEACHERS["LLM"] = pd.read_csv(cand).set_index("StudyInstanceUID")
        break
print(f"  teachers available: {list(TEACHERS)}")


def evaluate(X, teacher, n_pca=100):
    """Train per-label logistic on the teacher, score on the 58 gold studies."""
    have = [u for u in ids if u in teacher.index]
    keep = np.array([u in set(have) for u in ids])
    Xs = StandardScaler().fit_transform(np.nan_to_num(X[keep]))
    n_pca = min(n_pca, Xs.shape[1], Xs.shape[0] - 1)
    Z = PCA(n_components=n_pca, random_state=0).fit_transform(Xs)
    T = teacher.loc[ids[keep], LABELS].values.astype(float)
    g = gm[keep]
    out = []
    for j in range(len(LABELS)):
        y = (T[:, j] >= 0.5).astype(int)
        if len(np.unique(y)) < 2:
            continue
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Z, y)
        p = m.predict_proba(Z)[:, 1]
        yg = Yg[keep][g, j]
        if len(np.unique(yg)) == 2:
            out.append(roc_auc_score(yg, p[g]))
    return float(np.mean(out))


ARMS = {"embeddings": EMB, "topology": TOP, "persistence": PER}
ARMS = {k: v for k, v in ARMS.items() if v is not None}
if EMB is not None and TOP is not None:
    ARMS["emb+topology"] = np.hstack([EMB, TOP])
if EMB is not None and PER is not None:
    ARMS["emb+persistence"] = np.hstack([EMB, PER])

print(f"\n  {'arm':<20}" + "".join(f"{t:>12}" for t in TEACHERS))
for name, X in ARMS.items():
    row = f"  {name:<20}"
    for t, tab in TEACHERS.items():
        row += f"{evaluate(X, tab):>12.4f}"
    print(row, flush=True)

print("\n  Original rules-era numbers: embeddings 0.6811 | +topology 0.6862")
print("  | topology alone 0.6416 | persistence alone 0.5627.")
print("  A better teacher lifts everything; what matters is whether the GAP")
print("  between embeddings and embeddings+TDA changes, not the absolute level.")
