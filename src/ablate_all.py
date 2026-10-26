"""Full ablation over every feature set we have built. Local, seconds.

Four sources, all aligned by StudyInstanceUID:

    emb224   DINOv2 @224px, 3 planes, CLS token       2304 dims  (v6, LB 0.742)
    emb336   DINOv2 @336px, 1 plane, CLS+patch mean   1536 dims
    topo     Betti curves over image INTENSITY         384 dims
    pers     Asaad ULBP landmarks + distance filtration 288 dims

Three questions:

1. **Did the 336px trade work?** It changed two things at once: resolution up,
   planes 3->1, so on its own it cannot be attributed. Running both encoders
   separately and together is what separates "336 is worse" from "one plane is
   worse".
2. **Does combining the two encoders help?** They see different things; this is
   the ensembling that has never been tried on this project.
3. **Does either topology stream add anything** on top of a strong encoder?

Blocked PCA throughout: each source gets its own component budget, then they
are concatenated. A joint PCA lets the widest source drown the others: already
measured (joint −0.002 vs blocked +0.005 for topology).

Gold is the ruler. It called the graded-target change to within 0.001, while
derived CV had the sign wrong.
"""
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from rsna_labels import DATA, DERIVED, LABELS, ROOT

K = os.path.join(ROOT, "kaggle")
SOURCES = {
    "emb224": (f"{K}/dinov2_encode/out/dinov2_embeddings.npz", "X", 200),
    "emb336": (f"{K}/dinov2_336/out/dinov2_336_embeddings.npz", "X", 200),
    "topo":   (f"{K}/topology_features/out/topology_features.npz", "X", 64),
    "pers":   (f"{K}/persistence_features/out/persistence_features.npz", "X", 64),
}
ALPHA = 20.0

frames, ids = {}, None
for name, (path, key, _) in SOURCES.items():
    d = np.load(path, allow_pickle=True)
    frames[name] = pd.DataFrame(d[key], index=d["study_ids"])
    ids = frames[name].index if ids is None else ids.intersection(frames[name].index)
print("aligned studies:", len(ids))
for n, f in frames.items():
    frames[n] = f.loc[ids]
    print(f"  {n:<8}{frames[n].shape[1]:>6} dims")

lab = pd.read_csv(os.path.join(DERIVED, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
train_csv = pd.read_csv(os.path.join(DATA, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)

G = lab.loc[ids, LABELS].values.astype(np.float32)
B = (G >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in ids])
Y_gold = np.zeros((len(ids), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[np.asarray(ids)[gold_mask], LABELS].values.astype(int)
print(f"gold studies: {gold_mask.sum()}\n")

_cache = {}


def block(name):
    if name not in _cache:
        a = frames[name].values.astype(np.float32)
        a = (a - a.mean(0)) / (a.std(0) + 1e-6)
        n = min(SOURCES[name][2], a.shape[1])
        _cache[name] = PCA(n_components=n, random_state=0,
                           svd_solver="randomized").fit_transform(a)
    return _cache[name]


def evaluate(names):
    Z = np.hstack([block(n) for n in names])
    derived, gold, per = [], [], {}
    for j, l in enumerate(LABELS):
        if B[:, j].sum() < 25 or (1 - B[:, j]).sum() < 25:
            continue
        oof = cross_val_predict(Ridge(alpha=ALPHA), Z, G[:, j], cv=4)
        derived.append(roc_auc_score(B[:, j], oof))
        yg = Y_gold[gold_mask, j]
        if len(np.unique(yg)) == 2:
            g = roc_auc_score(yg, oof[gold_mask])
            gold.append(g)
            per[l] = g
    return float(np.mean(derived)), float(np.mean(gold)), Z.shape[1], per


COMBOS = [
    ("emb224",),                      # v6: the submitted model
    ("emb336",),
    ("topo",),
    ("pers",),
    ("emb224", "emb336"),             # do the two encoders combine?
    ("emb224", "topo"),
    ("emb224", "pers"),
    ("emb224", "topo", "pers"),
    ("emb224", "emb336", "topo", "pers"),
]

print("=" * 78)
print(f"{'features':<34}{'dims':>7}{'derived':>11}{'gold (58)':>12}{'vs v6':>13}")
print("=" * 78)
results = {}
base = None
for combo in COMBOS:
    d, g, dims, per = evaluate(combo)
    results[combo] = (d, g, per)
    if combo == ("emb224",):
        base = g
    delta = "" if base is None or combo == ("emb224",) else f"{g - base:+.4f}"
    star = "  <- v6, LB 0.742" if combo == ("emb224",) else ""
    print(f"{' + '.join(combo):<34}{dims:>7}{d:>11.4f}{g:>12.4f}{delta:>13}{star}")

best = max((c for c in results), key=lambda c: results[c][1])
print(f"\nbest on gold: {' + '.join(best)}  ({results[best][1]:.4f})")

print("\n" + "=" * 78)
print("PER LABEL, gold: v6 vs best combination")
print("=" * 78)
pb, pn = results[("emb224",)][2], results[best][2]
print(f"{'label':<20}{'v6':>9}{'best':>9}{'delta':>9}")
for l in LABELS:
    if l in pb and l in pn:
        print(f"{l:<20}{pb[l]:>9.3f}{pn[l]:>9.3f}{pn[l] - pb[l]:>+9.3f}")
print("\nnote: gold is 58 enriched studies; per-label swings under ~0.10 are noise.")
