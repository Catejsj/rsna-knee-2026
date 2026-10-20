"""Set-to-set retrieval: slice matching, Optimal Transport, and graph propagation.

Four retrieval ideas (2026-10-05), all measured on the SAME per-slice features so
they are comparable, and against the trained attention head on those features:

  mean     one vector per study, cosine kNN (the earlier retrieval baseline)
  match    "break both down and reassemble": every query slice finds its best
           counterpart slice; similarity = average best-match (a match %)
  OT       Optimal Transport (entropic EMD, Sinkhorn): the cheapest one-to-one
           slice correspondence between the two studies
  graph    kNN graph over ALL studies, teacher labels propagated along edges
           (label propagation -- retrieval as a graph)

Why set-to-set could matter: a study is a bag of 36 slices and a finding lives
in a few. Averaging dilutes a 2-slice tear with 34 normal slices; matching and
OT compare slice to slice.

Matching is WITHIN plane only (sagittal slice to sagittal slice). A blank-slice
mask is kept as a guard, but on this cache it masks nothing: checked
2026-10-05, every study has all three planes and no plane's 12 slice
embeddings are identical (minimum within-plane spread 9.0).

Features: frozen MedImageInsight per-slice embeddings from `rsna-mii-embed`
(`[4407, 36, 2048]`). Trained attention head on these exact features: gold 0.7986.

Usage:  MII_DIR=... .venv/bin/python src/retrieval_ot.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# output of the `rsna-mii-embed` kernel (mii_embeddings.npy, mii_study_ids.npy)
MII = os.environ.get("MII_DIR", os.path.join(R, "external", "mii_out"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANES, PER = 3, 12
HEAD_GOLD = 0.7986

E = np.load(f"{MII}/mii_embeddings.npy", mmap_mode="r")
ids = np.load(f"{MII}/mii_study_ids.npy", allow_pickle=True)
n, S, Dm = E.shape
assert S == PLANES * PER

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


def macro(P):
    out = []
    for j in range(len(LABELS)):
        y = Yg[:, j]
        if len(np.unique(y)) == 2:
            r = rankdata(P[:, j]); n1 = y.sum(); n0 = len(y) - n1
            out.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


# ---- blank-slice mask -------------------------------------------------------
# A black slice always yields the same embedding. Find it as the most common
# (rounded) slice vector and mask every slice that matches it.
X = np.asarray(E, dtype=np.float32)                       # [n, 36, D]
flat = X.reshape(-1, Dm)
key = np.round(flat[:, :8], 2)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
blank_vec = flat[np.where(inv.ravel() == cnt.argmax())[0][0]]
VALID = (np.linalg.norm(X - blank_vec, axis=2) > 1e-2 * np.linalg.norm(blank_vec))
print(f"blank slices masked: {(~VALID).mean():.1%} of all slices "
      f"(missing planes are written as zeros)")

# ---- reduce + normalise slices ----------------------------------------------
pca = PCA(n_components=128, random_state=0).fit(
    flat[np.random.default_rng(0).choice(len(flat), 40000, replace=False)])
Z = pca.transform(flat).reshape(n, S, -1).astype(np.float32)
Z /= np.maximum(np.linalg.norm(Z, axis=2, keepdims=True), 1e-9)


def vote(sim, k, tau):
    """sim: [n_q, n_db] (higher = closer). Kernel-weighted soft label vote."""
    top = np.argsort(-sim, axis=1)[:, :k]
    s = np.take_along_axis(sim, top, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    return np.einsum("qk,qkl->ql", w, G[db][top])


def best_vote(sim):
    best = (-1, None)
    for k in (10, 25, 50, 100, 200):
        for tau in (0.005, 0.02, 0.05, 0.2):
            a = macro(vote(sim, k, tau))
            if a > best[0]:
                best = (a, (k, tau))
    return best


# ---- (1) mean-pooled cosine -------------------------------------------------
M = (Z * VALID[..., None]).sum(1) / np.maximum(VALID.sum(1, keepdims=True), 1)
M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
sim_mean = M[q] @ M[db].T


# ---- (2) slice matching and (3) OT, per plane --------------------------------
def sinkhorn_cost(C, a, b, eps=0.05, iters=100):
    """Batched entropic OT. C [B,m,n], a [B,m], b [B,n] -> transport cost [B]."""
    K = np.exp(-C / eps)
    u = np.ones_like(a)
    for _ in range(iters):
        v = b / np.maximum(np.einsum("bmn,bm->bn", K, u), 1e-30)
        u = a / np.maximum(np.einsum("bmn,bn->bm", K, v), 1e-30)
    Pm = u[:, :, None] * K * v[:, None, :]
    return (Pm * C).sum((1, 2))


sim_match = np.zeros((len(q), len(db)), dtype=np.float32)
sim_ot = np.zeros((len(q), len(db)), dtype=np.float32)
for qi, qq in enumerate(q):
    tot_m = np.zeros(len(db)); tot_o = np.zeros(len(db)); shared = np.zeros(len(db))
    for p in range(PLANES):
        sl = slice(p * PER, (p + 1) * PER)
        vq = VALID[qq, sl]
        if not vq.any():
            continue
        Aq = Z[qq, sl][vq]                                   # [m, d]
        Bd = Z[db, sl]                                       # [N, 12, d]
        vd = VALID[db, sl]                                   # [N, 12]
        has = vd.any(1)
        cos = np.einsum("md,nkd->nmk", Aq, Bd)               # [N, m, 12]
        cos = np.where(vd[:, None, :], cos, -2.0)
        # match: each query slice's best counterpart, and symmetric
        fwd = cos.max(2).mean(1)
        bwd = np.where(vd, cos.max(1), np.nan)
        bwd = np.nanmean(np.where(vd, bwd, np.nan), 1)
        m_s = np.where(has, 0.5 * (fwd + np.nan_to_num(bwd, nan=-1)), 0.0)
        # OT: uniform mass over valid slices, cost = 1 - cos
        C = 1.0 - np.clip(cos, -1, 1)
        a = np.full((len(db), Aq.shape[0]), 1.0 / Aq.shape[0])
        b = vd / np.maximum(vd.sum(1, keepdims=True), 1)
        C = np.where(vd[:, None, :], C, 10.0)                # no mass goes to blanks
        o_s = np.where(has, -sinkhorn_cost(C, a, b), 0.0)
        tot_m += m_s * has; tot_o += o_s * has; shared += has
    sim_match[qi] = np.where(shared > 0, tot_m / np.maximum(shared, 1), -9)
    sim_ot[qi] = np.where(shared > 0, tot_o / np.maximum(shared, 1), -9)
    if (qi + 1) % 10 == 0:
        print(f"  queries {qi+1}/{len(q)}", flush=True)


# ---- (4) graph label propagation -------------------------------------------
def label_propagation(Mv, k=20, alpha=0.9, iters=40):
    Sall = Mv @ Mv.T
    np.fill_diagonal(Sall, -1)
    nb = np.argsort(-Sall, axis=1)[:, :k]
    W = np.zeros_like(Sall)
    rows = np.repeat(np.arange(len(Mv)), k)
    W[rows, nb.ravel()] = np.maximum(Sall[rows, nb.ravel()], 0)
    W = np.maximum(W, W.T)
    d = W.sum(1); d[d == 0] = 1
    Sn = W / np.sqrt(d[:, None] * d[None, :])
    Y0 = np.zeros_like(G); Y0[db] = G[db]
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * Sn @ F + (1 - alpha) * Y0
    return F[q]


print(f"\n  {'method':<34}{'gold':>8}   best (k, tau)")
for name, sim in [("mean-pooled cosine kNN", sim_mean),
                  ("slice match ('reassemble') kNN", sim_match),
                  ("Optimal Transport (EMD) kNN", sim_ot)]:
    a, cfg = best_vote(sim)
    print(f"  {name:<34}{a:>8.4f}   {cfg}", flush=True)
best_lp = max((macro(label_propagation(M, k, al)), k, al)
              for k in (10, 20, 50) for al in (0.5, 0.9, 0.99))
print(f"  {'graph label propagation':<34}{best_lp[0]:>8.4f}   (k={best_lp[1]}, alpha={best_lp[2]})")
print(f"  {'trained attention head (same features)':<34}{HEAD_GOLD:>8.4f}")
print("\n  Every retrieval row is the best over its grid -- optimistic in its favour.")
