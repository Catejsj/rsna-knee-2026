"""Step 2: replace 64px thumbnails with DINOv2 embeddings.

Everything so far has fed the model a 64x64 average of 24 slices — the crudest
representation available, chosen to answer a question rather than to be good.
It reached LB 0.669. Every public notebook above 0.809 uses DINOv2 instead.

Held identical to v4 so the leaderboard delta attributes to the encoder alone:
same 4407 studies, same three planes, same 24 slices, same geometry
normalization (physical slice order, 160 mm crop, laterality flip), same
PCA-100 + per-label logistic head.

Changed: each slice goes through a frozen DINOv2 ViT-S/14 at 224 px instead of
being shrunk to 64x64. Per plane the slice embeddings are pooled by mean and
std (384 -> 768); three planes give 2304 features.

Two notes carried from earlier results:

* The step-1 sweep found 224 px WORSE than 64 px — but that was raw pixels into
  a linear model, where extra pixels are extra noise. An encoder is built to
  use fine detail, so resolution is re-tested here rather than inherited.
* Judge this by the submission. Derived-label CV rated the three-plane change
  +0.061 (LB gave +0.005) and the geometry change +0.001 (LB gave +0.036). It
  rewards capacity and is blind to correctness.

Writes per-study embeddings so later head experiments skip the ~90 min decode.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import glob
import sys

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

SIZE = 224
SLICES = 24
PLANES = ("Sagittal", "Coronal", "Axial")
BATCH = 72
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

# ------------------------------------------------------------ load the encoder
print("model input tree:", flush=True)
for p in sorted(glob.glob("/kaggle/input/dinov2/**", recursive=True))[:25]:
    print("   ", p, flush=True)


def load_dinov2():
    """Kaggle's metaresearch/dinov2 instance. Layout is not documented, so try
    the plausible formats and say which one worked."""
    roots = [os.path.dirname(p) for p in
             glob.glob("/kaggle/input/**/config.json", recursive=True)
             if "dinov2" in p.lower()]
    for r in roots:
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(r)
            print(f"loaded HF-format DINOv2 from {r}", flush=True)
            return m, "hf"
        except Exception as e:
            print(f"  HF load failed at {r}: {str(e)[:90]}", flush=True)
    for pat in ("**/*.pth", "**/*.pt", "**/*.bin"):
        for w in glob.glob(f"/kaggle/input/{pat}", recursive=True):
            if "dinov2" not in w.lower():
                continue
            try:
                m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                   pretrained=False)
                sd = torch.load(w, map_location="cpu")
                m.load_state_dict(sd if not isinstance(sd, dict) or "state_dict"
                                  not in sd else sd["state_dict"])
                print(f"loaded torch.hub DINOv2 with weights {w}", flush=True)
                return m, "hub"
            except Exception as e:
                print(f"  hub load failed at {w}: {str(e)[:90]}", flush=True)
    raise RuntimeError("could not load DINOv2 — see the tree above")


model, kind = load_dinov2()
model.eval()
dev = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").mul_(2).cpu()
        dev = "cuda"
    except Exception as e:
        print("cuda unusable:", str(e)[:80], flush=True)
model = model.to(dev)
print(f"device: {dev}  format: {kind}", flush=True)


@torch.no_grad()
def embed(stack):
    """[k, H, W] float in [0,1] -> [k, D] embeddings."""
    x = torch.from_numpy(stack).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    out = []
    for i in range(0, len(x), BATCH):
        b = x[i:i + BATCH].to(dev)
        if kind == "hf":
            feats = model(pixel_values=b).last_hidden_state[:, 0]   # CLS token
        else:
            feats = model(b)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------- encode
ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies: {len(studies)}", flush=True)

X, keep = [], []
dim = None
t0 = time.time()
for i, uid in enumerate(studies):
    blocks, found = [], 0
    for plane in PLANES:
        row = F.pick_series_for_plane(by_study[uid], plane)
        stack = None
        if row is not None:
            slices = F.load_series_with_geometry(
                os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
            if len(slices) >= 3:
                stack = F.geometric_slice_stack(slices, n_slices=SLICES, size=SIZE)
        if stack is None:
            blocks.append(None)
        else:
            e = embed(stack)
            dim = e.shape[1]
            blocks.append(np.concatenate([e.mean(0), e.std(0)]))
            found += 1
    if found == 0:
        continue
    width = 2 * dim
    X.append(np.concatenate([b if b is not None else np.full(width, np.nan,
                                                             dtype=np.float32)
                             for b in blocks]).astype(np.float32))
    keep.append(uid)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(X)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

X = np.array(X, dtype=np.float32)
keep = np.array(keep)
elapsed = time.time() - t0
per_study = elapsed / max(len(studies), 1)
print(f"\nembeddings {X.shape} in {elapsed/60:.1f} min ({per_study:.2f} s/study)",
      flush=True)
print(f"=> 1300 test studies would take {per_study*1300/3600:.2f} h of 9 h", flush=True)

col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

np.savez_compressed("/kaggle/working/dinov2_embeddings.npz",
                    X=X, study_ids=keep, col_mean=col_mean,
                    size=SIZE, slices=SLICES, planes=np.array(PLANES), dim=dim)
print("cached embeddings -> /kaggle/working/dinov2_embeddings.npz", flush=True)

# ---------------------------------------------------------------------- head
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

mean = X.mean(0)
scale = X.std(0) + 1e-6
pca = PCA(n_components=100, random_state=0,
          svd_solver="randomized").fit((X - mean) / scale)
Z = pca.transform((X - mean) / scale)
print(f"PCA {Z.shape}, explained var {pca.explained_variance_ratio_.sum():.2f}",
      flush=True)

Y = (lab.loc[keep, LABELS].values >= 0.5).astype(int)
coefs, intercepts, prior, aucs = [], [], [], []
print(f"\n{'label':<18}{'n+':>6}{'cv AUC':>9}")
for j, l in enumerate(LABELS):
    y = Y[:, j]
    p = float(y.mean())
    prior.append(p)
    if y.sum() < 25 or (1 - y).sum() < 25:
        coefs.append(np.zeros(Z.shape[1]))
        intercepts.append(np.log((p + 1e-6) / (1 - p + 1e-6)))
        print(f"{l:<18}{y.sum():>6}{'skip':>9}")
        continue
    m = LogisticRegression(max_iter=3000, C=0.05).fit(Z, y)
    cv = cross_val_predict(LogisticRegression(max_iter=3000, C=0.05), Z, y,
                           cv=4, method="predict_proba")[:, 1]
    aucs.append(roc_auc_score(y, cv))
    coefs.append(m.coef_[0])
    intercepts.append(m.intercept_[0])
    print(f"{l:<18}{y.sum():>6}{aucs[-1]:>9.3f}", flush=True)

print(f"\nMACRO CV AUC: {np.mean(aucs):.4f}   (v4 pixels: CV 0.7123 / LB 0.669)",
      flush=True)
print("CV is not the ruler here — submit and read the leaderboard.", flush=True)

np.savez_compressed(
    "/kaggle/working/dinov2_model.npz",
    mean=mean, scale=scale, col_mean=col_mean,
    pca_mean=pca.mean_, pca_components=pca.components_,
    coefs=np.array(coefs), intercepts=np.array(intercepts),
    prior=np.array(prior), labels=np.array(LABELS), planes=np.array(PLANES),
    size=SIZE, slices=SLICES, dim=dim, per_study_seconds=per_study,
    cv_macro_auc=float(np.mean(aucs)),
)
print("saved /kaggle/working/dinov2_model.npz\nDONE", flush=True)
