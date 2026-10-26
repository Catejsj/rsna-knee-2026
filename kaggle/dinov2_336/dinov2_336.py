"""DINOv2 at 336 px, one plane, patch tokens: trading planes for resolution.

The measurement that motivates this: a DINOv2 ViT-S/14 patch token covers

    224 px over a 160 mm crop -> 16x16 patches -> 10.0 mm per patch
    336 px                    -> 24x24         ->  6.7 mm per patch

against a meniscus 3-5 mm thick and articular cartilage 2-4 mm. At 224 px a
meniscus is a third of one token: averaged away before the model sees it. Only
effusion (10-40 mm) and the ACL (~10 mm) are resolved at all.

Two changes, each independently justified:

  * **336 px instead of 224.** Halves the physical size of a patch.
  * **patch tokens, not just CLS.** The previous run used the CLS token alone
    and discarded all 256 patch tokens, which are where spatial detail lives.
    Here: CLS concatenated with the mean over patch tokens.

Paid for by dropping to sagittal only. Three planes were measured to add
**zero** real skill (gold +0.000 while derived CV rose +0.032), so this trades
something known worthless for something physically motivated, and comes out
*faster* than the current model: ~1.55 h inference against 2.5 h.

Also saves **per-slice** embeddings this time. The previous encode kept only
the pooled mean+std, which is why attention pooling would otherwise cost a
full 8 h re-encode. Same compute, and it makes that class of experiment free.

CPU kernel by design: Kaggle assigns a P100 (sm_60) that its own PyTorch build
cannot use, so the GPU is dead weight, and a CPU kernel gets 12 h instead of 9
and burns no GPU quota.
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

SIZE = 336
SLICES = 24
PLANES = ("Sagittal",)
BATCH = 24
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

roots = [os.path.dirname(p) for p in
         glob.glob("/kaggle/input/**/config.json", recursive=True)
         if "dinov2" in p.lower()]
from transformers import AutoModel
model = AutoModel.from_pretrained(roots[0]).eval()
dev = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").mul_(2).cpu()
        dev = "cuda"
    except Exception as e:
        print(f"cuda unusable ({str(e)[:60]}) -- CPU, as expected", flush=True)
model = model.to(dev)
print(f"encoder {roots[0]} on {dev}", flush=True)
print(f"{SIZE}px -> {(SIZE//14)**2} patches, {160.0/(SIZE//14):.2f} mm each",
      flush=True)


@torch.no_grad()
def embed(stack):
    """[k,H,W] -> [k, 2*D]: CLS token concatenated with the patch-token mean."""
    x = torch.from_numpy(stack).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    out = []
    for i in range(0, len(x), BATCH):
        h = model(pixel_values=x[i:i + BATCH].to(dev)).last_hidden_state
        cls, patches = h[:, 0], h[:, 1:].mean(1)
        out.append(torch.cat([cls, patches], dim=1).float().cpu().numpy())
    return np.concatenate(out)


ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies: {len(studies)}", flush=True)

pooled, per_slice, keep = [], [], []
t0 = time.time()
for i, uid in enumerate(studies):
    row = F.pick_series_for_plane(by_study[uid], PLANES[0])
    if row is None:
        continue
    slices = F.load_series_with_geometry(
        os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
    if len(slices) < 3:
        continue
    stack = F.geometric_slice_stack(slices, n_slices=SLICES, size=SIZE)
    if stack is None:
        continue
    e = embed(stack)                                  # [k, 768]
    pooled.append(np.concatenate([e.mean(0), e.std(0)]).astype(np.float32))
    # pad/truncate to a fixed slice count so this stacks into one array
    fixed = np.zeros((SLICES, e.shape[1]), dtype=np.float16)
    fixed[:len(e)] = e[:SLICES].astype(np.float16)
    per_slice.append(fixed)
    keep.append(uid)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(studies)} ok={len(keep)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(studies)-i-1)/60:.0f} min", flush=True)

X = np.array(pooled, dtype=np.float32)
S = np.stack(per_slice)
keep = np.array(keep)
elapsed = time.time() - t0
per_study = elapsed / max(len(studies), 1)
print(f"\npooled {X.shape} | per-slice {S.shape} in {elapsed/60:.1f} min "
      f"({per_study:.2f} s/study)", flush=True)
print(f"=> 1300 test studies: {per_study*1300/3600:.2f} h of the 9 h budget "
      f"(v6 at 224px x3 planes was 2.46 h)", flush=True)

np.savez_compressed("/kaggle/working/dinov2_336_embeddings.npz",
                    X=X, per_slice=S, study_ids=keep,
                    size=SIZE, slices=SLICES, planes=np.array(PLANES))
print("saved embeddings (pooled + per-slice)", flush=True)

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

train_csv = pd.read_csv(os.path.join(COMP, "train.csv"))
gold_df = train_csv[train_csv["ACL"].notna()].set_index("StudyInstanceUID")
gold_ids = set(gold_df.index)
G = lab.loc[keep, LABELS].values.astype(np.float32)
B = (G >= 0.5).astype(int)
gold_mask = np.array([u in gold_ids for u in keep])
Y_gold = np.zeros((len(keep), len(LABELS)), dtype=int)
Y_gold[gold_mask] = gold_df.loc[keep[gold_mask], LABELS].values.astype(int)

mean, scale = X.mean(0), X.std(0) + 1e-6
Z = PCA(n_components=200, random_state=0,
        svd_solver="randomized").fit_transform((X - mean) / scale)
d_aucs, g_aucs = [], []
print(f"\n{'label':<18}{'derived':>10}{'gold':>9}")
for j, l in enumerate(LABELS):
    if B[:, j].sum() < 25 or (1 - B[:, j]).sum() < 25:
        continue
    oof = cross_val_predict(Ridge(alpha=20.0), Z, G[:, j], cv=4)
    da = roc_auc_score(B[:, j], oof)
    d_aucs.append(da)
    yg = Y_gold[gold_mask, j]
    ga = roc_auc_score(yg, oof[gold_mask]) if len(np.unique(yg)) == 2 else float("nan")
    if ga == ga:
        g_aucs.append(ga)
    print(f"{l:<18}{da:>10.3f}{ga:>9.3f}", flush=True)
print(f"\nMACRO derived {np.mean(d_aucs):.4f}  gold {np.mean(g_aucs):.4f}")
print("v6 (224px, 3 planes, CLS only): derived 0.7643, gold 0.6815, LB 0.742")
print("gold is the ruler that has tracked the leaderboard.")
print("DONE", flush=True)
