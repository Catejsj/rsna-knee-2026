"""DINOv2 submission: hidden scans -> submission.csv. Internet OFF.

Same pipeline as the training encode, which is the point: geometry
normalization and the encoder both come from code the training run used, and
every setting (size, slices, planes, PCA, weights) is read out of the model
file rather than restated here, so this cannot silently disagree with what the
weights were fitted on.

RUNTIME: this runs on CPU by design. Kaggle assigns a Tesla P100 (sm_60) while
its PyTorch build supports sm_70+, so CUDA raises `no kernel image is
available`; `--accelerator nvidiaTeslaT4` was accepted but still yielded a
P100. Measured encode rate is ~14 slices/s, and the training run took 6.81
s/study, projecting **2.46 h for 1300 test studies** against the 9 h limit.
A CPU kernel is used deliberately: the GPU cannot execute a single op, and a
CPU kernel gets 12 h instead of 9 and burns no GPU quota.

Efficiency-track note: at ~2.5 h the runtime term `RuntimeSeconds/32400` is
~0.27, versus ~0.004 for the 20-minute pixel model. That is worth roughly
0.12 AUC in the efficiency metric, so this configuration is an accuracy play,
not an efficiency one. A cheaper variant (fewer slices, one plane) would be
needed to contest that track.

Contract: one row per StudyInstanceUID in test.csv ALWAYS; a study that fails
to decode falls back to the training prior rather than vanishing.
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

BATCH = 72
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

COMP = os.path.dirname(glob.glob("/kaggle/input/**/test_series.csv", recursive=True)[0])
MODEL_PATH = glob.glob("/kaggle/input/**/dinov2_model.npz", recursive=True)[0]
print(f"competition: {COMP}\nmodel: {MODEL_PATH}", flush=True)

M = np.load(MODEL_PATH, allow_pickle=True)
LABELS = [str(x) for x in M["labels"]]
SIZE = int(M["size"])
SLICES = int(M["slices"])
PLANES = tuple(str(p) for p in M["planes"])
PRIOR = M["prior"]
COL_MEAN = M["col_mean"]
print(f"model: {SIZE}px x {SLICES} slices, planes={'+'.join(PLANES)}, "
      f"train CV {float(M['cv_macro_auc']):.4f}", flush=True)

roots = [os.path.dirname(p) for p in
         glob.glob("/kaggle/input/**/config.json", recursive=True)
         if "dinov2" in p.lower()]
from transformers import AutoModel
encoder = AutoModel.from_pretrained(roots[0]).eval()
dev = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").mul_(2).cpu()
        dev = "cuda"
    except Exception as e:
        print(f"cuda unusable ({str(e)[:60]}) -- running on CPU", flush=True)
encoder = encoder.to(dev)
print(f"encoder from {roots[0]} on {dev}", flush=True)


@torch.no_grad()
def embed(stack):
    x = torch.from_numpy(stack).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    out = []
    for i in range(0, len(x), BATCH):
        b = x[i:i + BATCH].to(dev)
        out.append(encoder(pixel_values=b).last_hidden_state[:, 0]
                   .float().cpu().numpy())
    return np.concatenate(out)


def predict(vec):
    vec = np.where(np.isnan(vec), COL_MEAN, vec)
    z = (vec - M["mean"]) / M["scale"]
    z = (z - M["pca_mean"]) @ M["pca_components"].T
    logit = z @ M["coefs"].T + M["intercepts"]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


test = pd.read_csv(os.path.join(COMP, "test.csv"))
ser = pd.read_csv(os.path.join(COMP, "test_series.csv"))
root = os.path.join(COMP, "test_series")
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
print(f"test studies: {len(test)}", flush=True)

rows, n_fallback = [], 0
t0 = time.time()
for i, uid in enumerate(test.StudyInstanceUID):
    blocks, found, dim = [], 0, None
    if uid in by_study:
        for plane in PLANES:
            row = F.pick_series_for_plane(by_study[uid], plane)
            stack = None
            if row is not None:
                slices = F.load_series_with_geometry(
                    os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
                if len(slices) >= 3:
                    stack = F.geometric_slice_stack(slices, n_slices=SLICES,
                                                    size=SIZE)
            if stack is None:
                blocks.append(None)
            else:
                e = embed(stack)
                dim = e.shape[1]
                blocks.append(np.concatenate([e.mean(0), e.std(0)]))
                found += 1
    if found == 0:
        probs = PRIOR
        n_fallback += 1
    else:
        width = 2 * dim
        vec = np.concatenate([b if b is not None
                              else np.full(width, np.nan, dtype=np.float32)
                              for b in blocks]).astype(np.float32)
        probs = predict(vec)
    rows.append([uid] + [float(v) for v in probs])
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(test)} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(len(test)-i-1)/60:.0f} min", flush=True)

el = time.time() - t0
sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"\nwrote submission.csv shape={sub.shape}", flush=True)
print(f"fallback-to-prior: {n_fallback}/{len(test)}", flush=True)
print(f"total {el/3600:.2f} h of the 9 h budget "
      f"({el/max(len(test),1):.2f} s/study)", flush=True)
print(sub.head().to_string(), flush=True)
print("DONE", flush=True)
