"""Why is the DINOv2 encode taking >6 h? Measure it on 40 studies.

The full run has no visibility (Kaggle releases logs only on completion), so
this reproduces the identical setup on a small sample and times the two halves
separately. It answers the question the big run cannot:

  * did it actually get a GPU, or silently fall back to CPU?
  * how much of the cost is DICOM decode vs the encoder forward pass?
  * what scope (planes x slices x studies) fits a sane iteration loop?

This is the throughput measurement that should have preceded the full run.
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

N_STUDIES = 40
SIZE = 224
SLICES = 24
PLANES = ("Sagittal", "Coronal", "Axial")
BATCH = 72

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

print("=" * 66)
print("ENVIRONMENT")
print("=" * 66)
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device name: {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability(0)}")

dev = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(8, device="cuda").mul_(2).cpu()
        dev = "cuda"
    except Exception as e:
        print("cuda unusable:", str(e)[:120])
print(f"USING: {dev}", flush=True)

roots = [os.path.dirname(p) for p in
         glob.glob("/kaggle/input/**/config.json", recursive=True)
         if "dinov2" in p.lower()]
print("candidate model dirs:", roots, flush=True)
from transformers import AutoModel
model = AutoModel.from_pretrained(roots[0]).eval().to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"loaded {roots[0]}  params={n_params/1e6:.1f}M", flush=True)


@torch.no_grad()
def embed(stack):
    x = torch.from_numpy(stack).float().unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    out = []
    for i in range(0, len(x), BATCH):
        b = x[i:i + BATCH].to(dev)
        out.append(model(pixel_values=b).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(out)


# warm up so the first batch's lazy init is not charged to the measurement
_ = embed(np.zeros((8, SIZE, SIZE), dtype=np.float32))
if dev == "cuda":
    torch.cuda.synchronize()

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
uids = list(by_study)[:N_STUDIES]

t_decode = t_encode = 0.0
n_slices_total = 0
for uid in uids:
    for plane in PLANES:
        row = F.pick_series_for_plane(by_study[uid], plane)
        if row is None:
            continue
        t = time.time()
        slices = F.load_series_with_geometry(
            os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
        stack = (F.geometric_slice_stack(slices, n_slices=SLICES, size=SIZE)
                 if len(slices) >= 3 else None)
        t_decode += time.time() - t
        if stack is None:
            continue
        t = time.time()
        embed(stack)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_encode += time.time() - t
        n_slices_total += len(stack)

total = t_decode + t_encode
print("\n" + "=" * 66)
print(f"TIMING over {len(uids)} studies ({n_slices_total} slices encoded)")
print("=" * 66)
print(f"  decode + geometry : {t_decode:7.1f} s  ({t_decode/total*100:4.1f}%)"
      f"  = {t_decode/len(uids):.2f} s/study")
print(f"  encoder forward   : {t_encode:7.1f} s  ({t_encode/total*100:4.1f}%)"
      f"  = {t_encode/len(uids):.2f} s/study")
print(f"  encoder rate      : {n_slices_total/max(t_encode,1e-9):7.1f} slices/s")
print(f"  TOTAL             : {total/len(uids):.2f} s/study")

for n, label in ((4407, "full train set"), (1300, "test set")):
    print(f"\n  projected for {label} ({n} studies): "
          f"{total/len(uids)*n/3600:.2f} h")

print("\nscope options at the measured rate (train / test hours):")
for planes, sl in ((3, 24), (1, 24), (3, 12), (1, 12)):
    factor = (planes / 3) * (sl / 24)
    per = total / len(uids) * factor
    print(f"  {planes} plane(s) x {sl:2d} slices : "
          f"{per*4407/3600:5.2f} h / {per*1300/3600:5.2f} h")
print("\nDONE", flush=True)
