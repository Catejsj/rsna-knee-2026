"""Stage A at 336px: the last untested representation axis. CPU, ~2 h.

Resolution has never been tested fairly. The one 336px run was **1 plane**, and
it lost (gold 0.6523 vs 224px x 3 planes 0.6811) -- but comparing like with like
showed the deficit tracked *planes*, not resolution, so 336px at 3 planes was
left open. The 0.899 public notebook caches at 336px.

Note the central slice band is NOT a change: geometric_slice_stack already
samples np.linspace(0.15, 0.85) of the ordered stack. The public notebook uses
0.20-0.80. That lever was already pulled, so resolution is the real variable.

Size guard below: 4407 studies x 36 slices x 336^2 uint8 is ~17.9 GB against a
~20 GB kernel output cap. That is thin enough to check rather than assume, so
the job aborts before writing if the projection exceeds the budget.

**Do not train DINOv2-base on this cache.** 336px is 2.25x the compute of 224px,
and base already takes 4.2 h at 224px -- 9.5 h would exceed the 9 h GPU limit.
Small at 336px projects to ~3 h and is the intended consumer.
"""
import os
import time

import numpy as np
import pandas as pd

import glob
import sys

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

SIZE = 336
SLICES = 12
PLANES = ("Sagittal", "Coronal", "Axial")
PER_STUDY = SLICES * len(PLANES)

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
DERIV = os.path.dirname(glob.glob("/kaggle/input/**/labels_rules_v3.csv", recursive=True)[0])

ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
lab = pd.read_csv(os.path.join(DERIV, "labels_rules_v3.csv")).set_index("StudyInstanceUID")
studies = [s for s in lab.index if s in set(ser.StudyInstanceUID)]
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")

n = len(studies)
gb = n * PER_STUDY * SIZE * SIZE / 1e9
BUDGET_GB = 19.0
print(f"projected cache {gb:.1f} GB (cap ~20 GB, budget {BUDGET_GB})", flush=True)
if gb > BUDGET_GB:
    raise SystemExit(f"cache would be {gb:.1f} GB, over the {BUDGET_GB} GB "
                     f"budget -- reduce SIZE or SLICES rather than risk a "
                     f"2 h job dying at the write")
print(f"studies: {n} | {PER_STUDY} slices each | cache ~{gb:.1f} GB", flush=True)

out_path = "/kaggle/working/slices_224_12x3.npy"
cache = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                  shape=(n, PER_STUDY, SIZE, SIZE))
present = np.zeros((n, len(PLANES)), dtype=bool)
keep = []

t0 = time.time()
for i, uid in enumerate(studies):
    for p, plane in enumerate(PLANES):
        row = F.pick_series_for_plane(by_study[uid], plane)
        if row is None:
            continue
        slices = F.load_series_with_geometry(
            os.path.join(root, uid, row.SeriesInstanceUID), max_slices=32)
        if len(slices) < 3:
            continue
        stack = F.geometric_slice_stack(slices, n_slices=SLICES, size=SIZE)
        if stack is None:
            continue
        # geometric_slice_stack returns [0,1] floats; store as uint8
        k = min(len(stack), SLICES)
        cache[i, p * SLICES:p * SLICES + k] = (stack[:k] * 255).astype(np.uint8)
        present[i, p] = True
    keep.append(uid)
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{n} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(n-i-1)/60:.0f} min", flush=True)

cache.flush()
el = time.time() - t0
print(f"\ncached {n} studies in {el/60:.1f} min ({el/n:.2f} s/study)", flush=True)
print(f"planes present per study: {np.bincount(present.sum(1), minlength=4)[1:]} (1,2,3)",
      flush=True)

np.savez_compressed("/kaggle/working/slices_index.npz",
                    study_ids=np.array(keep), present=present,
                    size=SIZE, slices=SLICES, planes=np.array(PLANES))
print("saved index; cache at", out_path, flush=True)
print(f"=> 1300 test studies would decode in {el/n*1300/3600:.2f} h", flush=True)
print("DONE", flush=True)
