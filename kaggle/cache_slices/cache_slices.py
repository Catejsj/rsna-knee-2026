"""Stage A: decode every study once and cache the pixels. CPU kernel.

With the GPU now usable (torch 2.6.0+cu118 on the P100; see kaggle/cuda_probe),
the encoder runs at 419 slices/s and **DICOM decoding is the bottleneck**, at
~1.2 s/study. Decoding inside the training loop would cost 90 minutes per epoch
for work that never changes.

So decode once here, publish as a dataset, and every fine-tuning run afterwards
starts from pixels. That converts each training experiment from ~5 h into ~2 h,
which is what makes iterating on the model realistic in the weeks left.

Geometry normalization is applied here, not later: physical slice ordering,
160 mm centre crop, laterality flip. The cache is what the model trains on, so
it must already be in the corrected frame.

Layout: uint8 [n_studies, 3 planes * 12 slices, 224, 224] ~= 8 GB.
12 slices rather than 24 halves the cache and the epoch time; the step-1 sweep
measured 12 vs 24 at 0.6481 vs 0.6521, a small cost. Three planes are kept
because the 336px experiment showed planes matter for encoder features even
though they did not for raw pixels.
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

SIZE = 224
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
