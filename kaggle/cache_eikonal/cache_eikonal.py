"""Eikonal arrival-time channels, sagittal plane. CPU, ~5 h (probed).

`|grad u| = 1/f` -- u is when a wavefront reaches a voxel, f its local speed.
With `mode="bright"` the front runs fast through bright voxels, so on
fluid-sensitive sequences the arrival time measures **how far fluid has spread
from the joint centre**: an effusion sits in the joint, a Baker's cyst has
tracked posteromedially -- geodesically far but connected.

**v1 was cancelled at 45%.** It solved all three planes and ran at 12.6 s/study
on Kaggle against 3.4 s/study benchmarked locally -- 15.4 h projected against a
12 h limit, so it would have been killed at ~77% with no index file. Two changes:
**sagittal only** (features.py already selects sagittal fluid-sensitive for 4150
of 4407 studies, and effusion and Baker's cysts are clearest there), and a
**throughput probe that aborts** instead of discovering the overrun hours in.

Also correcting an overstatement from v1: at 56px the in-plane spacing is
2.86 mm against 3.0 mm between slices, so the volume is very nearly isotropic.
The physical scaling is still right, but the emphasis on anisotropy mattering
does not apply at this resolution.

**Why now.** This was designed and parked because it costs inference time and
the 9 h cap was binding. The plateau changed the arithmetic: all fifteen members
now land within +/-0.001 of the shipped ensemble, so an extra ResNet buys ~0.001
and giving up ~1.2 h of budget costs almost nothing. The bar this has to clear is
low.

**Why it is not a fourth topology attempt.** Betti curves (+0.005) and Asaad
persistence (0.5627) both re-described what DINOv2 already encodes. This does
not: the whole pipeline is 2D and unitless -- every encoder sees one slice at a
time and the head only pools -- so *nothing anywhere computes a 3D quantity or a
millimetre*. Arrival time is information the encoder cannot derive. That is the
rule which has predicted every win here.

Cost, as measured ON KAGGLE (the local numbers were 3.7x optimistic):

    3 planes, 12x56x56   12.6 s/study -> 15.4 h. Over the 12 h limit. Cancelled.
    1 plane,  12x56x56   ~4.2 s/study -> ~5 h train, ~1.5 h inference.

The solve is **3D within the plane**, which keeps the property that matters --
genuinely three-dimensional, in millimetres -- and drops only in-plane detail.
Arrival time is a smooth field, so upsampling 56 -> 224 at training time loses
little, and storing at 56 keeps this cache at ~0.17 GB.

Spacing is passed explicitly and is physically correct, though at 56px it is
2.86 mm in-plane against 3.0 mm between slices, i.e. nearly isotropic. It would
matter at full resolution; here it is close to a formality.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_eikonal as EK

SIZE = 56
PLANES = 1        # sagittal only -- see the header
PER_PLANE = 12
CROP_MM = 160.0            # the physical crop features.py already applies
SLICE_MM = 3.0             # typical knee MRI inter-slice spacing

_npy = [q for q in glob.glob("/kaggle/input/**/*.npy", recursive=True) if "slices" in q]
_idx = [q for q in glob.glob("/kaggle/input/**/*.npz", recursive=True) if "index" in q]
assert _npy and _idx, "cache not found"

meta = np.load(_idx[0], allow_pickle=True)
ids = meta["study_ids"]
X = np.load(_npy[0], mmap_mode="r")
n = len(ids)
print(f"cache {X.shape} | {n} studies", flush=True)
assert X.shape[-1] == 224, f"expected 224px cache, got {X.shape[-1]}px"

gb = n * PLANES * PER_PLANE * SIZE * SIZE / 1e9
print(f"projected eikonal cache {gb:.2f} GB at {SIZE}px", flush=True)
assert gb < 5.0, f"{gb:.1f} GB is larger than intended -- check SIZE"

out = np.lib.format.open_memmap("/kaggle/working/eikonal_56.npy", mode="w+",
                                dtype=np.uint8,
                                shape=(n, PLANES * PER_PLANE, SIZE, SIZE))

# in-plane mm per pixel after the 160 mm crop, then downsampling to 56
px_mm = CROP_MM / SIZE
spacing = np.array([SLICE_MM, px_mm, px_mm])
print(f"spacing (slice, y, x) = {spacing} mm", flush=True)


def downsample(block):
    """[12,224,224] uint8 -> [12,56,56] float in [0,1] by 4x4 mean."""
    b = block.reshape(PER_PLANE, SIZE, 4, SIZE, 4).mean(axis=(2, 4))
    return b.astype(np.float64) / 255.0


# ---- throughput probe -------------------------------------------------------
# The previous run estimated 4.5 h from a LOCAL benchmark and actually ran at
# 12.6 s/study -> 15.4 h, past the 12 h CPU limit. It would have been killed at
# ~77% with no index file written. Kaggle's CPU is slower than the dev machine;
# this is the fourth time that has cost a job. Measure here, on Kaggle, and
# refuse to start rather than discover it eight hours in.
BUDGET_H = 10.0
_t = time.time()
for i in range(20):
    stack = np.asarray(X[i])
    for p in range(PLANES):
        vol = downsample(stack[p * PER_PLANE:(p + 1) * PER_PLANE])
        if vol.max() <= 0:
            continue
        seeds = np.zeros(vol.shape, dtype=bool)
        seeds[PER_PLANE // 2, SIZE // 2, SIZE // 2] = True
        EK.fast_marching(EK.speed_from_intensity(vol, "bright"), seeds,
                         spacing=spacing)
per_study = (time.time() - _t) / 20
projected = per_study * n / 3600
print(f"PROBE: {per_study:.2f} s/study measured on Kaggle -> "
      f"{projected:.1f} h projected", flush=True)
if projected > BUDGET_H:
    raise SystemExit(
        f"projected {projected:.1f} h exceeds the {BUDGET_H} h budget (12 h "
        f"kernel limit). Reduce SIZE or PLANES -- do not start a job that will "
        f"be killed at ~{BUDGET_H/projected*100:.0f}% with no index written.")

t0 = time.time()
n_fail = 0
for i in range(n):
    stack = np.asarray(X[i])
    for p in range(PLANES):
        vol = downsample(stack[p * PER_PLANE:(p + 1) * PER_PLANE])
        if vol.max() <= 0:
            continue                       # plane absent; leave zeros
        try:
            speed = EK.speed_from_intensity(vol, "bright")
            seeds = np.zeros(vol.shape, dtype=bool)
            seeds[PER_PLANE // 2, SIZE // 2, SIZE // 2] = True
            u = EK.fast_marching(speed, seeds, spacing=spacing)
            finite = np.isfinite(u)
            if not finite.any():
                n_fail += 1
                continue
            # unreached voxels are "infinitely far": clamp to the max arrival
            hi = u[finite].max()
            u = np.where(finite, u, hi)
            if hi > 0:
                u = u / hi
            out[i, p * PER_PLANE:(p + 1) * PER_PLANE] = (u * 255).astype(np.uint8)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  study {i} plane {p} failed: {str(e)[:70]}", flush=True)
    if (i + 1) % 200 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{n}  {el/(i+1):.2f} s/study  "
              f"eta {el/(i+1)*(n-i-1)/3600:.1f} h", flush=True)

out.flush()
np.savez_compressed("/kaggle/working/eikonal_index.npz",
                    study_ids=ids, size=SIZE, planes=PLANES,
                    per_plane=PER_PLANE, spacing=spacing)
print(f"\ndone in {(time.time()-t0)/3600:.2f} h | failures {n_fail}", flush=True)
print("saved eikonal_56.npy + eikonal_index.npz\nDONE", flush=True)
