"""Physically contiguous slice triplets, sagittal. CPU, ~1 h (probed).

2.5D is the only refuted idea on this project with a measured reason to expect a
different answer. Unfreezing conv1 halved its deficit (-0.029 -> -0.0142), which
says neighbouring slices ARE information the frozen first layer could not read.
The residual gap was blamed on the "neighbours" not being neighbours. **That is
now confirmed by reading the code rather than guessing:**

    load_series_with_geometry(max_slices=32)   subsamples a ~40-file series to 32
    geometric_slice_stack                      takes 12 anchors across the 0.15-0.85 band

Both steps decimate. For a 40-slice series the 12 cached slices sit ~2.5 files
apart, i.e. **~7-8 mm** of anatomy, against a true inter-slice gap of ~3 mm. The
v18/v21 "2.5D" runs were therefore stacking slices from opposite sides of a
meniscus and calling them adjacent. This cache stacks the real neighbours.

### What makes this a clean test

Layout is `[n_studies, 12 anchors * 3 offsets, 224, 224]`, offset 0/1/2 =
prev/anchor/next, so **the control lives in the same cache**: replicate offset 1
three times for `[I,I,I]` and take 0/1/2 for true 2.5D. Same studies, same
anchors, same crop, same flip, one array -- the only variable is whether the two
outer channels are the physical neighbours or copies of the middle.

That was not true of the earlier attempts, which compared against a separately
built cache.

### Sagittal only, and why that is not a compromise

Three planes of triplets would be 24 GB, past the ~20 GB output cap. Sagittal is
the plane `features.py` already prefers (fluid-sensitive sagittal exists for 4150
of 4407 studies) and menisci and cruciates -- the findings a through-plane
neighbour should most help -- are read there. The control is sagittal-only too,
so the comparison stays single-variable; this measures the MECHANISM. If it wins,
the 3-plane version is a second, larger job.

### Cheaper than the cache it replaces

Header-only pass (`stop_before_pixels=True`) to establish physical order, then
pixels for just the 36 slices actually kept. The existing cache decodes 96
full images per study; this decodes 36. The probe below measures it rather
than trusting that reasoning.

It also records the two gap distributions in the index, so if the premise is
wrong -- if the anchors turn out to already be ~3 mm apart -- that shows up in
the log before a single GPU-hour is spent on it.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

SIZE = 224
ANCHORS = 12
OFFSETS = 3                # prev, anchor, next
PER_STUDY = ANCHORS * OFFSETS
PLANE = "Sagittal"
CROP_MM = F.CROP_MM
BUDGET_H = 9.0             # CPU kernels get 12 h; leave headroom

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
assert gb < 12.0, f"{gb:.1f} GB is larger than intended -- check SIZE/ANCHORS"


def build_study(uid):
    """[36, 224, 224] uint8 triplets + (true_gap_mm, anchor_gap_mm), or None."""
    row = F.pick_series_for_plane(by_study[uid], PLANE)
    if row is None:
        return None, None, None
    d = os.path.join(root, uid, row.SeriesInstanceUID)
    try:
        files = sorted(os.listdir(d))
    except OSError:
        return None, None, None
    if len(files) < 3:
        return None, None, None

    # pass 1: headers only -- no pixel decode, so reading the WHOLE series is
    # affordable and we never have to subsample before knowing the real order
    geo = []
    for name in files:
        try:
            ds = pydicom.dcmread(os.path.join(d, name), stop_before_pixels=True)
        except Exception:
            continue
        pos, spacing, side = F._slice_geometry(ds)
        geo.append({"name": name, "pos": pos, "spacing": spacing, "side": side})
    if len(geo) < 3:
        return None, None, None

    # physical order where available, else filename order (same rule as
    # geometric_slice_stack -- a partially sorted series is worse than an
    # unsorted one, so it is all or nothing)
    if all(g["pos"] is not None for g in geo):
        geo = sorted(geo, key=lambda g: g["pos"])
        pv = np.array([g["pos"] for g in geo], dtype=float)
        true_gap = float(np.median(np.abs(np.diff(pv)))) if len(pv) > 1 else np.nan
    else:
        pv, true_gap = None, np.nan

    N = len(geo)
    span = np.linspace(0.15, 0.85, min(ANCHORS, N)) * (N - 1)
    anchor_idx = [int(p) for p in span]
    if pv is not None and len(anchor_idx) > 1:
        anchor_gap = float(np.median(np.abs(np.diff(pv[anchor_idx]))))
    else:
        anchor_gap = np.nan

    sides = [geo[a]["side"] for a in anchor_idx if geo[a]["side"]]
    flip = bool(sides) and sides[0] != "R"

    # pass 2: pixels for the 36 slices we keep, cached so overlapping triplets
    # (short series) do not decode the same file twice
    px = {}

    def get(k):
        k = int(np.clip(k, 0, N - 1))
        if k not in px:
            try:
                ds = pydicom.dcmread(os.path.join(d, geo[k]["name"]))
                arr = ds.pixel_array.astype(np.float32)
            except Exception:
                return None
            if arr.ndim != 2 or min(arr.shape) < 16:
                return None
            px[k] = F._normalize_slice(arr, geo[k]["spacing"], SIZE, CROP_MM, flip)
        return px[k]

    stack = np.zeros((PER_STUDY, SIZE, SIZE), dtype=np.uint8)
    got = 0
    for t, a in enumerate(anchor_idx):
        mid = get(a)
        if mid is None:
            continue
        for o, k in enumerate((a - 1, a, a + 1)):
            img = mid if o == 1 else get(k)
            if img is None:
                img = mid                      # edge of the series: repeat centre
            stack[t * OFFSETS + o] = (img * 255).astype(np.uint8)
        got += 1
    if got == 0:
        return None, None, None
    return stack, true_gap, anchor_gap


# ---- throughput probe -------------------------------------------------------
# Local benchmarks have been 3.5x optimistic FOUR times on this project, once
# killing a 15 h job at 77% with no index written. Measure on Kaggle, refuse to
# start otherwise.
_t = time.time()
_probed = 0
for uid in studies[:20]:
    build_study(uid)
    _probed += 1
per_study = (time.time() - _t) / max(_probed, 1)
projected = per_study * n / 3600
print(f"PROBE: {per_study:.2f} s/study on Kaggle -> {projected:.1f} h projected",
      flush=True)
if projected > BUDGET_H:
    raise SystemExit(
        f"projected {projected:.1f} h exceeds the {BUDGET_H} h budget. Reduce "
        f"ANCHORS or SIZE -- do not start a job that dies at "
        f"~{BUDGET_H/projected*100:.0f}% with no index.")

cache = np.lib.format.open_memmap("/kaggle/working/slices25d_224_12x3.npy",
                                  mode="w+", dtype=np.uint8,
                                  shape=(n, PER_STUDY, SIZE, SIZE))
present = np.zeros(n, dtype=bool)
true_gaps = np.full(n, np.nan)
anchor_gaps = np.full(n, np.nan)

t0 = time.time()
for i, uid in enumerate(studies):
    stack, tg, ag = build_study(uid)
    if stack is not None:
        cache[i] = stack
        present[i] = True
        true_gaps[i], anchor_gaps[i] = tg, ag
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{n} {el/(i+1):.2f} s/study "
              f"eta {el/(i+1)*(n-i-1)/60:.0f} min", flush=True)

cache.flush()
el = time.time() - t0
print(f"\ncached {n} studies in {el/60:.1f} min ({el/n:.2f} s/study)", flush=True)
print(f"sagittal present: {present.sum()} / {n}", flush=True)

# The premise, measured. If these two numbers are close, the earlier 2.5D runs
# were NOT stacking distant slices and the whole hypothesis is dead here rather
# than after a GPU run.
tg = true_gaps[np.isfinite(true_gaps)]
ag = anchor_gaps[np.isfinite(anchor_gaps)]
if len(tg) and len(ag):
    print(f"\ninter-slice gap, median over {len(tg)} studies:", flush=True)
    print(f"  true neighbours (this cache)   {np.median(tg):.2f} mm  "
          f"[p10 {np.percentile(tg,10):.2f}, p90 {np.percentile(tg,90):.2f}]",
          flush=True)
    print(f"  anchors (what v18/v21 stacked) {np.median(ag):.2f} mm  "
          f"[p10 {np.percentile(ag,10):.2f}, p90 {np.percentile(ag,90):.2f}]",
          flush=True)
    print(f"  ratio {np.median(ag)/max(np.median(tg),1e-6):.1f}x", flush=True)

np.savez_compressed("/kaggle/working/slices25d_index.npz",
                    study_ids=np.array(studies), present=present,
                    true_gap_mm=true_gaps, anchor_gap_mm=anchor_gaps,
                    size=SIZE, anchors=ANCHORS, offsets=OFFSETS,
                    plane=PLANE, layout="anchor*3 + (prev,anchor,next)")
print("saved slices25d_224_12x3.npy + slices25d_index.npz", flush=True)
print("DONE", flush=True)
