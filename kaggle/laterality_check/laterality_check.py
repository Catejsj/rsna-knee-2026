"""Can we recover which knee it is from geometry, and is the recovery right?

The `Laterality` tag survives on 49.7% of studies. Flipping only those made
things worse, because half-normalized is two conventions rather than one. The
proposed fix infers the side from patient coordinates instead (LPS: +x is the
patient's left), which are present on 100%.

This does not assume the inference works: it measures it. Where the tag
exists, we have ground truth for the inference itself, so:

    agreement on the tagged 49.7%  ->  whether to trust it on the other 50.3%

Anything below ~95% and the inference is not usable; a systematic disagreement
(e.g. always wrong for one vendor) is more interesting than a random one, so
the mismatches are broken down rather than just counted.

Metadata only: `stop_before_pixels=True` skips the image data, so this reads
one file per study and runs in minutes rather than the 74 the geometry sweep
took.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import pydicom

import glob
import sys

sys.path.insert(0, os.path.dirname(
    glob.glob("/kaggle/input/**/rsna_features.py", recursive=True)[0]))
import rsna_features as F

COMP = os.path.dirname(glob.glob("/kaggle/input/**/train_series.csv", recursive=True)[0])
ser = pd.read_csv(os.path.join(COMP, "train_series.csv"))
by_study = {k: v for k, v in ser.groupby("StudyInstanceUID")}
root = os.path.join(COMP, "train_series")
print(f"studies: {len(by_study)}", flush=True)

rows = []
t0 = time.time()
for i, (uid, srows) in enumerate(by_study.items()):
    pick = F.pick_series_for_plane(srows, "Sagittal")
    if pick is None:
        continue
    d = os.path.join(root, uid, pick.SeriesInstanceUID)
    try:
        files = sorted(os.listdir(d))
    except OSError:
        continue
    if not files:
        continue
    # read a middle slice: ends of a series can be off the joint
    try:
        ds = pydicom.dcmread(os.path.join(d, files[len(files) // 2]),
                             stop_before_pixels=True)
    except Exception:
        continue

    tag = None
    for t in ("ImageLaterality", "Laterality"):
        v = getattr(ds, t, None)
        if v:
            tag = str(v).strip().upper()[:1]
            break
    inferred = F.infer_laterality(ds)
    centre_x = None
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        sp = np.asarray(ds.PixelSpacing, dtype=float)
        centre_x = float((ipp + iop[:3] * sp[1] * (int(ds.Columns) - 1) / 2
                          + iop[3:] * sp[0] * (int(ds.Rows) - 1) / 2)[0])
    except Exception:
        pass

    rows.append({
        "uid": uid, "tag": tag, "inferred": inferred, "centre_x": centre_x,
        "manufacturer": str(getattr(ds, "Manufacturer", "") or "")[:24],
    })
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(by_study)} {(time.time()-t0)/(i+1):.3f} s/study",
              flush=True)

df = pd.DataFrame(rows)
print(f"\nread {len(df)} studies in {(time.time()-t0)/60:.1f} min", flush=True)

print("\n" + "=" * 70)
print("COVERAGE")
print("=" * 70)
print(f"  Laterality tag present : {df.tag.notna().mean()*100:5.1f}%")
print(f"  inference available    : {df.inferred.notna().mean()*100:5.1f}%")

both = df[df.tag.notna() & df.inferred.notna()]
print("\n" + "=" * 70)
print(f"AGREEMENT on the {len(both)} studies where the tag exists")
print("=" * 70)
agree = (both.tag == both.inferred).mean()
print(f"  agreement: {agree*100:.2f}%")
print(f"  tag says      L/R : {Counter(both.tag)}")
print(f"  inference says L/R: {Counter(both.inferred)}")

bad = both[both.tag != both.inferred]
if len(bad):
    print(f"\n  {len(bad)} mismatches: are they systematic?")
    print("  by manufacturer:")
    for m, c in Counter(bad.manufacturer).most_common(6):
        total = (both.manufacturer == m).sum()
        print(f"    {m or '(blank)':<26}{c:>5} of {total:<6} "
              f"({c/max(total,1)*100:.1f}%)")
    print(f"  |centre_x| on mismatches: median {bad.centre_x.abs().median():.1f} mm")
    print(f"  |centre_x| on matches   : median "
          f"{both[both.tag == both.inferred].centre_x.abs().median():.1f} mm")

print("\n" + "=" * 70)
print("WHAT THE INFERENCE SAYS ABOUT THE WHOLE CORPUS")
print("=" * 70)
print(f"  inferred side over all studies: {Counter(df.inferred.dropna())}")
print(f"  |centre_x| distribution: "
      f"p05 {df.centre_x.abs().quantile(.05):.0f}  "
      f"median {df.centre_x.abs().median():.0f}  "
      f"p95 {df.centre_x.abs().quantile(.95):.0f} mm")
near_mid = (df.centre_x.abs() < 20).mean()
print(f"  studies within 20 mm of midline (ambiguous): {near_mid*100:.1f}%")

df.to_csv("/kaggle/working/laterality.csv", index=False)
print("\nwrote /kaggle/working/laterality.csv")
print("\nverdict: >=95% agreement => trust the inference and flip all 4407.")
print("DONE", flush=True)
