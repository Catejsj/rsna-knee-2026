"""The single implementation of the image feature pipeline.

Every kernel that turns DICOMs into a feature vector imports this. Before it
existed, `load_slice` was copy-pasted into four scripts and `pick_series` into
three — which the code graph surfaced and which is the real risk in this
project: training and submission must compute features *identically*. If one
copy drifts, the submission applies weights learned on one representation to a
different one, produces plausible numbers, and errors nowhere.

Constraints this file lives under:
  * numpy + pydicom only — it runs inside an offline Kaggle notebook
  * no I/O beyond reading DICOMs, no globals, no config files
  * pure functions, so the sweep can call it with many settings cheaply

Findings baked in as defaults (see README):
  * series choice: sagittal fluid-sensitive, else coronal fluid-sensitive.
    Covers 100% of training studies (4150 + 257).
  * SIZE 64 / SLICES 24: measured best in the step-1 sweep. Resolution above
    64 made a linear model *worse* (0.6555 -> 0.6498 at 224). That result is
    specific to raw pixels + a linear model, so an encoder should re-test it
    rather than inherit these numbers.
"""
import os

import numpy as np
import pydicom

DEFAULT_SIZE = 64
DEFAULT_SLICES = 24

# Preference order for which series of a study to read.
SERIES_PREFERENCE = (("Sagittal", 1), ("Coronal", 1), (None, None))


def resize(image, size):
    """Nearest-neighbour resize by index selection.

    Deliberately not interpolated: it is a few microseconds, has no
    dependencies beyond numpy, and the sweep showed fine detail does not help
    the linear model anyway.
    """
    h, w = image.shape
    rows = (np.arange(size) * h / size).astype(int)
    cols = (np.arange(size) * w / size).astype(int)
    return image[np.ix_(rows, cols)]


def load_slice(path, size=DEFAULT_SIZE):
    """One DICOM file -> normalized size x size float array, or None.

    MRI has no absolute intensity scale and these studies come from 16 sites,
    so each slice is normalized against its own 1st/99th percentile. Returns
    None rather than raising: a study with one unreadable slice should lose
    that slice, not the whole study.
    """
    try:
        pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
    except Exception:
        return None
    if pixels.ndim != 2 or min(pixels.shape) < 16:
        return None
    lo, hi = np.percentile(pixels, [1, 99])
    if hi <= lo:
        return None
    return resize(np.clip((pixels - lo) / (hi - lo), 0, 1), size)


def pick_series(series_rows):
    """Choose which series of a study to read. `series_rows` is the
    train_series/test_series slice for one StudyInstanceUID."""
    for plane, fluid in SERIES_PREFERENCE:
        if plane is None:
            candidates = series_rows
        else:
            candidates = series_rows[(series_rows.Anatomical_Plane == plane)
                                     & (series_rows.Fluid_Sensitive == fluid)]
        if len(candidates):
            return candidates.iloc[0]
    return None


def pick_series_for_plane(series_rows, plane, fluid=1):
    """Best series for one specific plane: fluid-sensitive if available, else
    any series in that plane, else None.

    Used for multi-plane features. Not every study has all three planes --
    across the 24371 training series the split is Sagittal 9864, Coronal 8609,
    Axial 5898 -- so callers must handle None rather than assume.
    """
    same_plane = series_rows[series_rows.Anatomical_Plane == plane]
    preferred = same_plane[same_plane.Fluid_Sensitive == fluid]
    if len(preferred):
        return preferred.iloc[0]
    if len(same_plane):
        return same_plane.iloc[0]
    return None


def load_stack(series_dir, n_slices=DEFAULT_SLICES, size=DEFAULT_SIZE,
               span=(0.15, 0.85)):
    """Evenly spaced slices from one series directory, as a list of arrays.

    `span` trims the ends of the series, where slices are mostly off the joint.
    Returns [] if the directory is unreadable or too few slices survive.
    """
    try:
        files = sorted(os.listdir(series_dir))
    except OSError:
        return []
    if len(files) < 3:
        return []
    take = min(n_slices, len(files))
    positions = np.linspace(span[0], span[1], take) * (len(files) - 1)
    slices = [load_slice(os.path.join(series_dir, files[int(p)]), size)
              for p in positions]
    return [s for s in slices if s is not None]


def aggregate(stack, method="mean"):
    """Collapse a slice stack into one feature vector.

    'meanstd' appends per-pixel variation across slices; the sweep measured it
    at +0.003 for double the dimensions, so 'mean' stays the default.
    """
    arr = np.stack(stack)
    if method == "meanstd":
        return np.concatenate([arr.mean(0).ravel(), arr.std(0).ravel()])
    return arr.mean(0).ravel()


def study_features(root, study_uid, series_uid, n_slices=DEFAULT_SLICES,
                   size=DEFAULT_SIZE, method="mean", min_slices=2):
    """Full path for one study: directory -> feature vector, or None."""
    stack = load_stack(os.path.join(root, study_uid, series_uid),
                       n_slices=n_slices, size=size)
    if len(stack) < min_slices:
        return None
    return aggregate(stack, method)


# ------------------------------------------------------- geometry-aware path
# The functions above ignore DICOM geometry entirely: slices are ordered by
# filename (SOP UID order, i.e. arbitrary), every image is stretched to a fixed
# pixel count regardless of how much anatomy it covers, and left and right knees
# are fed in mirrored. The three below fix that.
#
# Laterality matters most here. "Medial" is the inner side of the joint, which
# is the opposite side of the image for a left versus a right knee, so without
# flipping, a compartment-specific label is attached to mirrored anatomy and
# averages out. Our compartment labels are the weakest ones we have.

CROP_MM = 160.0    # acquired field of view runs ~70-320 mm, median ~160


def _slice_geometry(ds):
    """(normal-axis position, pixel spacing mm, laterality) from a DICOM.

    Every field is optional: returns None where a tag is absent so callers can
    count coverage rather than silently assume it worked.
    """
    pos = spacing = side = None
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        pos = float(np.dot(ipp, normal))
    except Exception:
        pass
    try:
        spacing = float(np.asarray(ds.PixelSpacing, dtype=float)[0])
    except Exception:
        pass
    for tag in ("ImageLaterality", "Laterality"):
        v = getattr(ds, tag, None)
        if v:
            side = str(v).strip().upper()[:1]
            break
    if side not in ("L", "R"):
        # tag missing (half the corpus) or a value like "B" -- fall back to
        # geometry, which agrees with the tag on 97.1% of the studies that
        # have one and is available on 100%
        side = infer_laterality(ds)
    return pos, spacing, side


def infer_laterality(ds):
    """Which knee, from patient coordinates rather than the Laterality tag.

    DICOM patient space is LPS: +x points to the patient's LEFT. A knee is well
    off the midline, so the sign of the image-centre x separates a left knee
    from a right one.

    This exists because the `Laterality` tag survives on only 49.7% of these
    studies, and flipping half a corpus is worse than flipping none of it --
    it creates two conventions instead of one. ImagePositionPatient and
    ImageOrientationPatient are present on 100%.

    Returns "L", "R", or None. Verify against the tagged half before trusting
    it on the untagged half; do not assume.
    """
    try:
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        spacing = np.asarray(ds.PixelSpacing, dtype=float)
        rows, cols = int(ds.Rows), int(ds.Columns)
    except Exception:
        return None
    # ipp is the centre of the first voxel; step to the middle of the frame
    row_dir, col_dir = iop[:3], iop[3:]
    centre = (ipp
              + row_dir * spacing[1] * (cols - 1) / 2.0
              + col_dir * spacing[0] * (rows - 1) / 2.0)
    if abs(centre[0]) < 1e-6:
        return None
    return "L" if centre[0] > 0 else "R"


def load_series_with_geometry(series_dir, max_slices=40):
    """Read a series once, keeping the geometry needed for normalization.

    Returns a list of dicts sorted by filename (the caller decides real order).
    """
    try:
        files = sorted(os.listdir(series_dir))
    except OSError:
        return []
    if len(files) < 3:
        return []
    if len(files) > max_slices:
        idx = np.linspace(0, len(files) - 1, max_slices).astype(int)
        files = [files[i] for i in idx]

    out = []
    for name in files:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, name))
            pixels = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if pixels.ndim != 2 or min(pixels.shape) < 16:
            continue
        pos, spacing, side = _slice_geometry(ds)
        out.append({"pixels": pixels, "pos": pos, "spacing": spacing, "side": side})
    return out


def _normalize_slice(pixels, spacing, size, crop_mm, flip):
    """Physical centre-crop -> intensity normalize -> resize -> optional flip."""
    if crop_mm and spacing:
        half = int(round((crop_mm / spacing) / 2))
        cy, cx = pixels.shape[0] // 2, pixels.shape[1] // 2
        y0, y1 = max(cy - half, 0), min(cy + half, pixels.shape[0])
        x0, x1 = max(cx - half, 0), min(cx + half, pixels.shape[1])
        if y1 - y0 >= 16 and x1 - x0 >= 16:
            pixels = pixels[y0:y1, x0:x1]
    lo, hi = np.percentile(pixels, [1, 99])
    if hi <= lo:
        return None
    img = resize(np.clip((pixels - lo) / (hi - lo), 0, 1), size)
    return img[:, ::-1].copy() if flip else img


def geometric_slice_stack(slices, n_slices=DEFAULT_SLICES, size=DEFAULT_SIZE,
                          crop_mm=CROP_MM, order="physical", flip_to="R"):
    """Geometry-normalized slices as a [k, size, size] array, or None.

    order   -- "physical" sorts by position along the slice normal;
               "filename" keeps the arbitrary order the files came in.
    crop_mm -- centre-crop to this many mm of anatomy before resizing;
               None keeps the old stretch-the-whole-frame behaviour.
    flip_to -- mirror left knees so every study faces the same way;
               None disables.

    An encoder needs the individual slices; `geometric_features` just averages
    what comes out of here.
    """
    if not slices:
        return None
    ordered = slices
    if order == "physical" and all(s["pos"] is not None for s in slices):
        ordered = sorted(slices, key=lambda s: s["pos"])

    span = np.linspace(0.15, 0.85, min(n_slices, len(ordered))) * (len(ordered) - 1)
    picked = [ordered[int(p)] for p in span]

    sides = [s["side"] for s in picked if s["side"]]
    flip = bool(flip_to) and bool(sides) and sides[0] != flip_to

    imgs = [_normalize_slice(s["pixels"], s["spacing"], size, crop_mm, flip)
            for s in picked]
    imgs = [i for i in imgs if i is not None]
    if len(imgs) < 2:
        return None
    return np.stack(imgs)


def geometric_features(slices, n_slices=DEFAULT_SLICES, size=DEFAULT_SIZE,
                       crop_mm=CROP_MM, order="physical", flip_to="R"):
    """One averaged feature vector with the geometry corrections switched on."""
    stack = geometric_slice_stack(slices, n_slices=n_slices, size=size,
                                  crop_mm=crop_mm, order=order, flip_to=flip_to)
    if stack is None:
        return None
    return stack.mean(0).ravel()


PLANES = ("Sagittal", "Coronal", "Axial")


def study_features_multiplane_geo(root, study_uid, series_rows, planes=PLANES,
                                  n_slices=DEFAULT_SLICES, size=DEFAULT_SIZE,
                                  crop_mm=CROP_MM, order="physical", flip_to="R",
                                  max_read=32):
    """Multi-plane features with the geometry corrections applied.

    Same contract as study_features_multiplane -- one block per plane, NaN for
    a plane the study lacks, returns (vector, n_planes_found) -- but each block
    is built from geometry-normalized slices: ordered along the slice normal,
    centre-cropped to a fixed physical extent, and mirrored so every knee faces
    the same way.
    """
    block = size * size
    parts, found = [], 0
    for plane in planes:
        row = pick_series_for_plane(series_rows, plane)
        vec = None
        if row is not None:
            slices = load_series_with_geometry(
                os.path.join(root, study_uid, row.SeriesInstanceUID),
                max_slices=max_read)
            if len(slices) >= 3:
                vec = geometric_features(slices, n_slices=n_slices, size=size,
                                         crop_mm=crop_mm, order=order,
                                         flip_to=flip_to)
        if vec is None:
            parts.append(np.full(block, np.nan, dtype=np.float32))
        else:
            parts.append(vec.astype(np.float32))
            found += 1
    return np.concatenate(parts), found


def study_features_multiplane(root, study_uid, series_rows, planes=PLANES,
                              n_slices=DEFAULT_SLICES, size=DEFAULT_SIZE,
                              method="mean"):
    """One vector per plane, concatenated in `planes` order.

    A plane the study does not have becomes NaN rather than zeros, so the
    caller can impute it with the column mean. Zeros would be a real, wrong
    value -- a black image -- and the model would learn from it.

    Returns (vector, n_planes_found).
    """
    block = size * size * (2 if method == "meanstd" else 1)
    parts, found = [], 0
    for plane in planes:
        row = pick_series_for_plane(series_rows, plane)
        vec = None
        if row is not None:
            vec = study_features(root, study_uid, row.SeriesInstanceUID,
                                 n_slices=n_slices, size=size, method=method)
        if vec is None:
            parts.append(np.full(block, np.nan, dtype=np.float32))
        else:
            parts.append(vec.astype(np.float32))
            found += 1
    return np.concatenate(parts), found
