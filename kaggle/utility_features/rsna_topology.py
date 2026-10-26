# GENERATED FILE - DO NOT EDIT.
# Copied from src/ by tools/sync_utility.py.
# Edit the source, then re-run the sync. Source sha256: 27c25ee10032dbcb
"""Topological features for knee MRI slices: Betti curves, numpy + scipy only.

Why not a TDA library: the submission notebook runs with internet disabled, and
`gudhi` / `giotto-tda` are not standard on Kaggle. Shipping wheels is possible
but is a dependency risk on the one code path that must not fail. In 2D the
quantities we want are computable directly:

    b0  = connected components of the thresholded image  (scipy.ndimage.label)
    chi = Euler characteristic, countable from pixels/edges/2x2 blocks
    b1  = b0 - chi                                       (holds in 2D)

Sweeping the threshold gives **Betti curves**: how many blobs, and how many
holes, exist at each brightness level. That is the same information a
persistence diagram summarizes, at a fraction of the cost and with no
dependencies.

Vectorization is deliberately small. Every capacity increase measured on this
project has been punished: 224px lost to 64px, PCA-300 lost to PCA-100,
no-PCA lost badly, and persistence images would add thousands of columns to a
4407-row problem. Betti curves give ~30-60 numbers per plane.

Why this should carry signal for knees: effusion and Baker's cysts are bright
fluid pockets, which are exactly H0 features in a superlevel filtration;
cartilage and meniscal texture is a mesh of small loops, which is H1.
"""
import numpy as np
from scipy import ndimage

# 4-connectivity for components, matching the cubical complex used for chi
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def euler_characteristic(mask):
    """chi = V - E + F for the cubical complex of a binary image.

    V = foreground pixels, E = adjacent foreground pairs (4-neighbour),
    F = 2x2 blocks entirely foreground.
    """
    v = int(mask.sum())
    if v == 0:
        return 0
    e = int((mask[:, :-1] & mask[:, 1:]).sum() + (mask[:-1, :] & mask[1:, :]).sum())
    f = int((mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]).sum())
    return v - e + f


def betti_numbers(mask):
    """(b0, b1) for a 2D binary image."""
    if not mask.any():
        return 0, 0
    b0 = int(ndimage.label(mask, structure=_CROSS)[1])
    return b0, b0 - euler_characteristic(mask)


def betti_curves(image, n_thresholds=16, direction="superlevel"):
    """Betti curves over a threshold sweep of a [0,1] image.

    direction "superlevel" thresholds `image >= t`: bright structures, which
    is where fluid lives on the fluid-sensitive sequences we select.
    "sublevel" (`image <= t`) picks out dark structures instead.

    Returns (b0_curve, b1_curve), each length n_thresholds.
    """
    ts = np.linspace(0.05, 0.95, n_thresholds)
    b0s = np.empty(n_thresholds, dtype=np.float32)
    b1s = np.empty(n_thresholds, dtype=np.float32)
    for i, t in enumerate(ts):
        mask = image >= t if direction == "superlevel" else image <= t
        b0s[i], b1s[i] = betti_numbers(mask)
    return b0s, b1s


def slice_features(image, n_thresholds=16):
    """Both filtration directions for one slice -> 4 * n_thresholds numbers."""
    sup0, sup1 = betti_curves(image, n_thresholds, "superlevel")
    sub0, sub1 = betti_curves(image, n_thresholds, "sublevel")
    return np.concatenate([sup0, sup1, sub0, sub1])


def stack_features(stack, n_thresholds=16):
    """A slice stack -> pooled topological descriptor.

    Mean and max across slices: a finding present on three slices out of
    twenty-four is diluted by the mean but survives the max, which is the same
    reason more slices helped the pixel model.
    """
    per_slice = np.stack([slice_features(s, n_thresholds) for s in stack])
    return np.concatenate([per_slice.mean(0), per_slice.max(0)]).astype(np.float32)


FEATURES_PER_PLANE = 8 * 16   # 4 curves x 16 thresholds x (mean, max)
