"""Asaad-style persistent homology: ULBP texture landmarks -> Rips filtration.

Follows the method in Aras Asaad's DPhil thesis (Buckingham, 2020), sections
5.1-5.2 and the mammography application in 6.3:

    1. LBP-code every pixel from its 8 neighbours
    2. keep *uniform* codes (<=2 circular bit transitions), grouped into
       G_lambda geometries by the number of 1s
    3. take the (x,y) positions of those pixels as a landmark point cloud
    4. filter by increasing distance threshold
    5. record B0 and B1 at each threshold -- the "persistent barcode"
    6. bin into a fixed-length vector

This filters over **spatial distance between texture landmarks**, which is a
different question from `topology.py`: that one filters over image intensity
and asks what shapes exist at each brightness. Asaad's hypothesis is that
disease distorts the natural spatial texture of tissue, which is what this
measures.

**Deviation from the thesis, deliberate.** The thesis builds Vietoris-Rips
complexes directly, which is O(n^2) in memory and expensive for B1. In 2D the
Euclidean minimum spanning tree is a subgraph of the Delaunay triangulation, so
computing B0 over Delaunay edges is *exactly* Rips B0 -- not an approximation --
because single-linkage merging only ever uses MST edges. B1 then follows from
V - E + F on the same complex. Scipy only, so it runs offline on Kaggle.

Second deviation: the thesis keeps all 8 rotations of each geometry separate
(800 dims per geometry). Every capacity increase measured on this project has
been punished, so rotations are merged and only a few geometries are used.
"""
import numpy as np
from scipy.spatial import Delaunay

# 8 neighbours, clockwise from top-left: the order the thesis specifies
_OFF = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def lbp_code(image):
    """8-bit LBP code per pixel. Border pixels are excluded (returned as -1)."""
    h, w = image.shape
    code = np.zeros((h, w), dtype=np.int16)
    centre = image[1:-1, 1:-1]
    for bit, (dy, dx) in enumerate(_OFF):
        nb = image[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code[1:-1, 1:-1] |= ((nb >= centre).astype(np.int16) << bit)
    code[0, :] = code[-1, :] = code[:, 0] = code[:, -1] = -1
    return code


def _transitions(c):
    """Circular 0<->1 transitions in an 8-bit code."""
    bits = [(c >> i) & 1 for i in range(8)]
    return sum(bits[i] != bits[(i + 1) % 8] for i in range(8))


# uniform codes (<=2 transitions), indexed by popcount == the G_lambda geometry
UNIFORM_BY_LAMBDA = {}
for _c in range(256):
    if _transitions(_c) <= 2:
        UNIFORM_BY_LAMBDA.setdefault(bin(_c).count("1"), []).append(_c)


def landmarks(image, lam, max_points=600, rng=None):
    """(x,y) positions of uniform-LBP pixels in geometry G_lambda.

    Subsampled to `max_points` because Delaunay cost grows with n and, more
    importantly, so the descriptor does not silently encode "how many landmarks
    there were", which is a texture-density feature, not a topological one.
    """
    codes = lbp_code(image)
    wanted = UNIFORM_BY_LAMBDA.get(lam, [])
    mask = np.isin(codes, wanted)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    if len(xs) > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    return np.column_stack([xs, ys]).astype(np.float64)


def betti_curves(points, thresholds):
    """B0 and B1 of the Delaunay-Rips filtration at each threshold.

    B0 is exact Rips B0 (the MST is a Delaunay subgraph). B1 = B0 - chi with
    chi = V - E + F over simplices whose longest edge is within threshold.
    """
    n = len(points)
    b0 = np.full(len(thresholds), float(n), dtype=np.float32)
    b1 = np.zeros(len(thresholds), dtype=np.float32)
    if n < 4:
        return b0, b1
    try:
        tri = Delaunay(points)
    except Exception:
        return b0, b1

    simplices = tri.simplices
    # unique edges with their lengths
    e = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    elen = np.linalg.norm(points[e[:, 0]] - points[e[:, 1]], axis=1)
    order = np.argsort(elen)
    e, elen = e[order], elen[order]

    # a triangle enters the complex when its longest edge does
    tside = np.stack([
        np.linalg.norm(points[simplices[:, 0]] - points[simplices[:, 1]], axis=1),
        np.linalg.norm(points[simplices[:, 1]] - points[simplices[:, 2]], axis=1),
        np.linalg.norm(points[simplices[:, 0]] - points[simplices[:, 2]], axis=1)])
    tmax = tside.max(0)

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    comp, ei = n, 0
    for k, t in enumerate(thresholds):
        while ei < len(elen) and elen[ei] <= t:
            ra, rb = find(e[ei, 0]), find(e[ei, 1])
            if ra != rb:
                parent[ra] = rb
                comp -= 1
            ei += 1
        n_edges = ei
        n_faces = int((tmax <= t).sum())
        b0[k] = comp
        b1[k] = comp - (n - n_edges + n_faces)   # b1 = b0 - chi
    return b0, b1


def slice_features(image, lambdas=(1, 4, 7), n_thresholds=24, max_dist=60.0,
                   max_points=600):
    """Betti curves for several ULBP geometries -> one vector for a slice."""
    ts = np.linspace(1.0, max_dist, n_thresholds)
    out = []
    for lam in lambdas:
        pts = landmarks(image, lam, max_points=max_points)
        if pts is None:
            out.append(np.zeros(2 * n_thresholds, dtype=np.float32))
            continue
        b0, b1 = betti_curves(pts, ts)
        # normalize B0 by point count: otherwise this encodes landmark density
        out.append(np.concatenate([b0 / len(pts), b1 / len(pts)]).astype(np.float32))
    return np.concatenate(out)


def stack_features(stack, **kw):
    """Slice stack -> pooled descriptor (mean and max across slices)."""
    per = np.stack([slice_features(s, **kw) for s in stack])
    return np.concatenate([per.mean(0), per.max(0)]).astype(np.float32)
