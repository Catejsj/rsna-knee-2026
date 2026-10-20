"""Eikonal solver (fast marching) for physically-scaled geometry on MRI volumes.

    |grad u(x)| = 1 / f(x)

`u` is the arrival time of a wavefront and `f` the local speed. With f == 1 the
solution is Euclidean distance from the source; with f driven by image content
it is *travel time*, which bends around slow tissue. That is the useful part:
travel time is a geometric measurement of the image, in millimetres.

**Why this is worth trying when topology and persistence both failed.** Those
were extra descriptors of the same thing DINOv2 already encodes, and both landed
inside the noise. This is different in kind: the whole pipeline is 2D and
unitless -- DINOv2 embeds each slice independently and the attention head only
pools them, so nothing anywhere computes a 3D quantity or a millimetre. Anything
this produces is information the encoder structurally cannot derive. "Add
information, not parameters" is the rule that separated every win on this project
from every failure.

**No scikit-fmm.** Submission notebooks run with internet disabled, the same
constraint that ruled out gudhi in topology.py. Fast marching is Dijkstra on a
grid with a Godunov upwind update, so it is written here in numpy + heapq.

Anisotropic spacing is a first-class argument, not a convenience: knee MRI voxels
are nothing like cubic (in-plane ~0.3mm, between slices ~3mm), and treating them
as cubic would make every distance wrong by an order of magnitude in one axis.
Pass real spacing in millimetres and the output is in millimetres.
"""
import heapq

import numpy as np

FAR, TRIAL, KNOWN = 0, 1, 2


def _solve_quadratic(vals, spacings, speed):
    """Godunov upwind update at one voxel.

    Solves sum_d ((u - a_d)/h_d)^2 = 1/f^2 over the subset of axes that actually
    constrain u. Axes are added in increasing order of arrival time; an axis
    whose neighbour arrives *later* than the candidate u cannot influence it, so
    the smallest consistent solution is the correct one.
    """
    order = np.argsort(vals)
    a = np.asarray(vals, dtype=np.float64)[order]
    h = np.asarray(spacings, dtype=np.float64)[order]
    rhs = 1.0 / (speed * speed)

    A = B = C = 0.0
    for k in range(len(a)):
        if not np.isfinite(a[k]):
            break
        w = 1.0 / (h[k] * h[k])
        A += w
        B += -2.0 * a[k] * w
        C += a[k] * a[k] * w
        disc = B * B - 4.0 * A * (C - rhs)
        if disc < 0:
            continue
        u = (-B + np.sqrt(disc)) / (2.0 * A)
        # valid when u dominates every axis used and does not reach the next one
        if u >= a[k] - 1e-12 and (k + 1 >= len(a) or not np.isfinite(a[k + 1])
                                  or u <= a[k + 1] + 1e-12):
            return u
    return np.inf


def fast_marching(speed, seeds, spacing=None, mask=None, max_time=np.inf):
    """Solve the Eikonal equation on a regular grid.

    speed    : array of positive local speeds, any dimension
    seeds    : bool array, True where u = 0
    spacing  : physical size of a voxel per axis (mm). Defaults to ones.
    mask     : optional bool array of voxels the front may enter
    max_time : stop once the front passes this arrival time

    Returns `u`, with np.inf wherever the front never arrived.
    """
    speed = np.asarray(speed, dtype=np.float64)
    nd = speed.ndim
    spacing = np.ones(nd) if spacing is None else np.asarray(spacing, float)
    if spacing.size != nd:
        raise ValueError(f"spacing needs {nd} values, got {spacing.size}")
    if mask is None:
        mask = np.ones(speed.shape, dtype=bool)

    u = np.full(speed.shape, np.inf)
    state = np.where(mask, FAR, KNOWN).astype(np.int8)  # blocked reads as done
    u[seeds & mask] = 0.0
    state[seeds & mask] = TRIAL

    heap = [(0.0, int(i)) for i in np.flatnonzero(seeds & mask)]
    heapq.heapify(heap)
    shape = speed.shape

    while heap:
        t, flat = heapq.heappop(heap)
        if t > u.flat[flat] + 1e-12 or state.flat[flat] == KNOWN:
            continue
        if t > max_time:
            break
        state.flat[flat] = KNOWN
        idx = np.unravel_index(flat, shape)

        for ax in range(nd):
            for step in (-1, 1):
                nb = list(idx)
                nb[ax] += step
                if not (0 <= nb[ax] < shape[ax]):
                    continue
                nb = tuple(nb)
                if state[nb] == KNOWN:
                    continue
                # best known arrival along each axis around this neighbour
                vals, hs = [], []
                for a2 in range(nd):
                    best = np.inf
                    for s2 in (-1, 1):
                        p = list(nb)
                        p[a2] += s2
                        if 0 <= p[a2] < shape[a2]:
                            p = tuple(p)
                            if state[p] == KNOWN and u[p] < best:
                                best = u[p]
                    vals.append(best)
                    hs.append(spacing[a2])
                f = speed[nb]
                if f <= 0:
                    continue
                cand = _solve_quadratic(vals, hs, f)
                if cand < u[nb]:
                    u[nb] = cand
                    state[nb] = TRIAL
                    heapq.heappush(heap, (float(cand),
                                          int(np.ravel_multi_index(nb, shape))))
    return u


def speed_from_intensity(vol, mode="edge", eps=0.05):
    """Turn a normalized [0,1] volume into a speed field.

    "edge"   -- slow at strong gradients, so the front is held back by tissue
                boundaries. This is the segmentation-flavoured choice: arrival
                time then measures how far a structure extends before a wall.
    "bright" -- fast in bright voxels. On fluid-sensitive sequences that means
                the front runs along fluid, so arrival time measures how far
                fluid has spread from its source -- which is the difference
                between an effusion sitting in the joint and a Baker's cyst
                that has tracked posteromedially.
    """
    vol = np.asarray(vol, dtype=np.float64)
    if mode == "bright":
        return np.clip(vol, eps, 1.0)
    g = np.zeros_like(vol)
    for ax in range(vol.ndim):
        d = np.gradient(vol, axis=ax)
        g += d * d
    g = np.sqrt(g)
    if g.max() > 0:
        g /= g.max()
    return np.clip(1.0 - g, eps, 1.0)


def _selftest():
    """Constant speed must reproduce Euclidean distance; a wall must bend it."""
    n = 81
    seeds = np.zeros((n, n), dtype=bool)
    seeds[40, 40] = True
    u = fast_marching(np.ones((n, n)), seeds)
    yy, xx = np.mgrid[0:n, 0:n]
    exact = np.sqrt((yy - 40.0) ** 2 + (xx - 40.0) ** 2)
    err = np.abs(u - exact)[exact <= 30]
    print(f"  2D constant speed : max err {err.max():.3f} px, "
          f"mean {err.mean():.3f} px  (first-order FMM, a few % is expected)")

    sp = (1.0, 4.0)
    u = fast_marching(np.ones((n, n)), seeds, spacing=sp)
    exact = np.sqrt(((yy - 40.0) * sp[0]) ** 2 + ((xx - 40.0) * sp[1]) ** 2)
    err = np.abs(u - exact)[exact <= 30]
    print(f"  anisotropic 1x4mm : max err {err.max():.3f} mm, "
          f"mean {err.mean():.3f} mm")

    m = 41
    s3 = np.zeros((m, m, m), dtype=bool)
    s3[20, 20, 20] = True
    u3 = fast_marching(np.ones((m, m, m)), s3, spacing=(3.0, 0.5, 0.5))
    zz, yy3, xx3 = np.mgrid[0:m, 0:m, 0:m]
    ex3 = np.sqrt(((zz - 20) * 3.0) ** 2 + ((yy3 - 20) * .5) ** 2
                  + ((xx3 - 20) * .5) ** 2)
    e3 = np.abs(u3 - ex3)[ex3 <= 8]
    print(f"  3D anisotropic    : max err {e3.max():.3f} mm, "
          f"mean {e3.mean():.3f} mm")

    # A wall spanning rows 30..50 at cols 39..41, source left of it, target
    # right. The straight line is blocked, so the front must round the top
    # corner at (29, 39) and come back down -- compare against that geodesic,
    # not against an arbitrary multiple of the straight-line distance.
    speed = np.ones((n, n))
    speed[30:51, 39:42] = 1e-3
    seeds2 = np.zeros((n, n), dtype=bool)
    seeds2[40, 20] = True
    u = fast_marching(speed, seeds2)
    straight = 40.0
    detour = (np.hypot(40 - 29, 39 - 20) + 3.0 + np.hypot(40 - 29, 60 - 42))
    got = u[40, 60]
    ok = got > straight * 1.05 and abs(got - detour) < 0.06 * detour
    print(f"  wall detour       : geodesic {got:.1f} px vs straight "
          f"{straight:.0f}, corner path {detour:.1f} -> "
          f"{'bends correctly' if ok else 'FAILS'}")

    ring = np.zeros((n, n), dtype=bool)
    ring[35:46, 35:46] = True
    u = fast_marching(np.ones((n, n)), ring)
    print(f"  seeded region     : u==0 on {int((u == 0).sum())} voxels "
          f"(expected {11*11})")


if __name__ == "__main__":
    print("eikonal self-test")
    _selftest()
