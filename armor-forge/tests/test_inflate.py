import numpy as np

from armor_forge.inflate import adjacency, inflate, subdivide, vertex_normals, weld
from armor_forge.sheet import ViewMask


def disc_view(r, cz=1.0, px_per_m=250):
    n = int(2 * px_per_m)
    u = -1 + (np.arange(n) + 0.5) / px_per_m
    z = 2 - (np.arange(n) + 0.5) / px_per_m
    arr = (u[None, :] ** 2 + (z[:, None] - cz) ** 2) <= r * r
    return ViewMask(arr, col0=0.0, row0=0.0, pxu=px_per_m, pxv=px_per_m)


def cube_sphere(r, centre, levels=3):
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    f = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    f = [list(q) for q in f]
    for _ in range(levels):
        v, f = subdivide(v, f)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * r + centre
    return v, f


def test_subdivide_counts():
    v, f = cube_sphere(0.2, [0, 0, 1], levels=0)
    v2, f2 = subdivide(v, f)
    assert len(f2) == 4 * len(f) and len(v2) == len(v) + 12 + 6  # edge mids + face centres


def test_weld_merges_duplicates():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], float)
    v2, f2, inv = weld(v, [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert len(v2) == 6 and len(f2) == 2


def test_normals_point_outward_on_sphere():
    v, f = cube_sphere(0.2, np.array([0, 0, 1.0]))
    n = vertex_normals(v, f)
    radial = (v - [0, 0, 1.0]) / 0.2
    # winding of the toy cube may be inward; accept either, but it must be consistent
    d = (n * radial).sum(axis=1)
    assert (d > 0.9).all() or (d < -0.9).all()


def test_sphere_inflates_to_silhouette_radius():
    v, f = cube_sphere(0.2, np.array([0, 0, 1.0]))
    n = vertex_normals(v, f)
    if ((n * (v - [0, 0, 1.0])).sum(axis=1) < 0).all():
        f = [q[::-1] for q in f]  # make outward
    dv = disc_view(0.3)
    views = {"front": dv, "back": dv, "side": dv, "side_noarm": dv}
    nv, t = inflate(v, f, views, gap=0.0, smooth_iters=0, t_max=0.5)
    # The limit is the visual hull (two perpendicular discs), not a sphere: every vertex must stay
    # inside both silhouettes, and the binding view should be pushed right up to its edge.
    rf = np.hypot(nv[:, 0], nv[:, 2] - 1.0)
    rs = np.hypot(nv[:, 1], nv[:, 2] - 1.0)
    assert max(rf.max(), rs.max()) < 0.305
    assert np.abs(np.maximum(rf, rs) - 0.3).max() < 0.012


def test_adjacency_symmetric():
    v, f = cube_sphere(0.2, np.array([0, 0, 1.0]), levels=1)
    nb = adjacency(f, len(v))
    for i, n in enumerate(nb):
        for j in n:
            assert i in nb[j]
