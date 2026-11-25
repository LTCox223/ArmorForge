import collections

import numpy as np

from armor_forge.carve import carve_envelope, make_grid, surface_nets, visual_hull
from armor_forge.sheet import ViewMask


def disc_view(r=0.5, cz=1.0, px_per_m=200):
    n = int(2 * px_per_m)
    u = -1 + (np.arange(n) + 0.5) / px_per_m
    z = 2 - (np.arange(n) + 0.5) / px_per_m
    arr = (u[None, :] ** 2 + (z[:, None] - cz) ** 2) <= r * r
    return ViewMask(arr, col0=0.0, row0=0.0, pxu=px_per_m, pxv=px_per_m)


def test_viewmask_sample():
    v = disc_view()
    assert v.sample(0.0, 1.0)
    assert not v.sample(0.9, 1.0)
    assert not v.sample(5.0, 1.0)  # out of bounds is False
    assert v.sample(np.array([[0.0], [0.9]]), np.array([[1.0, 1.9]])).shape == (2, 2)


def test_hull_of_disc_views_is_a_bicylinder():
    v = disc_view()
    views = {"front": v, "back": v, "side": v, "side_noarm": v}
    grid = make_grid(lo=(-0.6, -0.6, 0.4), hi=(0.6, 0.6, 1.6), voxel=0.01)
    hull = visual_hull(grid, views)
    expect = 16 / 3 * 0.5 ** 3  # front/back share an axis, so two perpendicular cylinders
    assert abs(hull.sum() * 0.01 ** 3 - expect) / expect < 0.03


def test_surface_nets_is_closed_and_outward():
    grid = make_grid(lo=(-0.5, -0.5, -0.5), hi=(0.5, 0.5, 0.5), voxel=0.02)
    xs, ys, zs = grid.axes()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    occ = X ** 2 + Y ** 2 + Z ** 2 <= 0.3 ** 2
    verts, quads = surface_nets(occ, grid)
    edges = collections.Counter()
    for q in quads:
        for k in range(4):
            edges[tuple(sorted((q[k], q[(k + 1) % 4])))] += 1
    assert set(edges.values()) == {2}  # watertight: every edge shared by exactly two quads
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    a, b, c = (verts[tris[:, i]] for i in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert abs(vol - 4 / 3 * np.pi * 0.3 ** 3) / (4 / 3 * np.pi * 0.3 ** 3) < 0.06  # >0 = outward


def test_envelope_respects_band():
    grid = make_grid(lo=(-0.5, -0.5, -0.5), hi=(0.5, 0.5, 0.5), voxel=0.02)
    hull = np.ones(grid.shape, bool)
    pts = np.zeros((1, 3))  # single body point at the origin
    env = carve_envelope(hull, grid, pts, band=0.2, min_voxels=1)
    xs, ys, zs = grid.axes()
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    assert env[np.sqrt(X ** 2 + Y ** 2 + Z ** 2) > 0.24].sum() == 0
    assert env[np.sqrt(X ** 2 + Y ** 2 + Z ** 2) < 0.16].all()


def test_obj_roundtrip_axes(tmp_path):
    from armor_forge.carve import write_obj
    from armor_forge.meshio import read_obj
    v = np.array([[0.1, -0.2, 1.5], [0.3, 0.4, 0.5], [-0.2, 0.0, 1.0]])
    p = tmp_path / "t.obj"
    write_obj(str(p), v, np.array([[0, 1, 2, 2]]))
    back, faces = read_obj(str(p))
    assert np.allclose(back, v, atol=1e-4) and faces == [[0, 1, 2, 2]]
