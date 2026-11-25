"""Space carving: turn four silhouettes into a rough 3D volume, then a mesh.

Coordinates are Blender world axes: +X = character's left, -Y = forward, +Z = up.
View mapping to panel axes:  front u = x | back u = -x | side u = y   (z is always vertical).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi


@dataclass
class Grid:
    lo: tuple          # coordinate of voxel (0,0,0) centre
    voxel: float
    shape: tuple

    def axes(self):
        return [self.lo[i] + self.voxel * np.arange(self.shape[i]) for i in range(3)]


def make_grid(lo=(-0.85, -0.35, 0.0), hi=(0.85, 0.35, 2.06), voxel=0.01):
    shape = tuple(int(round((hi[i] - lo[i]) / voxel)) + 1 for i in range(3))
    return Grid(tuple(lo), voxel, shape)


def visual_hull(grid, views, arm_x=0.42):
    """Voxels whose projection lies inside every silhouette.

    views: dict with 'front', 'back', 'side', 'side_noarm' ViewMask objects.
    The arms-removed side view is used near the torso (|x| <= arm_x); the full side view
    out where the arms are (|x| > arm_x).
    """
    xs, ys, zs = grid.axes()
    front = views["front"].sample(xs[:, None], zs[None, :])          # (nx, nz)
    back = views["back"].sample(-xs[:, None], zs[None, :])           # (nx, nz)
    side = views["side"].sample(ys[:, None], zs[None, :])            # (ny, nz)
    side_na = views["side_noarm"].sample(ys[:, None], zs[None, :])   # (ny, nz)
    near = (np.abs(xs) <= arm_x)[:, None, None]
    side_ok = (side_na[None] & near) | (side[None] & ~near)          # (nx, ny, nz)
    return front[:, None, :] & back[:, None, :] & side_ok


def drop_small(occ, min_voxels):
    lab, n = ndi.label(occ)
    if n == 0:
        return occ
    sizes = ndi.sum(occ, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= min_voxels) + 1
    return np.isin(lab, keep)


def carve_shell(armor_hull, body_hull, voxel, band=0.12, gap=0.004, min_voxels=200):
    """Armor volume = armor hull minus the body, limited to `band` metres from the body."""
    dist = ndi.distance_transform_edt(~body_hull, sampling=voxel)
    shell = armor_hull & (dist > gap) & (dist <= band)
    shell = ndi.binary_opening(shell, structure=np.ones((2, 2, 2)))
    return drop_small(shell, min_voxels)


def carve_envelope(armor_hull, grid, body_points, band=0.10, min_voxels=200):
    """Outer armor envelope: the armor hull limited to `band` metres from the body surface.

    Keeps the volume under the armor too (it is solid), so the result is a closed, chunky
    stand-in for the armor's outside shape. The silhouettes alone can't separate arms from
    torso in the side views; the body mesh does, via the distance limit.
    """
    from scipy.spatial import cKDTree
    idx = np.argwhere(armor_hull)
    xs, ys, zs = grid.axes()
    pts = np.stack([xs[idx[:, 0]], ys[idx[:, 1]], zs[idx[:, 2]]], axis=1)
    d, _ = cKDTree(body_points).query(pts, workers=-1)
    keep = idx[d <= band]
    env = np.zeros_like(armor_hull)
    env[keep[:, 0], keep[:, 1], keep[:, 2]] = True
    env = ndi.binary_opening(env, structure=np.ones((2, 2, 2)))
    return drop_small(env, min_voxels)


def surface_nets(occ, grid, sigma=1.2):
    """Occupancy -> (vertices (V,3), quads (Q,4)), smooth, watertight except at the grid border."""
    f = ndi.gaussian_filter(np.pad(occ.astype(np.float32), 1), sigma)
    iso = 0.5
    inside = f >= iso
    nx, ny, nz = f.shape
    offs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    cs = tuple(n - 1 for n in f.shape)
    corner_in = [inside[o[0]:o[0] + cs[0], o[1]:o[1] + cs[1], o[2]:o[2] + cs[2]] for o in offs]
    allin = np.logical_and.reduce(corner_in)
    anyin = np.logical_or.reduce(corner_in)
    mixed = anyin & ~allin
    idx = np.argwhere(mixed)
    vid = np.full(cs, -1, dtype=np.int64)
    vid[mixed] = np.arange(len(idx))
    cv = np.stack([f[idx[:, 0] + o[0], idx[:, 1] + o[1], idx[:, 2] + o[2]] for o in offs], axis=1)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    acc = np.zeros((len(idx), 3))
    cnt = np.zeros(len(idx))
    for a, b in edges:
        fa, fb = cv[:, a], cv[:, b]
        cross = (fa >= iso) != (fb >= iso)
        t = np.where(cross, (iso - fa) / np.where(fb - fa == 0, 1, fb - fa), 0)
        pa = np.array(offs[a], dtype=float)
        pb = np.array(offs[b], dtype=float)
        acc += np.where(cross[:, None], pa + t[:, None] * (pb - pa), 0)
        cnt += cross
    local = acc / np.maximum(cnt, 1)[:, None]
    lo = np.array(grid.lo) - grid.voxel  # padding shifted everything by one voxel
    verts = lo + (idx + local) * grid.voxel

    quads = []
    for axis in range(3):
        order = [axis, (axis + 1) % 3, (axis + 2) % 3]
        ins = np.transpose(inside, order)
        vt = np.transpose(vid, order)
        s = ins[:-1] != ins[1:]
        nb, nc = ins.shape[1], ins.shape[2]
        sel = s[:, 1:nb - 1, 1:nc - 1]
        i, j, k = np.nonzero(sel)
        j += 1
        k += 1
        q = np.stack([vt[i, j - 1, k - 1], vt[i, j, k - 1], vt[i, j, k], vt[i, j - 1, k]], axis=1)
        outward_pos = ins[:-1][i, j, k]  # inside at the lower end -> normal points +axis
        q = np.where(outward_pos[:, None], q, q[:, ::-1])
        good = (q >= 0).all(axis=1)
        quads.append(q[good])
    return verts, np.concatenate(quads)


def write_obj(path, verts, quads, z_up_to_y_up=True):
    """OBJ for Blender's default importer (Forward -Z, Up Y): world (x,y,z) -> file (x,z,-y)."""
    v = verts[:, [0, 2, 1]] * np.array([1, 1, -1]) if z_up_to_y_up else verts
    with open(path, "w") as f:
        f.write("# armor-forge rough carve\n")
        for p in v:
            f.write("v %.5f %.5f %.5f\n" % tuple(p))
        for q in quads:
            f.write("f %d %d %d %d\n" % tuple(q + 1))
