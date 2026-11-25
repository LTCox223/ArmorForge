"""Tiny OBJ reader and surface sampler for the body mesh (no Blender needed)."""
from __future__ import annotations

import numpy as np


def read_obj(path, y_up=True):
    """Read an OBJ. With y_up=True (Blender's default OBJ export: Forward -Z, Up Y) vertices are
    converted back to Blender world axes (Z up, -Y forward); False reads the file as-is."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    v = np.array(verts, dtype=float)
    if y_up:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return v, faces


def sample_surface(verts, faces, spacing=0.005, seed=0):
    """Random points on the surface at roughly `spacing` metres apart (n-gons are fanned)."""
    rng = np.random.default_rng(seed)
    tris = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append((f[0], f[k], f[k + 1]))
    tris = np.array(tris)
    a, b, c = (verts[tris[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / spacing ** 2 * 2).astype(int))
    tid = np.repeat(np.arange(len(tris)), n)
    r1, r2 = rng.random(len(tid)), rng.random(len(tid))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[tid] + r1[:, None] * (b - a)[tid] + r2[:, None] * (c - a)[tid]
