"""Face selection from a painted mask. Pure numpy/python so it can be tested without Blender."""
from __future__ import annotations

import numpy as np

from .projection import ortho_pixel_coords, sample_mask, view_direction


def face_centers_normals(verts, faces):
    """verts (V, 3); faces: list of vertex-index tuples (any size >= 3).

    Returns (centers (F, 3), unit normals (F, 3)); normals use Newell's method.
    """
    verts = np.asarray(verts, dtype=float)
    centers = np.zeros((len(faces), 3))
    normals = np.zeros((len(faces), 3))
    for i, f in enumerate(faces):
        p = verts[list(f)]
        centers[i] = p.mean(axis=0)
        q = np.roll(p, -1, axis=0)
        n = np.array([
            ((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
            ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
            ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum(),
        ])
        length = np.linalg.norm(n)
        normals[i] = n / length if length > 0 else n
    return centers, normals


def select_faces_for_view(centers, normals, cam_world_matrix, ortho_scale, res, mask,
                          occluded=None, facing_eps=0.05):
    """Bool array: faces that face the camera, are not occluded, and land inside the mask."""
    centers = np.asarray(centers, dtype=float)
    normals = np.asarray(normals, dtype=float)
    facing = normals @ (-view_direction(cam_world_matrix)) > facing_eps
    px, _ = ortho_pixel_coords(centers, cam_world_matrix, ortho_scale, res)
    chosen = facing & sample_mask(mask, px)
    if occluded is not None:
        chosen &= ~np.asarray(occluded, dtype=bool)
    return chosen


def face_adjacency(faces):
    """Faces are neighbours if they share an edge. Returns list of sets."""
    edge_map = {}
    for i, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            e = tuple(sorted((f[k], f[(k + 1) % n])))
            edge_map.setdefault(e, []).append(i)
    adj = [set() for _ in faces]
    for owners in edge_map.values():
        for a in owners:
            for b in owners:
                if a != b:
                    adj[a].add(b)
    return adj


def grow_selection(adj, selected, rings=1):
    """Dilate a bool selection by `rings` steps across shared edges.

    Useful because faces seen edge-on (the sides of a limb) never land inside a front/back mask.
    """
    sel = np.asarray(selected, dtype=bool).copy()
    for _ in range(rings):
        new = sel.copy()
        for i in np.flatnonzero(sel):
            for j in adj[i]:
                new[j] = True
        sel = new
    return sel
