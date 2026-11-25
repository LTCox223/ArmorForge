"""Inflate the base mesh to the armor silhouettes.

Instead of building new geometry from outlines (which loses the body's curves), push every
vertex of the base mesh outward along its own normal until it would leave an armor silhouette.
The result keeps the base mesh's topology, curves and vertex order; only positions change.

For a vertex moved by t along its normal n, its position in a view moves by t * (projected n).
So the furthest it can go before leaving that view's silhouette is
    t_view = (distance to the silhouette edge along the projected normal) / |projected n|
and the vertex may move min(t_view) over all views (the visual-hull condition, per vertex).
"""
from __future__ import annotations

import numpy as np


def weld(verts, faces, tol=1e-5):
    """Merge vertices closer than tol (e.g. the mirrored seam). Returns (verts, faces, old->new)."""
    key = np.round(np.asarray(verts) / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    new_faces = []
    for f in faces:
        g = []
        for i in f:
            j = int(inv[i])
            if not g or g[-1] != j:
                g.append(j)
        if len(g) > 1 and g[0] == g[-1]:
            g.pop()
        if len(g) >= 3 and len(set(g)) == len(g):
            new_faces.append(g)
    return np.asarray(verts)[first], new_faces, inv


def subdivide(verts, faces):
    """One level of linear quad subdivision (no smoothing): 4x the faces on the same surface."""
    verts = [tuple(v) for v in np.asarray(verts)]
    mid = {}

    def edge_mid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in mid:
            mid[k] = len(verts)
            verts.append(tuple((np.array(verts[a]) + np.array(verts[b])) / 2))
        return mid[k]

    out = []
    for f in faces:
        if len(f) != 4:
            raise ValueError("subdivide() needs an all-quad mesh")
        a, b, c, d = f
        ab, bc, cd, da = edge_mid(a, b), edge_mid(b, c), edge_mid(c, d), edge_mid(d, a)
        verts.append(tuple(np.mean([verts[i] for i in f], axis=0)))
        m = len(verts) - 1
        out += [[a, ab, m, da], [ab, b, bc, m], [m, bc, c, cd], [da, m, cd, d]]
    return np.array(verts), out


def vertex_normals(verts, faces):
    n = np.zeros_like(verts, dtype=float)
    for f in faces:
        p = verts[f]
        q = np.roll(p, -1, axis=0)
        fn = np.array([((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
                       ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
                       ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum()])
        n[f] += fn  # Newell normal; its length is proportional to the face area
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def adjacency(faces, nverts):
    nb = [set() for _ in range(nverts)]
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            nb[a].add(b)
            nb[b].add(a)
    return [np.array(sorted(s), dtype=int) for s in nb]


def _march(mask, start_u, start_z, du, dz, proj, step, t_cap):
    """Distance along the in-view direction until the silhouette is left (inf if never within cap)."""
    n = len(start_u)
    dist = np.full(n, np.inf)
    active = proj > 1e-6
    steps = int(t_cap / step)
    ux = np.where(active, du / np.maximum(proj, 1e-9), 0.0)
    uz = np.where(active, dz / np.maximum(proj, 1e-9), 0.0)
    alive = active.copy()
    for k in range(0, steps + 1):
        d = k * step
        inside = mask.sample(start_u + ux * d, start_z + uz * d)
        out = alive & ~inside
        dist[out] = d
        alive &= inside
        if not alive.any():
            break
    return dist


def inflate_offsets(verts, normals, views, step=0.002, t_max=0.12, arm_x=0.42, arm_t_max=0.06,
                    min_proj=0.2):
    """Per-vertex push distance t >= 0 along the vertex normal (see module docstring)."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    cap = t_max * 1.2
    t = np.full(len(verts), np.inf)

    def constrain(mask, su, sz, a, b):
        proj = np.hypot(a, b)
        d = _march(mask, su, sz, a, b, proj, step, cap * 1.0)
        tv = np.where(proj >= min_proj, d / np.maximum(proj, 1e-9), np.inf)
        return tv

    t = np.minimum(t, constrain(views["front"], x, z, nx, nz))
    t = np.minimum(t, constrain(views["back"], -x, z, -nx, nz))
    near = np.abs(x) <= arm_x
    t_near = constrain(views["side_noarm"], y, z, ny, nz)
    t_arm = constrain(views["side"], y, z, ny, nz)
    t = np.minimum(t, np.where(near, t_near, t_arm))
    limit = np.where(near, t_max, arm_t_max)
    return np.clip(np.where(np.isfinite(t), t, limit), 0.0, limit)


def smooth_field(t, nbrs, iters=3, lam=0.5):
    t = t.copy()
    for _ in range(iters):
        avg = np.array([t[n].mean() if len(n) else t[i] for i, n in enumerate(nbrs)])
        t = (1 - lam) * t + lam * avg
    return t


def inflate(verts, faces, views, gap=0.004, smooth_iters=3, **kw):
    """Return (new_verts, t) for a welded mesh."""
    nrm = vertex_normals(verts, faces)
    t = inflate_offsets(verts, nrm, views, **kw)
    if smooth_iters:
        t = smooth_field(t, adjacency(faces, len(verts)), smooth_iters)
    t = np.where(t > 0, t + gap, t)
    return verts + nrm * t[:, None], t


def silhouette_iou(verts, faces, views, px=0.004):
    """Rasterise the mesh in the front/back/side views and compare with the armor silhouettes."""
    from PIL import Image, ImageDraw
    res = {}
    n = int(round(2.0 / px))
    for name, (axis_u, sign) in {"front": (0, 1), "back": (0, -1), "side": (1, 1)}.items():
        img = Image.new("L", (n, int(round(2.06 / px))), 0)
        dr = ImageDraw.Draw(img)
        for f in faces:
            for k in range(1, len(f) - 1):
                tri = verts[[f[0], f[k], f[k + 1]]]
                pts = [((sign * p[axis_u] + 1.0) / px, (2.06 - p[2]) / px) for p in tri]
                dr.polygon(pts, fill=255)
        mine = np.asarray(img) > 127
        h = mine.shape[0]
        zz = 2.06 - (np.arange(h) + 0.5) * px
        uu = -1.0 + (np.arange(n) + 0.5) * px
        mask = views[name].sample(uu[None, :], zz[:, None])
        res[name] = (mine & mask).sum() / max(1, (mine | mask).sum())
    return res
