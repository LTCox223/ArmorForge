"""Small software renderer for quick looks at a quad mesh (no Blender or GPU needed)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .inflate import vertex_normals


def render(verts, quads, az_deg, el_deg, size=(560, 940), pad=0.04, bg=(233, 238, 241),
           smooth=True, face_colors=None):
    verts = np.asarray(verts, dtype=float)
    quads = np.asarray(quads)
    az, el = np.radians(az_deg), np.radians(el_deg)
    c = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    r = np.cross(-c, [0, 0, 1.0]); r /= np.linalg.norm(r)
    up = np.cross(r, -c)
    vn = vertex_normals(verts, [list(q) for q in quads]) if smooth else None
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if smooth:
        n = vn[tris].mean(axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    valid = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    keep = ((n @ c) > -0.05) & valid
    fc = None if face_colors is None else np.concatenate([face_colors, face_colors])[keep]
    tris, p, n = tris[keep], p[keep], n[keep]
    u, v, d = p @ r, p @ up, p @ c
    w, h = size
    s = (h * (1 - 2 * pad)) / 2.1
    x = w / 2 + u * s
    y = h * (1 - pad) - v * s
    light = 0.65 * c - 0.45 * r + 0.55 * up   # light follows the camera so every view is lit
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.clip(n @ light, 0, 1)
    img = Image.new("RGB", size, bg)
    dr = ImageDraw.Draw(img)
    for i in np.argsort(d.mean(axis=1)):
        if fc is None:
            g = int(np.clip(shade[i], 0, 1) * 225)
            col = (g, int(g * 1.03), int(g * 1.08))
        else:
            k = 0.55 + 0.45 * float(np.clip(shade[i], 0, 1))
            col = tuple(int(min(255, ch * k)) for ch in fc[i])
        dr.polygon([(x[i, 0], y[i, 0]), (x[i, 1], y[i, 1]), (x[i, 2], y[i, 2])], fill=col)
    return img


def contact_sheet(verts, quads, views=((-35, 12), (145, 12), (90, 0)), face_colors=None):
    imgs = [render(verts, quads, a, e, face_colors=face_colors) for a, e in views]
    sheet = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 0))
        x += i.width
    return sheet
