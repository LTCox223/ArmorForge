"""Score a built armor set against the part-ID masks it was built from.

The set is rendered into a visible-part ID image in each of the four views (painter's algorithm) and
compared with the mask: per part, intersection-over-union of the visible pixels, plus the silhouette.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

VIEWS = ("Front", "Back", "Side", "Side_Without_Arms")
ARM_IDS = {4, 5, 6, 7, 8}


def id_buffer(parts, bodies, drawing, view):
    n = drawing.lab[view].shape[0]
    img = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(img)
    tris = []

    def add(pts, faces, pid):
        pts = np.asarray(pts)
        if view in ("Front", "Back"):
            u = pts[:, 0] * (1 if view == "Front" else -1)
            d = -pts[:, 1] if view == "Front" else pts[:, 1]
        else:
            u, d = pts[:, 1], pts[:, 0]
        c = drawing.ox + u * drawing.ppm
        r = drawing.oy - pts[:, 2] * drawing.ppm
        for f in faces:
            for k in range(1, len(f) - 1):
                t = [f[0], f[k], f[k + 1]]
                tris.append((float(d[t].mean()), pid, [(c[i], r[i]) for i in t]))

    base = bodies["armless"] if view == "Side_Without_Arms" else bodies["body"]
    add(base.v, base.faces, 17)
    for name, (pts, faces) in parts.items():
        pid = int(name[:2])
        if view == "Side_Without_Arms" and pid in ARM_IDS:
            continue
        add(pts, faces, pid)
    for _, pid, poly in sorted(tris, key=lambda t: t[0]):
        dr.polygon(poly, fill=pid)
    return np.asarray(img)


def score(parts, bodies, drawing, verbose=True):
    inter, union = {}, {}
    sil_i = sil_u = 0
    for view in VIEWS:
        buf = id_buffer(parts, bodies, drawing, view)
        lab = drawing.lab[view]
        for pid in range(1, 18):
            m, r = lab == pid, buf == pid
            inter[pid] = inter.get(pid, 0) + int((m & r).sum())
            union[pid] = union.get(pid, 0) + int((m | r).sum())
        sil_i += int(((lab > 0) & (buf > 0)).sum())
        sil_u += int(((lab > 0) | (buf > 0)).sum())
    res = {pid: inter[pid] / union[pid] for pid in inter if union[pid]}
    if verbose:
        for pid, v in sorted(res.items()):
            print(f"  part {pid:2d}  visible-pixel IoU {100 * v:5.1f}%")
        print(f"  silhouette IoU {100 * sil_i / sil_u:5.1f}%   mean part IoU {100 * np.mean([res[p] for p in res if p < 17]):5.1f}%")
    return res, sil_i / sil_u
