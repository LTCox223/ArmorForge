"""Label mesh faces by armor part using the colour-coded orthographic part masks.

A mask is a 4096x4096 RGBA image (transparent background) whose pixels are painted with one
palette colour per part. For each view we rasterise face ids (painter's algorithm) to find which
face is visible at each pixel, then let each face vote with the mask colours it covers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["Helmet", "Gorget", "Cuirass", "Pauldrons", "Shoulder lames", "Padded sleeves",
         "Vambraces", "Gauntlets", "Belt", "Upper tassets", "Lower tassets", "Split tabard",
         "Thigh plates", "Knee guards", "Greaves", "Armored boots", "Undersuit"]
PALETTE = np.array([(244, 93, 89), (255, 211, 155), (54, 117, 222), (137, 87, 212),
                    (220, 157, 242), (146, 99, 76), (2, 191, 166), (228, 73, 153),
                    (245, 173, 46), (237, 114, 47), (201, 217, 71), (173, 40, 87),
                    (139, 136, 200), (255, 232, 114), (45, 173, 103), (106, 208, 240),
                    (83, 95, 112)], dtype=np.uint8)
BARE = NAMES.index("Undersuit")


def load_mask(path, tol=25):
    """-> int8 array (H, W): part index, or -1 for background / unmatched colours."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    rgb = a[..., :3].astype(np.int32)
    lab = np.full(a.shape[:2], -1, np.int8)
    best = np.full(a.shape[:2], 1 << 30, np.int64)
    for i, c in enumerate(PALETTE):
        d = ((rgb - c.astype(np.int32)) ** 2).sum(-1)
        m = d < best
        best[m] = d[m]
        lab[m] = i
    lab[(a[..., 3] < 128) | (best > tol * tol)] = -1
    return lab


@dataclass
class ViewCam:
    """Maps world (x, y, z) to mask pixels: col = c0 + s*(u_sign*axis), row = r0 - s*z."""
    name: str
    axis: int      # 0: u = x, 1: u = y
    sign: float
    c0: float
    r0: float
    s: float = 1600.0
    depth_sign: float = 1.0   # larger depth = closer to the camera

    def project(self, p):
        u = self.sign * p[:, self.axis]
        return self.c0 + self.s * u, self.r0 - self.s * p[:, 2]

    def depth(self, p):
        # front looks along +Y (camera at -Y), back at -Y, side camera at +X or -X
        return self.depth_sign * p[:, 1 - self.axis if self.axis == 0 else 0]


def default_cams(s=1600.0):
    return {
        "front": ViewCam("front", 0, 1.0, 2048, 3708, s, depth_sign=-1.0),
        "back": ViewCam("back", 0, -1.0, 2048, 3708, s, depth_sign=+1.0),
        "side": ViewCam("side", 1, 1.0, 2048, 3708, s, depth_sign=+1.0),
        "side_noarm": ViewCam("side_noarm", 1, 1.0, 2048, 3708, s, depth_sign=+1.0),
    }


def _tris(faces):
    q = np.asarray(faces)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), np.concatenate([np.arange(len(q))] * 2)


def raster_ids(verts, faces, cam, size=4096, subset=None, scale=1):
    """Visible face id per pixel (-1 = none), painter's algorithm on the face centroid depth."""
    tris, owner = _tris(faces)
    if subset is not None:
        keep = subset[owner]
        tris, owner = tris[keep], owner[keep]
    p = verts[tris]                                   # (T, 3, 3)
    cx, cy = cam.project(p.reshape(-1, 3))
    cx = (cx / scale).reshape(-1, 3)
    cy = (cy / scale).reshape(-1, 3)
    d = cam.depth(p.reshape(-1, 3)).reshape(-1, 3).mean(1)
    img = Image.new("RGB", (size // scale, size // scale), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    for i in np.argsort(d):                           # far first, near overwrites
        k = int(owner[i]) + 1
        dr.polygon([(cx[i, 0], cy[i, 0]), (cx[i, 1], cy[i, 1]), (cx[i, 2], cy[i, 2])],
                   fill=(k & 255, (k >> 8) & 255, (k >> 16) & 255))
    a = np.asarray(img).astype(np.int64)
    return (a[..., 0] | (a[..., 1] << 8) | (a[..., 2] << 16)) - 1


def silhouette(verts, faces, cam, size=4096, scale=4):
    return raster_ids(verts, faces, cam, size, scale=scale) >= 0


def fit_cam(verts, faces, cam, mask, search=48, step=4, scale=4):
    """Nudge (c0, r0) so the projected silhouette best overlaps the mask. Returns (cam, iou)."""
    m = (mask >= 0)[::scale, ::scale]
    best = None
    for dc in range(-search, search + 1, step):
        for dr in range(-search, search + 1, step):
            c = ViewCam(cam.name, cam.axis, cam.sign, cam.c0 + dc, cam.r0 + dr, cam.s, cam.depth_sign)
            sil = silhouette(verts, faces, c, scale=scale)
            iou = (sil & m).sum() / max((sil | m).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, c)
    return best[1], best[0]


def face_votes(verts, faces, cams, masks, arm_x=0.42, scale=2):
    """votes[f, part] += visible pixel count weighted by how squarely the face faces the camera."""
    faces = np.asarray(faces)
    nf = len(faces)
    votes = np.zeros((nf, len(NAMES)))
    cen = verts[faces].mean(1)
    arm = np.abs(cen[:, 0]) > arm_x
    for name, cam in cams.items():
        subset = None
        if name == "side":
            subset = None                      # arms must be allowed to occlude, but only arms vote
        if name == "side_noarm":
            subset = ~arm                      # the arm-free sheet has no arms in it
        ids = raster_ids(verts, faces, cam, scale=scale, subset=subset)
        lab = masks[name][::scale, ::scale]
        ok = (ids >= 0) & (lab >= 0)
        f, l = ids[ok], lab[ok]
        cnt = np.zeros((nf, len(NAMES)))
        np.add.at(cnt, (f, l), 1)
        # facing weight
        p = verts[faces]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        axis = np.zeros(3)
        if cam.axis == 0:
            axis[1] = cam.depth_sign
        else:
            axis[0] = cam.depth_sign
        w = np.clip(n @ axis, 0, 1)
        if name == "side":
            cnt[~arm] = 0
        votes += cnt * (0.15 + w)[:, None]
    return votes


def propagate(verts, faces, labels, min_conf):
    """Fill unlabelled faces from their mirror image across x=0, then from neighbours."""
    from scipy.spatial import cKDTree
    faces = np.asarray(faces)
    cen = verts[faces].mean(1)
    lab = labels.copy()
    mir = cen * np.array([-1, 1, 1])
    _, j = cKDTree(cen).query(mir)
    for i in np.nonzero(lab < 0)[0]:
        if lab[j[i]] >= 0:
            lab[i] = lab[j[i]]
    # neighbour flood
    edge = {}
    for fi, f in enumerate(faces):
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            edge.setdefault((min(a, b), max(a, b)), []).append(fi)
    nb = [[] for _ in faces]
    for fs in edge.values():
        for a in fs:
            nb[a] += [b for b in fs if b != a]
    for _ in range(200):
        todo = np.nonzero(lab < 0)[0]
        if not len(todo):
            break
        new = lab.copy()
        for i in todo:
            c = [lab[b] for b in nb[i] if lab[b] >= 0]
            if c:
                new[i] = np.bincount(c).argmax()
        lab = new
    return lab


def face_neighbours(faces):
    faces = np.asarray(faces)
    edge = {}
    for fi, f in enumerate(faces):
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            edge.setdefault((min(a, b), max(a, b)), []).append(fi)
    nb = [[] for _ in faces]
    for fs in edge.values():
        for a in fs:
            nb[a] += [b for b in fs if b != a]
    return nb


def smooth_labels(faces, lab, passes=3, need=0.6):
    """Relabel a face only when most of its neighbours agree on a different part (keeps edges crisp)."""
    nb = face_neighbours(faces)
    lab = lab.copy()
    for _ in range(passes):
        new = lab.copy()
        for i, ns in enumerate(nb):
            if not ns:
                continue
            c = np.bincount([lab[b] for b in ns], minlength=len(NAMES))
            j = int(c.argmax())
            if j != lab[i] and c[j] >= need * len(ns):
                new[i] = j
        lab = new
    return lab


def label_faces(verts, faces, cams, masks, arm_x=0.42, min_votes=3.0, bare_weight=0.35):
    votes = face_votes(verts, faces, cams, masks, arm_x)
    votes[:, BARE] *= bare_weight    # a plate seen from any view beats 'no plate' seen from another
    tot = votes.sum(1)
    lab = np.where(tot >= min_votes, votes.argmax(1), -1)
    conf = np.where(tot > 0, votes.max(1) / np.maximum(tot, 1e-9), 0)
    return smooth_labels(faces, propagate(verts, faces, lab, min_votes)), lab, conf
