"""Body-shell fitting: build every armor/cloth piece from the mannequin's own surface (ArmorForge step 1).

The AI draws the concept and a flat-colour piece map (one colour per piece, four orthographic views).
This module turns that into pieces without inventing topology:

1. Label: every body face votes for the piece drawn over it in each view that faces it and can see it.
2. Select: a piece is the set of body faces it won, cleaned up (holes filled, specks removed).
3. Snap: the piece's open edge is slid along the surface onto the drawn outline, then smoothed,
   so hems follow the drawing instead of the body's face grid.
4. Lift: each vertex moves out along its normal until it would leave the piece's outline in some
   view (a per-piece visual hull), smoothed, so stand-off pieces (pauldrons, helmet, knees) get volume.

Everything here is numpy only; blender/build_shell_set.py feeds it the body and makes the objects.
Coordinates: world metres, Z up, character faces -Y, character left is +X.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

VIEW_FILES = {"front": "Front", "back": "Back", "side": "Side", "side_noarm": "Side_Without_Arms"}
# Unit vector from the body toward each camera.
TO_CAM = {"front": (0.0, -1.0, 0.0), "back": (0.0, 1.0, 0.0),
          "side": (1.0, 0.0, 0.0), "side_noarm": (1.0, 0.0, 0.0)}

# ----------------------------------------------------------------------------- piece settings
# kind: "shell"   - built from the body surface (fitted cloth, plate rough-in, straps)
#       "hanging" - not on the body; built as a hanging panel (tabard, cape)
# gap:  minimum distance from the skin (m); lift: max extra stand-off from the outline (m)
# thickness: Solidify thickness (m)
PIECES = {
    1: dict(key="helmet", kind="shell", gap=0.012, lift=0.08, thickness=0.006),
    2: dict(key="gorget", kind="shell", gap=0.008, lift=0.04, thickness=0.006),
    3: dict(key="cuirass", kind="shell", gap=0.010, lift=0.06, thickness=0.006),
    4: dict(key="pauldrons", kind="shell", gap=0.014, lift=0.10, thickness=0.006),
    5: dict(key="shoulder_lames", kind="shell", gap=0.012, lift=0.08, thickness=0.005),
    6: dict(key="sleeves", kind="shell", gap=0.004, lift=0.02, thickness=0.004),
    7: dict(key="vambraces", kind="shell", gap=0.008, lift=0.04, thickness=0.005),
    8: dict(key="gauntlets", kind="shell", gap=0.004, lift=0.03, thickness=0.004),
    9: dict(key="belt", kind="shell", gap=0.008, lift=0.05, thickness=0.008),
    10: dict(key="upper_tassets", kind="shell", gap=0.014, lift=0.08, thickness=0.005),
    11: dict(key="lower_tassets", kind="shell", gap=0.012, lift=0.08, thickness=0.005),
    12: dict(key="tabard", kind="hanging", gap=0.010, lift=0.0, thickness=0.003),
    13: dict(key="thigh_plates", kind="shell", gap=0.008, lift=0.05, thickness=0.005),
    14: dict(key="knee_guards", kind="shell", gap=0.010, lift=0.07, thickness=0.006),
    15: dict(key="greaves", kind="shell", gap=0.008, lift=0.05, thickness=0.006),
    16: dict(key="boots", kind="shell", gap=0.006, lift=0.05, thickness=0.006),
    17: dict(key="undersuit", kind="base", gap=0.002, lift=0.0, thickness=0.003, skip=(1, 8, 16)),
}
# kind "base": one continuous layer over the whole body except the faces won by `skip` pieces
# (they replace that body section: helmet over the head, gauntlets over the hands, boots over the feet).
ARM_IDS = {4, 5, 6, 7, 8}
BASE = 17                   # undersuit: the layer under everything
SHOULDER_IDS = {4, 5}       # arm pieces that also cover the torso's shoulder   # pieces drawn only in the Side view (not Side_Without_Arms)

# Layering (same table as templates.py): UNDER[pid] = pieces pid lies on top of.
UNDER = {1: {2}, 2: {3}, 4: {3, 5, 6}, 5: {3, 6}, 7: {6}, 8: {7}, 9: {3, 10, 11, 12},
         10: {11}, 12: {10, 11, 13}, 14: {15}, 15: {16}}
OVER: dict[int, set] = {}
for _top, _unders in UNDER.items():
    for _u in _unders:
        OVER.setdefault(_u, set()).add(_top)
OVER.setdefault(13, set()).update({10, 11})


# ----------------------------------------------------------------------------- views
@dataclass
class View:
    """One orthographic label grid. col = (c0 + s*sign*u) / f, row = (r0 - s*z) / f, u = x or y."""
    name: str
    labels: np.ndarray          # (H, W) uint8 part ids, 0 = background
    c0: float
    r0: float
    s: float = 1600.0           # px per metre at full resolution
    f: float = 4.0              # downsample factor of the label grid

    @property
    def axis(self):
        return 1 if self.name.startswith("side") else 0

    @property
    def sign(self):
        return -1.0 if self.name == "back" else 1.0

    @property
    def to_cam(self):
        return np.array(TO_CAM[self.name])

    @property
    def px_per_m(self):
        return self.s / self.f

    def project(self, p):
        p = np.atleast_2d(p)
        col = (self.c0 + self.s * self.sign * p[:, self.axis]) / self.f
        row = (self.r0 - self.s * p[:, 2]) / self.f
        return col, row

    def image_dir(self, n):
        """Direction (dcol, drow) in pixels per metre of a 3D direction, as seen in this view."""
        n = np.atleast_2d(n)
        return np.stack([self.sign * n[:, self.axis], -n[:, 2]], 1) * self.px_per_m

    def sample(self, p):
        col, row = self.project(p)
        return self.sample_px(col, row)

    def sample_px(self, col, row):
        h, w = self.labels.shape
        c = np.floor(col).astype(int)
        r = np.floor(row).astype(int)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.zeros(len(c), np.uint8)
        out[ok] = self.labels[r[ok], c[ok]]
        return out


def load_views(labels_dir, cams_json=None, meta_json=None):
    """Load <View>_labels.npy grids. cams.json (optional) holds calibrated full-res origins per view."""
    meta = json.load(open(meta_json)) if meta_json else {}
    s = float(meta.get("pixels_per_meter", 1600.0))
    c0, r0 = meta.get("world_origin_px", (2048, 3712))
    cams = json.load(open(cams_json)) if cams_json else {}
    views = {}
    for name, stem in VIEW_FILES.items():
        lab = np.load(os.path.join(labels_dir, f"{stem}_labels.npy"))
        f = 4096 / lab.shape[1]
        cc, rr = cams.get(name, (c0, r0))
        views[name] = View(name, lab, float(cc), float(rr), s, f)
    return views


def mirror(p):
    q = np.array(p, float, copy=True)
    q[..., 0] *= -1
    return q


# ----------------------------------------------------------------------------- mesh helpers
def face_adjacency(faces):
    """Faces sharing an edge. -> list of arrays."""
    edge_faces = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    adj = [[] for _ in faces]
    for fs in edge_faces.values():
        for a in fs:
            for b in fs:
                if a != b:
                    adj[a].append(b)
    return [np.array(sorted(set(a)), int) for a in adj]


def vertex_neighbors(n_verts, faces):
    nb = [set() for _ in range(n_verts)]
    for f in faces:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            nb[a].add(b)
            nb[b].add(a)
    return [np.array(sorted(s), int) for s in nb]


def boundary_edges(faces):
    count = {}
    for f in faces:
        for a, b in zip(f, list(f[1:]) + [f[0]]):
            k = (min(a, b), max(a, b))
            count[k] = count.get(k, 0) + 1
    return [k for k, c in count.items() if c == 1]


def components(mask, adj):
    """Connected components of the faces where mask is True. -> list of index arrays."""
    seen = np.zeros(len(mask), bool)
    out = []
    for s in np.flatnonzero(mask):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            f = stack.pop()
            comp.append(f)
            for g in adj[f]:
                if mask[g] and not seen[g]:
                    seen[g] = True
                    stack.append(g)
        out.append(np.array(comp))
    return out


def smooth_scalar(values, neighbors, iters=8, lam=0.5, pinned=None):
    v = np.asarray(values, float).copy()
    for _ in range(iters):
        avg = np.array([v[n].mean() if len(n) else v[i] for i, n in enumerate(neighbors)])
        new = v + lam * (avg - v)
        if pinned is not None:
            new[pinned] = v[pinned]
        v = new
    return v


# ----------------------------------------------------------------------------- 1. labelling
def face_votes(centers, normals, views, visible, is_arm, symmetric=True, min_facing=0.1):
    """Weighted votes (F, 18) for which piece is drawn over each face.

    visible[view] : (F,) bool, face centre has a clear line to that camera.
    is_arm        : (F,) bool, the face belongs to an arm (uses Side, else Side_Without_Arms).
    symmetric     : also sample the mirrored position in front/back views (drawings are never
                    perfectly symmetric; the build is, so average both halves of the drawing).
    """
    votes = np.zeros((len(centers), 18))
    rows = np.arange(len(centers))
    for name, v in views.items():
        w = normals @ v.to_cam
        use = (w > min_facing) & visible[name]
        samples = [(centers, 1.0)]
        if symmetric and v.axis == 0:
            samples = [(centers, 0.5), (mirror(centers), 0.5)]
        for pts, k in samples:
            lab = v.sample(pts)
            m = use & (lab > 0)
            if name == "side":
                # arm faces take arm pieces (the arm hangs in front of the hips in this view), and
                # torso faces take the shoulder pieces that the no-arms view removed
                m &= (is_arm & np.isin(lab, list(ARM_IDS))) | np.isin(lab, list(SHOULDER_IDS))
            elif name == "side_noarm":
                m &= ~is_arm
            np.add.at(votes, (rows[m], lab[m]), (w * k)[m])
    # Faces no view looks at squarely (tops of shoulders, soles, crotch) get a weak silhouette vote:
    # whatever is drawn where they project, in every view, ignoring facing and occlusion.
    weak = votes.sum(1) == 0
    if weak.any():
        for name, v in views.items():
            ok = weak & (is_arm if name == "side" else ~is_arm if name == "side_noarm" else True)
            lab = v.sample(centers)
            m = ok & (lab > 0)
            np.add.at(votes, (rows[m], lab[m]), 0.01)
    return votes


def assign_labels(votes, adj, skip_ids=(), min_votes=0.05, speck=6, passes=2, over_ratio=0.35):
    """Pick one piece per face. Faces whose only votes are for skip_ids (hanging pieces) or that no
    view sees are filled from their neighbours. Small islands are absorbed by their surroundings."""
    v = votes.copy()
    v[:, 0] = 0
    for s in skip_ids:
        # what is under a hanging piece can't be seen: default it to the base layer
        v[:, BASE] += v[:, s]
        v[:, s] = 0
    lab = np.where(v.max(1) > min_votes, v.argmax(1), 0)
    lab = prefer_upper_layers(lab, v, over_ratio)
    lab = fill_unlabelled(lab, adj)
    for _ in range(passes):
        lab = majority_filter(lab, adj)
        lab = absorb_specks(lab, adj, speck)
    return lab


def prefer_upper_layers(lab, votes, ratio):
    """If a piece that is layered over the winner got a fair share of the votes, it wins: the view that
    saw the lower piece was probably looking underneath the upper one's edge."""
    lab = lab.copy()
    for _ in range(3):                      # follow chains (pauldron > lame > sleeve)
        changed = False
        for top in range(1, 18):
            if top == BASE:
                continue
            unders = UNDER.get(top, set()) | {BASE}
            m = np.isin(lab, list(unders)) & (votes[:, top] >= ratio * votes.max(1)) & (votes[:, top] > 0)
            if m.any():
                lab[m] = top
                changed = True
        if not changed:
            break
    return lab


def fill_unlabelled(lab, adj):
    lab = lab.copy()
    while (lab == 0).any():
        changed = False
        new = lab.copy()
        for f in np.flatnonzero(lab == 0):
            nl = lab[adj[f]]
            nl = nl[nl > 0]
            if len(nl):
                new[f] = np.bincount(nl).argmax()
                changed = True
        lab = new
        if not changed:
            break
    return lab


def majority_filter(lab, adj):
    new = lab.copy()
    for f, nb in enumerate(adj):
        if len(nb) < 3:
            continue
        nl = lab[nb]
        best = np.bincount(nl).argmax()
        if best != lab[f] and (nl == best).sum() >= len(nb) - 1 and (nl == lab[f]).sum() <= 1:
            new[f] = best
    return new


def absorb_specks(lab, adj, speck):
    lab = lab.copy()
    for pid in np.unique(lab):
        for comp in components(lab == pid, adj):
            if len(comp) >= speck:
                continue
            ring = np.concatenate([adj[f] for f in comp])
            ring = lab[ring]
            ring = ring[ring != pid]
            if len(ring):
                lab[comp] = np.bincount(ring).argmax()
    return lab


# ----------------------------------------------------------------------------- 2. per-vertex view choice
def best_views(normals, is_arm, views):
    """For each vertex, the view that faces it most squarely (front/back/side/side_noarm)."""
    names = list(views)
    score = np.stack([normals @ views[n].to_cam for n in names], 1)
    for j, n in enumerate(names):
        if n == "side":
            score[~is_arm, j] = -9
        elif n == "side_noarm":
            score[is_arm, j] = -9
    return np.array(names, object)[score.argmax(1)], score.max(1)


def label_at(points, view_names, views, pid, symmetric=True):
    """True where each point (in its own view) lands on piece pid (either half, if symmetric)."""
    out = np.zeros(len(points), bool)
    for name in set(view_names):
        m = view_names == name
        v = views[name]
        hit = v.sample(points[m]) == pid
        if symmetric and v.axis == 0:
            hit |= v.sample(mirror(points[m])) == pid
        out[m] = hit
    return out


# ----------------------------------------------------------------------------- 3. outline snapping
def snap_boundary(verts, faces, piece_faces, pid, views, vnames, seam, steps=8, smooth_iters=3):
    """Slide the open edge of a piece onto the drawn outline.

    verts: (V, 3) body vertices; faces: body faces; piece_faces: indices of this piece's faces.
    vnames: (V,) best view per vertex; seam: (V,) True on the mirror plane (never moved sideways).
    Returns a copy of verts with the piece's boundary vertices moved (other vertices untouched).
    """
    sub = [faces[i] for i in piece_faces]
    bnd = boundary_edges(sub)
    if not bnd:
        return verts.copy()
    piece_v = np.unique(np.concatenate([np.asarray(f) for f in sub]))
    inside = np.zeros(len(verts), bool)
    inside[piece_v] = True
    bverts = np.unique(np.array(bnd).ravel())
    bverts = bverts[~seam[bverts]]
    nb_all = vertex_neighbors(len(verts), faces)
    out = verts.copy()
    ts = np.linspace(0, 1, steps + 1)[1:]
    for b in bverts:
        here = label_at(verts[b:b + 1], vnames[b:b + 1], views, pid)[0]
        moves = []
        for o in nb_all[b]:
            # outward edges if b is inside the drawing, inward edges if it is outside
            if here == inside[o]:
                continue
            pts = verts[b] + ts[:, None] * (verts[o] - verts[b])
            hit = label_at(pts, np.full(len(pts), vnames[b], object), views, pid)
            if here:
                k = np.flatnonzero(~hit)
                t = ts[k[0] - 1] if len(k) and k[0] > 0 else (0.0 if len(k) else 1.0)
                t = min(t, 0.85)        # never collapse onto the outside vertex
            else:
                k = np.flatnonzero(hit)
                t = min(ts[k[0]] if len(k) else 0.0, 0.6)
            moves.append(t * (verts[o] - verts[b]))
        if moves:
            out[b] = verts[b] + np.mean(moves, 0)
    # smooth along the boundary loop only (removes the face-grid staircase)
    bset = set(bverts.tolist())
    bnb = {b: [] for b in bverts}
    for a, c in bnd:
        if a in bset and c in bset:
            bnb[a].append(c)
            bnb[c].append(a)
    for _ in range(smooth_iters):
        new = out.copy()
        for b, nb in bnb.items():
            if len(nb) == 2:
                new[b] = 0.5 * out[b] + 0.25 * (out[nb[0]] + out[nb[1]])
        out = new
    return out


# ----------------------------------------------------------------------------- 4. lift (visual hull)
def lift_distance(points, normals, views, pid, max_lift, is_arm, symmetric=True, min_proj=0.35):
    """How far each point can move along its normal and still land on the piece (or a piece drawn
    over it) in every view where that motion is visible. -> (N,) metres in [0, max_lift]."""
    allowed = np.array(sorted({pid} | OVER.get(pid, set())))
    t = np.full(len(points), max_lift)
    if max_lift <= 0:
        return np.zeros(len(points))
    for name, v in views.items():
        if name == "side":
            rows = is_arm
        elif name == "side_noarm":
            rows = ~is_arm
        else:
            rows = np.ones(len(points), bool)
        samples = [(points, normals)]
        if symmetric and v.axis == 0:
            samples.append((mirror(points), mirror(normals)))
        tv = np.full((len(samples), len(points)), np.nan)
        for k, (p, n) in enumerate(samples):
            d = v.image_dir(n)                       # px per metre of motion
            speed = np.linalg.norm(d, axis=1)
            ok = rows & (speed > min_proj * v.px_per_m)
            col, row = v.project(p)
            n_steps = int(np.ceil(max_lift * v.px_per_m)) + 1
            reach = np.zeros(len(p))
            start = v.sample_px(col, row)
            alive = np.isin(start, allowed)
            # starting on another piece's colour: this view can't tell how far ours reaches here
            ok &= alive | (start == 0)
            for s in range(1, n_steps + 1):
                dist = s / v.px_per_m                # metres of image travel
                tt = dist / np.maximum(speed / v.px_per_m, 1e-6)
                cc = col + d[:, 0] / np.maximum(speed, 1e-9) * s
                rr = row + d[:, 1] / np.maximum(speed, 1e-9) * s
                alive &= np.isin(v.sample_px(cc, rr), allowed) & (tt <= max_lift)
                reach = np.where(alive, tt, reach)
            tv[k] = np.where(ok, reach, np.nan)
        valid = ~np.isnan(tv).all(0)
        tv = np.nanmean(np.where(np.isnan(tv), np.nan, tv), 0) if valid.any() else np.zeros(len(points))
        t = np.where(valid, np.minimum(t, np.nan_to_num(tv)), t)
    return np.clip(t, 0, max_lift)


# ----------------------------------------------------------------------------- 5. hanging cloth
def _rows_of(view, pid, symmetric=True):
    """World z and outer half-width |x| of the piece for every drawn row of a front/back view."""
    lab = view.labels == pid
    rows = np.flatnonzero(lab.any(1))
    z = (view.r0 / view.f - rows - 0.5) * view.f / view.s
    half = []
    for r in rows:
        cols = np.flatnonzero(lab[r])
        u = ((cols + 0.5) * view.f - view.c0) / view.s * view.sign
        half.append(np.abs(u).max() if symmetric else u.max())
    return z, np.array(half)


def side_depth(view, pid, z, front=True):
    """Front-most (front=True, min y) or rear-most y of the piece in a side view at heights z.
    NaN where that half of the side view shows none of the piece."""
    lab = view.labels == pid
    out = np.full(len(z), np.nan)
    rows = np.round(view.r0 / view.f - z * view.s / view.f - 0.5).astype(int)
    for i, r in enumerate(rows):
        if not 0 <= r < lab.shape[0]:
            continue
        cols = np.flatnonzero(lab[r])
        if not len(cols):
            continue
        y = ((cols + 0.5) * view.f - view.c0) / view.s
        y = y[y < 0] if front else y[y > 0]
        if len(y):
            out[i] = y.min() if front else y.max()
    return out


def hanging_panel(views, pid, face, under_fn, gap, cell=0.03, smooth_iters=6):
    """A hanging cloth panel (tabard, cape) as a quad grid on the +X half (mirror it).

    face: "front" or "back" (which view draws the panel; the side view gives how far it stands out).
    under_fn(x, z) -> y of the first surface behind the panel (body or pieces), NaN if none; the panel
    keeps `gap` in front of it. Rows follow the drawn outline; cells outside the drawing are dropped,
    so slits and pointed hems come out of the grid.
    Returns verts (N, 3), faces (list of quads), or (None, None) if the view has no such piece.
    """
    v = views[face]
    z_rows, half = _rows_of(v, pid)
    if len(z_rows) < 3:
        return None, None
    z_top, z_bot = z_rows.max(), z_rows.min()
    nz = max(2, int(np.ceil((z_top - z_bot) / cell)))
    zs = np.linspace(z_top, z_bot, nz + 1)
    order = np.argsort(z_rows)
    hw = np.interp(zs, z_rows[order], half[order])
    nx = max(2, int(np.ceil(hw.max() / cell)))
    front = face == "front"
    sv = views["side"]
    ys_side = side_depth(sv, pid, zs, front=front)
    P = np.zeros((nz + 1, nx + 1, 3))
    for i, z in enumerate(zs):
        xs = np.linspace(0.0, hw[i], nx + 1)
        P[i, :, 0] = xs
        P[i, :, 2] = z
        y_under = np.array([under_fn(x, z) for x in xs])
        y_need = y_under - gap if front else y_under + gap
        # cloth drops as one sheet: each row clears everything under it
        if np.isnan(y_need).all():
            y = y_need.copy()
        else:
            y = np.full(nx + 1, np.nanmin(y_need) if front else np.nanmax(y_need))
        if not np.isnan(ys_side[i]):
            # hang out to the side-view outline, but never into what is underneath
            y = np.where(np.isnan(y), ys_side[i],
                         np.minimum(y, ys_side[i]) if front else np.maximum(y, ys_side[i]))
        P[i, :, 1] = y
    # fill rows with no information from their neighbours, then smooth the depth field
    Y = P[..., 1]
    for _ in range(nz + nx):
        bad = np.isnan(Y)
        if not bad.any():
            break
        pad = np.pad(Y, 1, constant_values=np.nan)
        nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        Y[bad] = np.nanmean(nb, 0)[bad]
    Y = np.nan_to_num(Y, nan=0.0)
    for _ in range(smooth_iters):
        pad = np.pad(Y, 1, mode="edge")
        avg = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
        sm = 0.5 * Y + 0.5 * avg
        # smoothing may only move cloth away from the body, never into it
        Y = np.minimum(Y, sm) if front else np.maximum(Y, sm)
    P[..., 1] = Y
    verts = P.reshape(-1, 3)
    idx = np.arange(len(verts)).reshape(nz + 1, nx + 1)
    keep = np.zeros((nz, nx), bool)
    for i in range(nz):
        for j in range(nx):
            c = P[i:i + 2, j:j + 2].reshape(-1, 3).mean(0)
            # drop cells on background or bare undersuit (slits, pointed hems); keep cells where
            # another piece is drawn over the cloth
            lab_c = max(v.sample(c[None])[0], v.sample(mirror(c[None]))[0])
            keep[i, j] = lab_c not in (0, BASE)
    # fill pinholes: a dropped cell with 3+ kept neighbours (the centre column counts its mirror)
    pad = np.pad(keep, 1)
    pad[1:-1, 0] = keep[:, 0]
    nb = pad[:-2, 1:-1].astype(int) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    keep |= nb >= 3
    faces = []
    for i, j in zip(*np.nonzero(keep)):
        q = [idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]]
        faces.append(q[::-1] if front else q)   # normals face away from the body
    return (verts, faces) if faces else (None, None)
