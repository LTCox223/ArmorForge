"""Helmet recipe: every helmet is a skull plus a face plate, joined at a standard face seam.

The whole helmet is one lofted surface (helmet.build); the face seam is laid on its edges (the brow
line across the front as a ring, the cheek edges as two meridians) and the surface is split there:

  * face plate = below the brow, between the cheek edges; skull = everything else;
  * both pieces share the seam's exact points, so welding is one Merge by Distance.

Face modes:
  separate   a visor / face plate over the opening: the plate is moved out along its normals
             (weld = False);
  integral   the face is part of the helmet, it was only split to keep the structure: weld = True;
  open       open-faced helmet: the face area is open, the face plate is a rim around the opening
             (rim_rings rings under the brow, rim_cols columns inside each cheek edge); weld = True.

Where the seam goes:
  * from the concept's trim (gold): brow = the face's top line, cheek edges = its outer side lines
    (front view), and with a real side view the face takes the visor's wedge shape (helmet_parts);
  * otherwise defaults: brow at the side profile's step (or the helmet's own brow), cheek edges at
    default_cheek of the helmet's half-width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import helmet as hm
from . import helmet_parts as hp


@dataclass
class FaceParams:
    mode: str = "separate"       # separate | integral | open
    plate_offset: float = 0.004  # separate: how far the plate stands off the skull (m)
    rim_rings: int = 1           # open: rim height in rings under the brow
    rim_cols: int = 1            # open: rim width in columns inside each cheek edge
    default_cheek: float = 0.75  # no cues: cheek edges at this fraction of the half-width


@dataclass
class Piece:
    verts: np.ndarray
    faces: list
    seam: list           # vertex indices on the seam (this piece's numbering)
    crease: list         # edges (this piece's numbering)


@dataclass
class Helmet:
    skull: Piece
    face: Piece
    weld: bool
    mode: str
    info: dict


def _piece(verts, faces, keep, seam_verts, crease):
    sub = [f for f, k in zip(faces, keep) if k]
    used = sorted({v for f in sub for v in f})
    remap = {v: i for i, v in enumerate(used)}
    return Piece(verts[used].copy(), [[remap[v] for v in f] for f in sub],
                 sorted(remap[v] for v in seam_verts if v in remap),
                 [(remap[a], remap[b]) for a, b in crease if a in remap and b in remap])


def vertex_normals(verts, faces):
    n = np.zeros_like(verts)
    for f in faces:
        p = verts[f]
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        if len(f) == 4:
            fn = fn + np.cross(p[2] - p[0], p[3] - p[0])
        n[f] += fn
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def split(verts, faces, info, fp: FaceParams):
    """Skull and face plate from a seamed build (helmet.build with seam=...)."""
    fm = info["face_mask"]
    rk = info["face_rk"]
    K, i_b, kc = info["K"], info["brow_ring"], info["cheek_col"]
    seam_verts = {v for e in info["seam_edges"] for v in e}
    if fp.mode == "open":
        # keep a rim: faces within rim_rings of the brow ring or rim_cols of a cheek column
        def col_from_cheek(k):
            # columns between kc and the centre, measured inward from the cheek edge
            d = min(abs(k - (kc - 1)), abs(k - (K - kc)))
            return d
        rim = np.array([m and (r >= i_b - fp.rim_rings or col_from_cheek(k) < fp.rim_cols)
                        for m, (r, k) in zip(fm, rk)])
        face_keep, skull_keep = rim, ~fm
    else:
        face_keep, skull_keep = fm, ~fm
    face = _piece(verts, faces, face_keep, seam_verts, info["crease"])
    skull = _piece(verts, faces, skull_keep, seam_verts, [])
    if fp.mode == "separate":
        face.verts = face.verts + fp.plate_offset * vertex_normals(face.verts, face.faces)
    return Helmet(skull, face, weld=fp.mode != "separate", mode=fp.mode, info=info)


def crop_neck(side_sil, cam, lines=None, s=1600.0):
    """The AI masks often count the neck behind the face as helmet. Below the neck-guard rim, keep
    only what is in front of the face's rear edge (or, without cues, the front half)."""
    floor = hp.neck_floor(side_sil, cam, s)
    c0, r0 = cam
    rows, cols = np.nonzero(side_sil)
    y = (cols + 0.5 - c0) / s
    z = (r0 - rows - 0.5) / s
    low = z < floor
    if lines is not None:
        behind = y > lines.y_rear(z)
    else:
        at_floor = np.abs(z - floor) < 0.005
        y_lo, y_hi = y[at_floor].min(), y[at_floor].max()
        behind = y > 0.5 * (y_lo + y_hi)
    out = side_sil.copy()
    drop = low & behind
    out[rows[drop], cols[drop]] = False
    return out


def skull_hem(seam, rim_z=None):
    """Hem: the drawn bottom edge, but outside the face the skull stops at the neck-guard rim (the
    front view's bottom edge next to the face is usually the face plate's)."""
    def fn(o, x, y):
        h = hm.hem_heights(o, x, y)
        if rim_z is None:
            return h
        face = (np.abs(x) < seam.half_width + 0.01) & (y < np.median(y))
        return np.where(face, h, np.maximum(h, rim_z))
    return fn


def default_seam(o: hm.Outlines, p: hm.HelmetParams, fp: FaceParams):
    """No cues: a level brow at the helmet's brow height, cheek edges at default_cheek of the width."""
    P = hm.shape_profiles(o, p)
    zb = P["brow"]
    a = float(np.interp(zb, P["z"], P["a"]))
    return hm.Seam(lambda x, zb=zb: zb + 0 * np.asarray(x, float), fp.default_cheek * a)


def seam_from_front(front_lines):
    """Seam from the front view's face lines (helmet_parts.detect_face_front)."""
    sides, (z0, k) = front_lines
    x0, t, zm = sides

    def brow(x):
        return z0 + k * np.abs(np.asarray(x, float))
    return hm.Seam(brow, float(x0 + t * (z0 - zm)))


def untwist_lines(lines: hp.VisorLines, turn):
    """The visor's side-view edges, corrected for a side view drawn turned by `turn` (radians, as
    plate.project: u = y cos - x sin about the helmet's centre line x = 0). The rear edge's gold is
    on the near (+x) corner at x = w(z), so its true depth is y = (u + w sin) / cos; the top edge
    runs from the ridge (x = 0) to that corner."""
    if not turn:
        return lines
    c, s_ = np.cos(turn), np.sin(turn)
    zs = np.array([lines.rear[2] - 0.05, lines.rear[2] + 0.05])
    yr = (lines.y_rear(zs) + lines.w(zs) * s_) / c
    t = (yr[1] - yr[0]) / (zs[1] - zs[0])
    rear = (float(yr[0] + t * (lines.rear[2] - zs[0])), float(t), lines.rear[2])
    z0, k, yref = lines.top_side
    u_r = float(lines.y_rear(z0))                       # the drawn top edge ends at the rear edge
    z_r = z0 + k * (u_r - yref)
    y_f = yref / c
    y_r = (u_r + float(lines.w(z_r)) * s_) / c
    k2 = (z_r - z0) / (y_r - y_f) if abs(y_r - y_f) > 1e-6 else k
    return hp.VisorLines(rear, (z0, float(k2), float(y_f)), lines.sides, lines.top_front)


def turned_side_mask(meshes, cam, shape, turn, s=1600.0):
    """Side-view silhouette of the meshes as drawn turned by `turn` about x = 0."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for verts, faces in meshes:
        u = verts[:, 1] * np.cos(turn) - verts[:, 0] * np.sin(turn)
        cols, rows = cam[0] + u * s, cam[1] - verts[:, 2] * s
        for f in faces:
            d.polygon([(cols[i], rows[i]) for i in f], fill=255)
    return np.asarray(img) > 0


def _scale_depth(o: hm.Outlines, scale, yc):
    sd = o.side
    return hm.Outlines(o.width, hm.Profile(sd.z, yc + scale * (sd.lo - yc), yc + scale * (sd.hi - yc)),
                       o.bottom_front, o.bottom_back, o.bottom_side)


def build(front_sil, back_sil, side_sil, cams, fp=FaceParams(), p=None, art_rgb=None, regs=None,
          side_is_real=True, s=1600.0, side_turn=0.0):
    """The recipe. With art + registrations the seam (and, with a real side view, the face's wedge)
    comes from the concept's gold trim; without, defaults. side_turn: the side view drawn turned
    by this angle (radians, sheet.json): the visor's side-view edges are corrected for it, and the
    depth is scaled so the helmet, seen turned, matches the drawn side outline."""
    p = p or hm.HelmetParams()
    beak_fn = None
    lines = None
    lines_drawn = None
    front_lines = None
    side_full = side_sil
    if art_rgb is not None and regs is not None:
        front = hp.view_data("front", art_rgb, regs["front"], cams["front"], front_sil)
        if side_is_real:
            side = hp.view_data("side", art_rgb, regs["side"], cams["side"], side_sil)
            lines_drawn = hp.detect_visor(front, side)
            lines = untwist_lines(lines_drawn, side_turn)
            front_lines = (lines.sides, lines.top_front)
            beak_fn = hp.skull_beak(lines, gap=0.0)
        else:
            front_lines = hp.detect_face_front(front)
    rim_z = None
    if side_is_real:
        rim_z = hp.neck_floor(side_sil, cams["side"], s)
        side_sil = crop_neck(side_sil, cams["side"], lines, s)
    o = hm.outlines_from_masks(front_sil, back_sil, side_sil, cams, s)
    if front_lines is not None:
        seam = seam_from_front(front_lines)
        if p.brow <= 0:
            # the plan blends from face to skull at the top of the face's rear corners
            zc = float(seam.brow_fn(seam.half_width))
            p = hm.HelmetParams(**{**vars(p), "brow": zc})
    else:
        seam = default_seam(o, p, fp)

    def make(outl):
        verts, faces, info = hm.build(outl, p, beak_fn=beak_fn, seam=seam, hem_fn=skull_hem(seam, rim_z))
        return verts, faces, info

    verts, faces, info = make(o)
    depth_scale = 1.0
    if side_is_real and side_turn:
        # the drawn side outline of a turned helmet is deeper than the helmet: find the depth scale
        # whose turned projection matches it best
        yc = float(np.median(0.5 * (o.side.lo + o.side.hi)))
        target = side_sil
        best = (-1.0, 1.0)
        for sc in np.arange(0.86, 1.041, 0.01):
            v = verts.copy()
            v[:, 1] = yc + sc * (v[:, 1] - yc)
            m = turned_side_mask([(v, faces)], cams["side"], target.shape, side_turn, s)
            iou = (m & target).sum() / max((m | target).sum(), 1)
            if iou > best[0]:
                best = (iou, float(sc))
        depth_scale = best[1]
        if abs(depth_scale - 1.0) > 1e-6:
            o = _scale_depth(o, depth_scale, yc)
            verts, faces, info = make(o)
    helmet = split(verts, faces, info, fp)
    helmet.info.update(lines=lines, lines_drawn=lines_drawn, front_lines=front_lines, seam_def=seam,
                       outlines=o, side_turn=side_turn, depth_scale=depth_scale, side_mask=side_sil)
    return helmet


def silhouette_iou(helmet: Helmet, masks, cams):
    meshes = [(helmet.skull.verts, helmet.skull.faces), (helmet.face.verts, helmet.face.faces)]
    out = {}
    for v, m in masks.items():
        b = hp.project_mask(meshes, v, cams[v], m.shape)
        out[v] = float((b & m).sum() / max((b | m).sum(), 1))
    return out

