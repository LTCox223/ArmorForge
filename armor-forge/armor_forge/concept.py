"""Trace a piece's outline from the concept art itself, using the AI's piece mask only as a guide.

The flat-colour masks (4096 px artboard, 1600 px/m) are rough: wobbly and sometimes a centimetre off
the painted edge. The concept sheet is low resolution (about 230 px/m), but its edge against the light
panel background is clean, and resampling it smoothly gives sub-pixel edges.

Steps per view:
  1. register: fit the mask artboard onto the concept panel (scale + offset) by matching the whole
     figure's silhouette in both;
  2. foreground score: how dark each concept pixel is, resampled into the mask frame (smooth);
  3. trace: inside a reach band around the piece's mask, the piece is where the art is foreground,
     except where the mask says another piece is (so a helmet never grows into the gorget). Thin
     lines (grid axes) are opened away.

The traced mask is written in the same frame as the originals, so everything downstream (helmet.py)
reads it unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from . import sheet

MASK_FILES = {"front": "Front", "back": "Back", "side": "Side", "side_noarm": "Side_Without_Arms"}


@dataclass
class Registration:
    """art (col, row) = scale * mask (col, row) + (bx, by)."""
    scale: float
    bx: float
    by: float
    iou: float

    def to_art(self, c, r):
        return self.scale * np.asarray(c) + self.bx, self.scale * np.asarray(r) + self.by

    def to_mask(self, x, y):
        return (np.asarray(x) - self.bx) / self.scale, (np.asarray(y) - self.by) / self.scale

    def as_dict(self):
        return dict(scale=self.scale, bx=self.bx, by=self.by, iou=self.iou)


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def fg_score(rgb, lo=170.0, hi=205.0):
    """0 on the light panel, 1 on the painted figure; a ramp in between (min channel, like sheet.py).
    The 0.5 level sits at (lo + hi) / 2."""
    m = rgb.min(axis=2)
    return np.clip((hi - m) / (hi - lo), 0.0, 1.0)


def panel_box(view, shape, layout=sheet.LAYOUT, z_extra=0.06):
    """(x0, x1, y0, y1) art pixels of a view's plot rectangle (2 m x 2 m), a bit taller on top."""
    H, W = shape[:2]
    cl, rw = sheet.PANELS[view]
    c0, c1 = (np.array(layout["cols"][cl]) * W)
    r0, r1 = (np.array(layout["rows"][rw]) * H)
    pxv = (r1 - r0) / 2.0
    return c0, c1, r0 - z_extra * pxv, r1


def initial_registration(view, art_shape, cam, ppm=1600.0, layout=sheet.LAYOUT):
    """From the sheet layout and the calibrated mask origin (cams.json): mask col c0 is u = 0, row r0
    is z = 0."""
    H, W = art_shape[:2]
    cl, rw = sheet.PANELS[view]
    x0, x1 = np.array(layout["cols"][cl]) * W
    y0, y1 = np.array(layout["rows"][rw]) * H
    pxu = (x1 - x0) / 2.0
    pxv = (y1 - y0) / 2.0
    scale = 0.5 * (pxu + pxv) / ppm
    c0, r0 = cam
    # u = 0 is the plot's centre column, z = 0 its bottom row
    bx = 0.5 * (x0 + x1) - scale * c0
    by = y1 - scale * r0
    return Registration(scale, bx, by, float("nan"))


def register(view, art_rgb, mask_rgba, cam, ds=4, blur_art=0.7, ppm=1600.0):
    """Fit scale + offset so the mask's whole-figure silhouette lies on the painted figure."""
    from scipy.optimize import minimize

    reg0 = initial_registration(view, art_rgb.shape, cam, ppm)
    x0, x1, y0, y1 = (int(round(v)) for v in panel_box(view, art_rgb.shape))
    panel = art_rgb[y0:y1, x0:x1]
    fig = sheet.figure_mask(panel).astype(np.float32)
    fig = ndi.gaussian_filter(fig, blur_art)
    # the mask's union of all pieces, downsampled (box filter) to a soft silhouette
    union = (mask_rgba[..., 3] > 128).astype(np.float32)
    h, w = union.shape
    small = union[: h - h % ds, : w - w % ds].reshape(h // ds, ds, w // ds, ds).mean((1, 3))
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    ys += 0.5
    xs += 0.5

    def sample(params):
        s, bx, by = params
        mc, mr = (xs - bx) / s, (ys - by) / s
        return ndi.map_coordinates(small, [mr / ds - 0.5, mc / ds - 0.5], order=1, cval=0.0)

    def loss(params):
        m = sample(params)
        inter = (m * fig).sum()
        union_ = (m + fig - m * fig).sum()
        return 1.0 - inter / max(union_, 1e-9)

    p0 = np.array([reg0.scale, reg0.bx, reg0.by])
    res = minimize(loss, p0, method="Nelder-Mead",
                   options=dict(initial_simplex=[p0, p0 + [0.002, 0, 0], p0 + [0, 2.0, 0], p0 + [0, 0, 2.0]],
                                xatol=1e-3, fatol=1e-6, maxiter=400))
    s, bx, by = res.x
    return Registration(float(s), float(bx), float(by), float(1.0 - res.fun))


def part_mask(rgba, rgb):
    return (rgba[..., 0] == rgb[0]) & (rgba[..., 1] == rgb[1]) & (rgba[..., 2] == rgb[2]) & (rgba[..., 3] > 128)


def _disk(r):
    r = max(1, int(round(r)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def trace_part(art_rgb, mask_rgba, reg: Registration, rgb, reach_m=0.03, ppm=1600.0, thin_art_px=1.3,
               margin_m=0.06):
    """-> (traced bool mask in the mask frame, info). See the module docstring."""
    part = part_mask(mask_rgba, rgb)
    other = (mask_rgba[..., 3] > 128) & ~part
    rows, cols = np.nonzero(part)
    if len(rows) == 0:
        return part, dict(empty=True)
    pad = int((reach_m + margin_m) * ppm)
    r0, r1 = max(0, rows.min() - pad), min(part.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(part.shape[1], cols.max() + pad + 1)
    P, O = part[r0:r1, c0:c1], other[r0:r1, c0:c1]
    # concept foreground, resampled smoothly (cubic on the darkness) into the mask frame
    score = fg_score(art_rgb)
    rr, cc = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    ax, ay = reg.to_art(cc + 0.5, rr + 0.5)
    f = ndi.map_coordinates(score, [ay - 0.5, ax - 0.5], order=3, mode="nearest") > 0.5
    # drop thin lines (grid axes, hairlines) narrower than ~thin_art_px concept pixels
    f = ndi.binary_opening(f, structure=_disk(thin_art_px / reg.scale / 2 + 1))
    reach = ndi.binary_dilation(P, structure=_disk(reach_m * ppm / 2), iterations=2)
    out = f & reach & ~O
    # keep what is connected to the mask's core, then fill pin holes (visor slits are dark: fg anyway)
    core = ndi.binary_erosion(P, structure=_disk(0.01 * ppm))
    lab, n = ndi.label(out)
    keep = np.unique(lab[core & out])
    out = np.isin(lab, keep[keep > 0])
    out = ndi.binary_fill_holes(out)
    full = np.zeros_like(part)
    full[r0:r1, c0:c1] = out
    added, removed = (full & ~part).sum(), (part & ~full).sum()
    iou = (full & part).sum() / max((full | part).sum(), 1)
    return full, dict(added_px=int(added), removed_px=int(removed), iou_vs_mask=float(iou),
                      bbox=(int(r0), int(r1), int(c0), int(c1)))


def edge_offsets(old, new, ppm=1600.0):
    """Signed distance (m) from each old-mask edge pixel to the new mask's edge: + where the new
    outline is outside the old. Summary stats only."""
    edge = old & ~ndi.binary_erosion(old)
    d_out = ndi.distance_transform_edt(~new)       # distance to the new mask, for points outside it
    d_in = ndi.distance_transform_edt(new)         # distance to new background, for points inside it
    s = np.where(new[edge], d_in[edge], -d_out[edge]) / ppm
    return s


def trace_views(art_path, mask_dir, cams_json, rgb, views=("front", "back", "side"), out_dir=None, **kw):
    """Register each view, trace the part, optionally write <out_dir>/<View>.png (part colour only)
    and registration.json. -> {view: (mask, reg, info)}."""
    cams = json.load(open(cams_json))
    art = load_rgb(art_path)
    res = {}
    for v in views:
        rgba = load_rgba(os.path.join(mask_dir, f"{MASK_FILES[v]}.png"))
        reg = register(v, art, rgba, cams[v])
        m, info = trace_part(art, rgba, reg, rgb, **kw)
        res[v] = (m, reg, info)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        meta = {}
        for v, (m, reg, info) in res.items():
            img = np.zeros(m.shape + (4,), np.uint8)
            img[m] = (*rgb, 255)
            Image.fromarray(img).save(os.path.join(out_dir, f"{MASK_FILES[v]}.png"))
            meta[v] = dict(registration=reg.as_dict(), **info)
        json.dump(meta, open(os.path.join(out_dir, "trace.json"), "w"), indent=1)
    return res
