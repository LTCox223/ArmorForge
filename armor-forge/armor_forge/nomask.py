"""Helmet outlines from a concept sheet that has no piece masks and no side view.

For sheets like Thornwood (front and back panels on a metric grid, nothing else):

  * calibrate:   the panel's evenly spaced horizontal grid lines give the scale (grid_m apart,
                 the bottom one is 0 m); the figure's mirror-symmetry axis gives u = 0;
  * cut:         the helmet is the top of the figure down to the neck: below the head's widest row,
                 the biggest step inward of the figure's central width (the rim's bottom edge)
                 before the shoulders;
  * side view:   there is none, so depth is borrowed from the head target (SkullTarget sliced into
                 head_target.json): at each height the helmet keeps the target's depth-to-width
                 ratio and centre. This is a stand-in; a real side view replaces it.

Masks are written in the same artboard frame as real masks (4096 px, origin (2048, 3712),
1600 px/m), so helmet.py and helmet_parts.py read them unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from . import concept as cc

ARTBOARD = 4096
ORIGIN = (2048.0, 3712.0)
PPM = 1600.0
HELMET_RGB = (244, 93, 89)


@dataclass
class PanelCal:
    ppm: float        # concept pixels per metre
    z0_row: float     # concept row of z = 0
    cx: float         # concept column of u = 0

    def registration(self):
        """Artboard -> concept mapping, as concept.Registration."""
        s = self.ppm / PPM
        return cc.Registration(s, self.cx - s * ORIGIN[0], self.z0_row - s * ORIGIN[1], float("nan"))


def figure_mask(rgb):
    return cc.fg_score(rgb) > 0.5


def grid_rows(rgb, x0, x1, depth=4.0):
    """Rows of faint horizontal grid lines in a panel (dips in the background brightness)."""
    lum = rgb[:, x0:x1].mean(axis=(1, 2))
    rows = [y for y in range(3, len(lum) - 3)
            if lum[y] < lum[y - 3] - depth and lum[y] < lum[y + 3] - depth and lum[y] > 200]
    # merge 1-2 px doubles
    out = []
    for y in rows:
        if out and y - out[-1][-1] <= 2:
            out[-1].append(y)
        else:
            out.append([y])
    return [float(np.mean(g)) + 0.5 for g in out]


def symmetry_axis(fig, x0, x1, y0, y1, search=None):
    m = fig[y0:y1, x0:x1]
    cols = np.arange(m.shape[1])
    lo, hi = search or (m.shape[1] * 0.3, m.shape[1] * 0.7)
    best = (-1, 0.0)
    for c in np.arange(lo, hi, 0.5):
        mc = np.round(2 * c - cols).astype(int)
        ok = (mc >= 0) & (mc < m.shape[1])
        s = (m[:, cols[ok]] & m[:, mc[ok]]).sum() / max(m[:, cols[ok]].sum(), 1)
        if s > best[0]:
            best = (s, c)
    return x0 + best[1] + 0.5


def regular_lattice(rows, tol=2.0):
    """The longest evenly spaced chain of rows, walking up from the bottom one. -> (spacing, chain)."""
    rows = sorted(rows)
    if len(rows) < 3:
        raise ValueError("too few grid lines")
    diffs = np.diff(rows)
    best = (0, None, [])
    for d in diffs:
        if d < 10:
            continue
        chain = [rows[-1]]
        y = rows[-1]
        while True:
            near = [r for r in rows if abs(r - (y - d)) <= tol]
            if not near:
                break
            y = near[0]
            chain.append(y)
        if len(chain) > best[0]:
            best = (len(chain), d, chain)
    _, d, chain = best
    if d is None or len(chain) < 3:
        raise ValueError("no regular grid found")
    return float((chain[0] - chain[-1]) / (len(chain) - 1)), chain


def calibrate_panel(rgb, x0, x1, bg_cols=None, grid_m=0.25, band=50):
    """bg_cols: a column range inside the panel clear of the figure (for the grid lines); by default
    every band-wide strip of the panel is tried and the longest regular grid wins. The grid's bottom
    line is z = 0; lines are grid_m apart."""
    strips = [bg_cols] if bg_cols else [(c, c + band) for c in range(x0 + 10, x1 - band - 10, band // 2)]
    best = None
    for st in strips:
        try:
            sp, ch = regular_lattice(grid_rows(rgb, *st))
        except ValueError:
            continue
        if best is None or len(ch) > len(best[1]):
            best = (sp, ch)
    if best is None:
        raise ValueError("no regular grid found in the panel")
    spacing, chain = best
    ppm = spacing / grid_m
    bottom = max(chain)
    fig = figure_mask(rgb)
    cx = symmetry_axis(fig, x0, x1, int(min(chain)), int(bottom))
    return PanelCal(ppm, bottom, cx)


def central_widths(fig, cx, y0, y1):
    """(rows, left, right) of the connected run through column cx, per row (nan where empty)."""
    c = int(round(cx))
    ys = np.arange(y0, y1)
    L = np.full(len(ys), np.nan)
    R = np.full(len(ys), np.nan)
    for i, y in enumerate(ys):
        row = fig[y]
        if not row[c]:
            continue
        lab, _ = ndi.label(row)
        idx = np.flatnonzero(lab == lab[c])
        L[i], R[i] = idx.min(), idx.max()
    return ys, L, R


def panel_figure(fig, x0, x1):
    """The figure in one panel: its largest connected component (drops titles and labels)."""
    m = np.zeros_like(fig)
    m[:, x0:x1] = fig[:, x0:x1]
    lab, n = ndi.label(m)
    if n == 0:
        return m
    return lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))


def head_cut(fig, cal: PanelCal, shoulder_factor=2.0, z_search=(1.5, 2.4)):
    """Concept row of the helmet's bottom edge: in the lower half between the head's widest row and
    the shoulders, where the central width steps inward the most (the rim's bottom edge). fig: the panel's figure (panel_figure)."""
    y0 = max(0, int(cal.z0_row - z_search[1] * cal.ppm))
    y1 = int(cal.z0_row - z_search[0] * cal.ppm)
    ys, L, R = central_widths(fig, cal.cx, y0, y1)
    w = R - L
    top = int(np.flatnonzero(~np.isnan(w))[0])
    # shoulders: first row where the run is much wider than the head so far
    head_w = np.nanmax(w[top:top + int(0.25 * cal.ppm)])
    wide = np.flatnonzero((w > shoulder_factor * head_w) & (np.arange(len(w)) > top))
    sh = wide[0] if len(wide) else len(w) - 1
    upper = top + max(2, int(0.6 * (sh - top)))           # the head's widest row is in its upper part
    widest = top + int(np.nanargmax(w[top:upper]))
    seg = np.nan_to_num(w[widest:sh], nan=0.0)
    drop = seg[:-1] - seg[1:]                     # width lost going one row down
    lo = int(0.02 * cal.ppm)                      # stay clear of the shoulder row itself
    start = len(drop) // 2                        # the rim is in the lower half (the face may taper above)
    end = max(start + 1, len(drop) - lo)
    k = start + int(np.argmax(drop[start:end]))
    return float(ys[widest + k + 1]), dict(top_row=float(ys[top]), widest_row=float(ys[widest]),
                                          shoulder_row=float(ys[sh]))


def helmet_mask(rgb, cal: PanelCal, cut_row, thin_px=1.3):
    """Helmet silhouette in the artboard frame: the figure above the cut, its largest component,
    holes filled, thin lines (grid) opened away."""
    reg = cal.registration()
    r0 = max(0, int(ORIGIN[1] - 2.3 * PPM))
    r1 = int(ORIGIN[1] - 1.4 * PPM)
    c0, c1 = int(ORIGIN[0] - 0.4 * PPM), int(ORIGIN[0] + 0.4 * PPM)
    rr, cc_ = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    ax, ay = reg.to_art(cc_ + 0.5, rr + 0.5)
    f = ndi.map_coordinates(cc.fg_score(rgb), [ay - 0.5, ax - 0.5], order=3, mode="nearest") > 0.5
    f &= ay < cut_row
    f = ndi.binary_opening(f, structure=cc._disk(thin_px / reg.scale / 2 + 1))
    lab, n = ndi.label(f)
    if n == 0:
        raise ValueError("no figure above the cut")
    keep = 1 + int(np.argmax(ndi.sum(f, lab, range(1, n + 1))))      # the helmet, not stray text
    m = ndi.binary_fill_holes(lab == keep)
    out = np.zeros((ARTBOARD, ARTBOARD), bool)
    out[r0:r1, c0:c1] = m
    return out


def load_head_target(path):
    d = json.load(open(path))
    a = np.array(d["rows"], float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def synthetic_side(front, back, head_target):
    """Side silhouette (artboard frame) from the front/back half-width and the head target's
    depth-to-width ratio and centre at each height."""
    zt, at, yft, ybt = head_target
    k = (0.5 * (ybt - yft)) / np.maximum(at, 1e-3)
    yc = 0.5 * (yft + ybt)
    ok = at > 0.03                              # ratio is unreliable where the target pinches shut
    zt, k, yc = zt[ok], k[ok], yc[ok]
    side = np.zeros_like(front)
    rows = np.flatnonzero(front.any(1) | back.any(1))
    for r in rows:
        z = (ORIGIN[1] - r - 0.5) / PPM
        half = []
        for m in (front, back):
            cols = np.flatnonzero(m[r])
            if len(cols):
                u = (cols + 0.5 - ORIGIN[0]) / PPM
                half.append(0.5 * (u.max() - u.min()))
        a = max(half)
        d = np.interp(z, zt, k) * a
        c = np.interp(z, zt, yc)
        lo, hi = int(ORIGIN[0] + (c - d) * PPM), int(ORIGIN[0] + (c + d) * PPM)
        side[r, lo:hi + 1] = True
    return side


def write_masks(out_dir, masks, regs, rgb=HELMET_RGB):
    """Front/Back/Side PNGs in the artboard frame, cams.json and trace.json (registrations)."""
    os.makedirs(out_dir, exist_ok=True)
    for v, m in masks.items():
        img = np.zeros(m.shape + (4,), np.uint8)
        img[m] = (*rgb, 255)
        Image.fromarray(img).save(os.path.join(out_dir, f"{cc.MASK_FILES[v]}.png"))
    json.dump({v: list(ORIGIN) for v in ("front", "back", "side", "side_noarm")},
              open(os.path.join(out_dir, "cams.json"), "w"))
    meta = {v: dict(registration=r.as_dict()) for v, r in regs.items()}
    meta["side_stand_in"] = True          # the side mask is built from the head target, not drawn
    json.dump(meta, open(os.path.join(out_dir, "trace.json"), "w"), indent=1)
