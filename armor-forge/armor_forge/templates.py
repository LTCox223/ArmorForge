"""Parametric piece templates (ArmorForge v3).

Each piece type is a clean parametric surface. The masks and the body only set a handful of smoothed
parameters per template - edge lines, hem planes/curves, depth, stand-off - each one a robust low-order
fit, never raw pixel outlines. That keeps hems straight, rims smooth and pieces seated on the body.

Occlusion: masks label only the visible surface, so an edge that runs into a piece drawn on top of it
(OVER) is not a real edge. Such sides fall back to the body; such hems are tucked under the cover.

Clearance is one smooth offset per piece, the smallest linear-in-z function that clears the body (and
the pieces it lies over), so pushing the piece out never introduces per-vertex spikes.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

from .fit import NAMES, SPECS, Fitter, hull_radius, shell, superellipse

# Fixed overlap order.
# From the brief: pauldron > lame > sleeve, knee > greave, upper tasset > lower tasset,
# tabard > tassets and thigh plates. Assumed (natural layering, to confirm): belt over cuirass, tabard
# and tassets (they hang from it); gorget over cuirass; helmet over gorget; greave over boot; pauldron
# and lame over cuirass; vambrace over sleeve; tassets over thigh plates.
#
# UNDER[pid] = pieces pid lies on top of: pid's cross-section clears them (built first).
UNDER = {
    1: {2},
    2: {3},
    4: {3, 5, 6},
    5: {3, 6},
    7: {6},
    8: {7},
    9: {3, 10, 11, 12},
    10: {11},
    12: {10, 11, 13},
    14: {15},
    15: {16},
}
# OVER[pid] = pieces drawn on top of pid where they meet (so pid's edge there is hidden, not real).
OVER = {}
for _top, _unders in UNDER.items():
    for _u in _unders:
        OVER.setdefault(_u, set()).add(_top)
OVER.setdefault(13, set()).update({10, 11})


# ----------------------------------------------------------------------------- robust fits
def robust_polyfit(x, y, deg, k=2.5, iters=4):
    """Least-squares polynomial with iterative outlier rejection (MAD). Returns coefficients."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    deg = min(deg, max(len(x) - 1, 0))
    m = np.ones(len(x), bool)
    c = np.polyfit(x, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, x)
        s = 1.4826 * np.median(np.abs(r[m])) + 1e-4
        m2 = np.abs(r) < k * s
        if m2.sum() <= deg + 2 or np.array_equal(m2, m):
            break
        m = m2
        c = np.polyfit(x[m], y[m], deg)
    return c


class Curve:
    """Robust low-order fit of y(z), evaluated with constant extension outside the measured range."""

    def __init__(self, z, y, deg):
        self.lo, self.hi = float(np.min(z)), float(np.max(z))
        self.c = robust_polyfit(z, y, deg)

    def __call__(self, z):
        return np.polyval(self.c, np.clip(z, self.lo, self.hi))


def robust_plane(x, y, z, k=2.5, iters=4, max_slope=1.2):
    """z = a + b x + c y, outlier-rejected; slopes clamped so a bad fit can't fold the piece."""
    A = np.c_[np.ones(len(x)), x, y]
    m = np.ones(len(x), bool)
    p = np.linalg.lstsq(A, z, rcond=None)[0]
    for _ in range(iters):
        r = z - A @ p
        s = 1.4826 * np.median(np.abs(r[m])) + 1e-4
        m2 = np.abs(r) < k * s
        if m2.sum() < 6 or np.array_equal(m2, m):
            break
        m = m2
        p = np.linalg.lstsq(A[m], z[m], rcond=None)[0]
    p[1:] = np.clip(p[1:], -max_slope, max_slope)
    return p


def envelope(z, need, deg=1):
    """Smallest polynomial g(z) of degree `deg` with g >= need and g >= 0 at every sample.

    Minimises the mean of g over the samples (a linear program), so the offset is only as large as
    the worst point requires and varies smoothly with height - never per vertex.
    """
    z, need = np.asarray(z, float), np.asarray(need, float)
    if not len(z) or need.max() <= 0:
        return lambda q: np.zeros_like(np.asarray(q, float))
    zc, zs = z.mean(), max(np.ptp(z) / 2, 1e-3)
    A = np.vander((z - zc) / zs, deg + 1, increasing=True)
    lb = np.maximum(need, 0.0)
    res = linprog(c=A.sum(0), A_ub=-A, b_ub=-lb, bounds=[(None, None)] * (deg + 1), method="highs")
    if not res.success:
        m = lb.max()
        return lambda q: np.full_like(np.asarray(q, float), m)
    p = res.x

    def g(q):
        t = (np.clip(np.asarray(q, float), z.min(), z.max()) - zc) / zs
        return np.maximum(np.vander(np.atleast_1d(t).ravel(), deg + 1, increasing=True) @ p, 0.0).reshape(np.shape(q))
    return g


# ----------------------------------------------------------------------------- mask measurements
def _world_cols(d, view):
    n = d.lab[view].shape[1]
    x = (np.arange(n) + 0.5 - d.ox) / d.ppm
    return -x if view == "Back" else x


def _is(g, r, c, ids):
    return 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) in ids


def spans(d, view, ids, half=0, over=()):
    """Per drawn row: z, low/high horizontal world coordinate, and whether each side is occluded."""
    g = d.lab[view]
    sel = np.isin(g, ids)
    xw = _world_cols(d, view)
    if half and view in ("Front", "Back"):
        sel = sel & (xw * half > 0)[None, :]
    out = []
    for r in np.flatnonzero(sel.any(axis=1)):
        c = np.flatnonzero(sel[r])
        if len(c) < 3:
            continue
        c0, c1 = c.min(), c.max()
        a = (xw[c0], _is(g, r, c0 - 1, over))
        b = (xw[c1], _is(g, r, c1 + 1, over))
        if a[0] > b[0]:
            a, b = b, a
        out.append(((d.oy - r - 0.5) / d.ppm, a[0], b[0], a[1], b[1]))
    if not out:
        return None
    z, lo, hi, olo, ohi = map(np.array, zip(*out))
    return z, lo, hi, olo.astype(bool), ohi.astype(bool)


def side_strips(d, view, ids, over=()):
    """Side view of hanging front/back panels: per row, the front strip (y0, y1) and back strip (y0, y1).

    The front strip is the run holding the row's most-forward pixel, the back strip the run holding its
    rearmost pixel. A strip's depth is how far the panel wraps back from its leading edge."""
    g = d.lab[view]
    sel = np.isin(g, ids)
    yw = _world_cols(d, view)
    out = []
    for r in np.flatnonzero(sel.any(axis=1)):
        c = np.flatnonzero(sel[r])
        if len(c) < 3:
            continue
        brk = np.flatnonzero(np.diff(c) > 2)
        starts, ends = np.r_[c[0], c[brk + 1]], np.r_[c[brk], c[-1]]
        f, b = (starts[0], ends[0]), (starts[-1], ends[-1])
        out.append(((d.oy - r - 0.5) / d.ppm, yw[f[0]], yw[f[1]], yw[b[0]], yw[b[1]], len(starts) > 1))
    if not out:
        return None
    return tuple(map(np.array, zip(*out)))


def col_edges(d, view, ids, half=0, over=(), trim=0.1):
    """Per drawn column (inner 1-2*trim of the span): world u, bottom/top z and their occlusion."""
    g = d.lab[view]
    sel = np.isin(g, ids)
    xw = _world_cols(d, view)
    if half and view in ("Front", "Back"):
        sel = sel & (xw * half > 0)[None, :]
    cols = np.flatnonzero(sel.any(axis=0))
    if len(cols) < 4:
        return None
    k = int(len(cols) * trim)
    cols = cols[k:len(cols) - k] if len(cols) - 2 * k >= 4 else cols
    out = []
    for c in cols:
        rows = np.flatnonzero(sel[:, c])
        rb, rt = rows.max(), rows.min()
        out.append((xw[c], (d.oy - rb - 0.5) / d.ppm, _is(g, rb + 1, c, over),
                    (d.oy - rt - 0.5) / d.ppm, _is(g, rt - 1, c, over)))
    u, zb, ob, zt, ot = map(np.array, zip(*out))
    return u, zb, ob.astype(bool), zt, ot.astype(bool)


# ----------------------------------------------------------------------------- piece frames
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Frame:
    """A piece's local axes: t along the axis `a`, u along `e1` (both in the x-z plane, so the front
    view sees them undistorted), v along world y (depth, seen by the side view)."""

    def __init__(self, o, a, e1, window=0.2, flood=False):
        self.o, self.a = np.asarray(o, float), _unit(a)
        e1 = np.asarray(e1, float) - np.dot(e1, self.a) * self.a
        self.e1, self.e2 = _unit(e1), np.array([0.0, 1.0, 0.0])
        self.window, self.flood = window, flood
        self.vertical = abs(self.a[2]) > 0.999
        self.key = tuple(np.round(np.r_[self.o, self.a], 4))

    def local(self, P):
        d = np.asarray(P, float) - self.o
        return d @ self.a, d @ self.e1, d @ self.e2

    def world(self, t, u, v):
        return self.o + t * self.a + u * self.e1 + v * self.e2

    def xz(self, x, z):
        """Front-view (world x, z) -> (t, u)."""
        dx, dz = x - self.o[0], z - self.o[2]
        return dx * self.a[0] + dz * self.a[2], dx * self.e1[0] + dz * self.e1[2]


VERTICAL = Frame((0, 0, 0), (0, 0, 1), (1, 0, 0))


def _px(d, view, x, z):
    c = (x if view != "Back" else -x) * d.ppm + d.ox - 0.5
    r = d.oy - z * d.ppm - 0.5
    return int(round(r)), int(round(c))


def _front_pixels(d, view, ids, half):
    g = d.lab[view]
    rows, cols = np.nonzero(np.isin(g, ids))
    x, z = d.world(view, rows, cols)
    if half:
        k = x * half > 0
        x, z = x[k], z[k]
    return x, z


def frame_spans(d, views, ids, half, over, fr):
    """Front/back views in a piece frame: per t-bin, lowest/highest u and whether each side is hidden."""
    out = []
    step = 1.5 / d.ppm
    for view in views:
        x, z = _front_pixels(d, view, ids, half)
        if len(x) < 20:
            continue
        g = d.lab[view]
        t, u = fr.xz(x, z)
        b = np.floor(t * d.ppm).astype(int)
        o = np.lexsort((u, b))
        b, t, u, x, z = b[o], t[o], u[o], x[o], z[o]
        cut = np.flatnonzero(np.diff(b)) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(b)]):
            if e - s < 3:
                continue
            ends = []
            for i, sgn in ((s, -1), (e - 1, 1)):
                px = x[i] + sgn * step * fr.e1[0]
                pz = z[i] + sgn * step * fr.e1[2]
                ends.append(_is(g, *_px(d, view, px, pz), over))
            out.append((t[s:e].mean(), u[s], u[e - 1], ends[0], ends[1]))
    if not out:
        return None
    t, lo, hi, olo, ohi = map(np.array, zip(*out))
    return t, lo, hi, olo.astype(bool), ohi.astype(bool)


def frame_edges(d, view, ids, half, over, fr, trim=0.1):
    """Front/back view in a piece frame: per u-bin, lowest/highest t and whether each end is hidden."""
    x, z = _front_pixels(d, view, ids, half)
    if len(x) < 20:
        return None
    g = d.lab[view]
    step = 1.5 / d.ppm
    t, u = fr.xz(x, z)
    b = np.floor(u * d.ppm).astype(int)
    o = np.lexsort((t, b))
    b, t, u, x, z = b[o], t[o], u[o], x[o], z[o]
    cut = np.flatnonzero(np.diff(b)) + 1
    groups = list(zip(np.r_[0, cut], np.r_[cut, len(b)]))
    k = int(len(groups) * trim)
    groups = groups[k:len(groups) - k] if len(groups) - 2 * k >= 4 else groups
    out = []
    for s, e in groups:
        ends = []
        for i, sgn in ((s, -1), (e - 1, 1)):
            px, pz = x[i] + sgn * step * fr.a[0], z[i] + sgn * step * fr.a[2]
            ends.append(_is(g, *_px(d, view, px, pz), over))
        out.append((u[s:e].mean(), t[s], ends[0], t[e - 1], ends[1]))
    if len(out) < 4:
        return None
    u, tb, ob, tt, ot = map(np.array, zip(*out))
    return u, tb, ob.astype(bool), tt, ot.astype(bool)


def side_depth(d, view, ids, fr, ts, band=0.02, limb_ids=(4, 5, 6, 7, 8)):
    """Depth extents of a limb piece in the side view, at each axis station.

    Measured across the limb's own direction in that view (PCA of all drawn limb pieces), not along a
    horizontal band: a forearm reaching forward would otherwise read as far deeper than it is."""
    g = d.lab[view]
    rows, cols = np.nonzero(np.isin(g, limb_ids))
    ya, za = d.world(view, rows, cols)
    A = np.c_[ya, za]
    ca = A.mean(0)
    s = np.linalg.svd(A - ca, full_matrices=False)[2][0]
    s = s if s[1] < 0 else -s                      # along the limb, pointing down like the arm frame
    n = np.array([-s[1], s[0]])
    rows, cols = np.nonzero(np.isin(g, ids))
    y, z = d.world(view, rows, cols)
    P = np.c_[y, z] - ca
    along, across = P @ s, P @ n
    sign = 1.0 if n[0] >= 0 else -1.0             # report depth increasing toward +y like world y
    out = []
    for t in ts:
        zc = fr.o[2] + t * fr.a[2]
        k = (zc - ca[1]) / s[1]                    # station on the drawn limb line at that height
        m = np.abs(along - k) <= band
        if m.sum() >= 6:
            lo, hi = np.percentile(across[m] * sign, [2, 98])
            yc = ca[0] + k * s[0] - fr.o[1]            # the drawn limb's centre line, in frame depth
            out.append((t, yc + lo, yc + hi))
    if len(out) < 3:
        return None
    t, lo, hi = map(np.array, zip(*out))
    return t, lo, hi, np.zeros(len(t), bool), np.zeros(len(t), bool)


def _flood(p, seed, reach=0.06):
    """Points connected to `seed` through steps shorter than `reach` (one limb out of a plane cut)."""
    if not len(p):
        return p
    keep = np.zeros(len(p), bool)
    keep[np.argmin(np.linalg.norm(p - seed, axis=1))] = True
    while True:
        dist = np.min(np.linalg.norm(p[:, None, :] - p[None, keep, :], axis=2), axis=1)
        new = (dist < reach) & ~keep
        if not new.any():
            return p[keep]
        keep |= new


def _orient_outward(pts, faces, axis_of):
    """Flip all faces together so that, on average, they face away from the piece's axis."""
    tot = 0.0
    for f in faces:
        p = pts[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        c = p.mean(0)
        tot += float(n @ (c - axis_of(c)))
    return [f[::-1] for f in faces] if tot < 0 else faces


def solidify_rims(pts, faces, thickness):
    """Give every open edge a rim wall `thickness` deep, so the piece has real edge thickness."""
    from .fit import _vnormals
    pts = np.asarray(pts, float)
    if not thickness:
        return pts, faces
    n = _vnormals(pts, faces)
    edge = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            k = (min(a, b), max(a, b))
            if k in edge:
                del edge[k]
            else:
                edge[k] = (a, b)
    if not edge:
        return pts, faces
    m = len(pts)
    pts = np.vstack([pts, pts - thickness * n])
    return pts, faces + [[b, a, a + m, b + m] for a, b in edge.values()]


class TemplateFitter(Fitter):
    """Every piece from a parametric template: tubes (any axis) with optional plate sector and end caps,
    and hanging cloth panels. Masks and body only set smoothed parameters (see module docstring)."""

    tuck = 0.025          # how far a hidden hem runs under the piece covering it, m
    harmonics = 3         # smoothness of a limb outline around its axis (Fourier terms)
    sector_margin = 0.02  # how far past the drawn span a plate's sector may reach, m

    # piece type per part: tube on a frame, with an optional plate sector and end cap. Per piece TYPE,
    # never per armor set: bodies = mannequin parts it wraps; frame = vertical / upper arm / forearm.
    TUBES = {
        16: dict(bodies=("feet", "body"), half=True, end="flat"),
        15: dict(bodies=("body",), half=True),
        14: dict(bodies=("body",), half=True, sector=True),
        13: dict(bodies=("body",), half=True),
        3: dict(bodies=("armless",)),
        11: dict(bodies=("body",), half=True, sector=True),
        10: dict(bodies=("body",), half=True, sector=True),
        9: dict(bodies=("armless",)),
        2: dict(bodies=("armless", "head")),
        1: dict(bodies=("head",), end="dome"),
        6: dict(bodies=("body",), half=True, frame="upper"),
        5: dict(bodies=("body",), half=True, frame="upper", shell="drawn", open_under=0.35),
        4: dict(bodies=("body",), half=True, frame="upper", shell="drawn", open_under=0.35, end="dome_prox"),
        7: dict(bodies=("body",), half=True, frame="fore"),
        8: dict(bodies=("body",), half=True, frame="hand", end="dome", hug=True),
    }
    ORDER = (16, 15, 14, 13, 3, 11, 10, 12, 9, 2, 1, 6, 5, 4, 7, 8)

    def __init__(self, drawing, bodies):
        super().__init__(drawing, bodies)
        self._sec = {}
        self._frames = {}

    def section(self, body, z):
        key = (body, round(float(z), 3))
        if key not in self._sec:
            self._sec[key] = self.b[body].section(key[1])
        return self._sec[key]

    _fp = None            # footprint box (u0, u1, v0, v1) of the piece being built; see tube()

    def fsection(self, bodies, fr, t, half, obstacles=None):
        """Body (+ obstacle pieces) cut by the plane at station t of frame fr, as local (u, v) points,
        limited to the current piece's footprint: a piece only wraps what lies inside its drawing."""
        uv = self._raw_section(bodies, fr, t, half)
        if obstacles is not None and len(obstacles):
            ot, ou, ov = obstacles
            m = np.abs(ot - t) < 0.015
            if m.any():
                uv = np.vstack([uv, np.c_[ou[m], ov[m]]])
        if self._fp is not None and len(uv):
            u0, u1, v0, v1 = self._fp
            uv = uv[(uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)]
        return uv

    def _raw_section(self, bodies, fr, t, half):
        key = (bodies, fr.key, half, round(float(t), 3))
        if key not in self._sec:
            pts = []
            for b in bodies:
                if fr.vertical:
                    p = self.section(b, fr.o[2] + key[3])
                else:
                    p = self.b[b].section(key[3], fr.a, fr.o)
                if half and len(p):
                    p = p[p[:, 0] * half > 0]
                pts.append(p)
            p = np.vstack(pts) if pts else np.zeros((0, 3))
            _, u, v = fr.local(p)
            uv = np.c_[u, v]
            if not fr.vertical and len(uv):         # limb frames: only the limb around the axis
                cv = self._cv.get(fr.key, 0.0)
                uv = uv[(np.abs(uv[:, 0]) < fr.window) & (np.abs(uv[:, 1] - cv) < fr.window)]
                if fr.flood and len(uv):
                    uv = _flood(uv, np.array([0.0, cv]))
            self._sec[key] = uv
        return self._sec[key]

    _cv = {}

    # ---- frames
    def frame(self, kind, half, pid):
        if kind is None:
            return VERTICAL
        if kind == "hand":
            return self._hand_frame(half)
        key = ("arm", half)          # the mannequin's arm is near straight in this pose: one axis
        if key not in self._frames:
            arm = self._arm_vertices(half)
            P = arm[:, [0, 2]]
            c = P.mean(0)
            A = np.linalg.svd(P - c, full_matrices=False)[2][0]
            A = A if A[0] * half > 0 else -A
            o, a = np.array([c[0], float(arm[:, 1].mean()), c[1]]), np.array([A[0], 0.0, A[1]])
            e1 = np.array([-A[1], 0.0, A[0]])
            e1 = e1 if e1[2] > 0 else -e1
            fr = Frame(o, a, e1, window=0.11, flood=True)
            self._cv[fr.key] = 0.0
            self._frames[key] = fr
        return self._frames[key]

    def drawn_frame(self, half):
        """The concept's own arm axis (all drawn arm pieces), used to MEASURE arm pieces. The drawn arm
        need not have the mannequin's angle; measuring along it keeps widths true instead of smearing a
        tilted strip across the mannequin's axis."""
        key = ("drawn_arm", half)
        if key not in self._frames:
            body = self.frame("upper", half, None)
            x, z = [], []
            for view in ("Front", "Back"):
                xx, zz = _front_pixels(self.d, view, [4, 5, 6, 7, 8], half)
                x.append(xx)
                z.append(zz)
            P = np.c_[np.concatenate(x), np.concatenate(z)]
            c = P.mean(0)
            A = np.linalg.svd(P - c, full_matrices=False)[2][0]
            A = A if A[0] * half > 0 else -A
            e1 = np.array([-A[1], 0.0, A[0]])
            e1 = e1 if e1[2] > 0 else -e1
            self._frames[key] = Frame((c[0], body.o[1], c[1]), (A[0], 0.0, A[1]), e1)
        return self._frames[key]

    def arm_map(self, half):
        """Drawn arm station -> mannequin arm station (both measured from the mannequin's shoulder).

        The concept's arm need not be as long as the mannequin's (here its hands reach ~8 cm further),
        so stations are scaled: the shoulder stays put and the drawn hand end lands on the mannequin's
        hand end. Every arm piece then sits at the same fraction of the arm as in the drawing."""
        key = ("armmap", half)
        if key not in self._frames:
            fr, fd = self.frame("upper", half, None), self.drawn_frame(half)
            arm = self._arm_vertices(half)
            ta, _, _ = fr.local(arm)
            ts, hb = float(np.percentile(ta, 1)), float(np.percentile(ta, 99.5))
            x, z = [], []
            for view in ("Front", "Back"):
                xx, zz = _front_pixels(self.d, view, [8], half)
                x.append(xx)
                z.append(zz)
            td, _ = fd.xz(np.concatenate(x), np.concatenate(z))
            hd = float(np.percentile(td, 99.5)) + float(np.dot(fd.o - fr.o, fr.a))
            k = (hb - ts) / (hd - ts) if hd - ts > 0.05 else 1.0
            self._frames[key] = (ts, k)
        ts, k = self._frames[key]
        return lambda t: ts + (np.asarray(t, float) - ts) * k

    def _arm_vertices(self, half):
        """The mannequin's arm: full-body vertices sticking out sideways past the armless torso, keeping
        the largest edge-connected group (not stray bits of leg)."""
        key = ("armv", half)
        if key not in self._frames:
            full, torso = self.b["body"].v, self.b["armless"].v
            zb = np.linspace(torso[:, 2].min(), torso[:, 2].max(), 60)
            wide = np.array([np.abs(torso[np.abs(torso[:, 2] - z) < 0.03, 0]).max(initial=0.0) for z in zb])
            sel = full[:, 0] * half > np.interp(full[:, 2], zb, wide) + 0.02
            parent = np.arange(len(full))

            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            for a_, b_ in self.b["body"].e:
                if sel[a_] and sel[b_]:
                    parent[root(a_)] = root(b_)
            roots = np.array([root(i) for i in range(len(full))])
            ids, cnt = np.unique(roots[sel], return_counts=True)
            self._frames[key] = full[sel & (roots == ids[np.argmax(cnt)])]
        return self._frames[key]

    limb_overlap = 0.012   # how far a limb piece runs under the next one along the limb, m
    limb_hem_slope = 0.35  # limb rims sit near square to the limb (max t change per m across it)

    def _hand_frame(self, half):
        """The mannequin's hand (arm vertices past the drawn wrist), on its own axis: in a grip pose the
        fist sits off the forearm's line, so the forearm axis would cut part of the hand away."""
        key = ("hand", half)
        if key not in self._frames:
            arm_fr = self.frame("upper", half, None)
            arm = self._arm_vertices(half)
            ta, _, _ = arm_fr.local(arm)
            wrist = self._drawn_start(8, half)
            hand = arm[ta > (wrist if wrist is not None else np.percentile(ta, 70)) - 0.01]
            P = hand[:, [0, 2]]
            c = P.mean(0)
            A = np.linalg.svd(P - c, full_matrices=False)[2][0]
            A = A if np.dot(A, arm_fr.a[[0, 2]]) > 0 else -A
            e1 = np.array([-A[1], 0.0, A[0]])
            e1 = e1 if e1[2] > 0 else -e1
            fr = Frame((c[0], float(hand[:, 1].mean()), c[1]), (A[0], 0.0, A[1]), e1, window=0.13, flood=True)
            self._cv[fr.key] = 0.0
            self._frames[key] = fr
        return self._frames[key]

    def _drawn_start(self, pid, half):
        """Where piece pid begins along the mannequin's arm (drawn, mapped like every arm station)."""
        fr, fd = self.frame("upper", half, None), self.drawn_frame(half)
        x, z = [], []
        for view in ("Front", "Back"):
            xx, zz = _front_pixels(self.d, view, [pid], half)
            x.append(xx)
            z.append(zz)
        x, z = np.concatenate(x), np.concatenate(z)
        if len(x) < 20:
            return None
        td, _ = fd.xz(x, z)
        return float(self.arm_map(half)(np.percentile(td, 2) + float(np.dot(fd.o - fr.o, fr.a))))

    def _limb_offsets(self, fr, half, ts):
        """Where the mannequin's limb really is, per station: section centroid (u, v) in the frame."""
        cu, cv = [], []
        for t in ts:
            uv = self._raw_section(("body",), fr, t, half)
            if len(uv) >= 3:
                cu.append(uv[:, 0].mean())
                cv.append(uv[:, 1].mean())
        return (float(np.median(cu)) if cu else 0.0), (float(np.median(cv)) if cv else 0.0)

    def _obstacles(self, pid, half, fr):
        pts = [pv for name, (pv, _) in self.parts.items()
               if int(name[:2]) in UNDER.get(pid, ()) and (not half or np.mean(pv[:, 0]) * half > -0.02)]
        if not pts:
            return None
        P = np.vstack(pts)
        return fr.local(P)

    # ---- the tube template
    def tube(self, pid, half=0, **kw):
        try:
            return self._tube(pid, half, **kw)
        finally:
            self._fp = None

    def _footprint(self, lo, hi, olo, ohi, sd, cu, cv, margin=0.03, reach=0.25):
        """Box the piece wraps: the drawn span on each side that is mostly visible (+ margin); a side
        that is mostly hidden under a covering piece is open (up to `reach` from the centre)."""
        def side(vals, occ, out, c):
            vis = ~occ
            if vis.sum() >= 4 and vis.mean() >= 0.5:
                return float(np.percentile(vals[vis], 2 if out < 0 else 98)) + out * margin
            return c + out * reach
        return (side(lo, olo, -1, cu), side(hi, ohi, +1, cu), side(sd[1], sd[3], -1, cv), side(sd[2], sd[4], +1, cv))

    def _tube(self, pid, half=0, frame=None, bodies=("body",), sector=False, end=None, hug=False, slices=16,
              shell="body", open_under=None, exponent=2.3):
        """shell="body": cross-section = body (+under-pieces) outline, grown to the drawing where wider.
        shell="drawn": cross-section = a rounded box spanning the drawn extents (for pieces that stand
        off the body, like pauldrons), still pushed clear of body and under-pieces.
        open_under: leave the underside open (keep angles whose 'up' component >= -open_under).
        end: "dome" closes the far end, "dome_prox" the near (shoulder) end, "flat" the near end."""
        s = SPECS.get(pid)
        pad = s.pad if s else 0.006
        thick = (s.thick if s and s.thick else 0.006)
        spacing = s.spacing if s else 0.04
        taper = s.taper if s else 0.0
        d, over = self.d, OVER.get(pid, set())
        fr = self.frame(frame, half, pid)
        side_view = "Side" if frame else "Side_Without_Arms"
        # measure in the drawing's own frame (for limbs: the concept's arm axis), then map onto the body
        fm = fr if fr.vertical else self.drawn_frame(half)
        f = frame_spans(d, ("Front", "Back"), [pid], half, over, fm)
        if f is None:
            return
        tq, lo, hi, olo, ohi = f
        t0, t1 = np.percentile(tq, [0.5, 99.5])
        if fr.vertical:
            sd = spans(d, side_view, [pid], 0, over)
        else:
            sd = side_depth(d, side_view, [pid], fm, np.linspace(t0, t1, 12))
        if sd is None:
            return
        edges = {v: frame_edges(d, v, [pid], half, over, fm) for v in ("Front", "Back")}
        if not fr.vertical:
            # a limb piece takes its length and thickness from the drawing, not its position or angle:
            # the concept's arm need not sit where the mannequin's does. Slide the drawn stations onto
            # the mannequin's axis (same distances from the shoulder) and re-centre the spans on it.
            arm_fr = self.frame("upper", half, None)
            shift = float(np.dot(fm.o - arm_fr.o, arm_fr.a))
            R = self.arm_map(half)
            # drawn arm station -> mannequin arm station -> this piece's own frame (hand frame differs)
            conv = float(np.dot(arm_fr.a, fr.a)), float(np.dot(arm_fr.o - fr.o, fr.a))

            def M(t):
                return R(np.asarray(t, float) + shift) * conv[0] + conv[1]
            tq, t0, t1 = M(tq), float(M(t0)), float(M(t1))
            sd = (M(sd[0]), *sd[1:])
            edges = {v: (e[0], M(e[1]), e[2], M(e[3]), e[4]) if e is not None else None
                     for v, e in edges.items()}
            if shell == "drawn":        # stands off the torso's shoulder, which the concept does match
                du = dv = 0.0
            else:
                bu, bv = self._limb_offsets(fr, half, np.linspace(t0, t1, 7))
                du = bu - float(np.median((lo + hi) / 2))
                dv = bv - float(np.median((sd[1] + sd[2]) / 2))
            lo, hi = lo + du, hi + du
            sd = (sd[0], sd[1] + dv, sd[2] + dv, sd[3], sd[4])
            edges = {v: (e[0] + du, *e[1:]) if e is not None else None for v, e in edges.items()}
        deg = 3 if t1 - t0 > 0.3 else 2 if t1 - t0 > 0.2 else 1
        obstacles = self._obstacles(pid, half, fr)

        tf = np.linspace(min(t0, sd[0].min()) - self.tuck, max(t1, sd[0].max()) + self.tuck, 40)
        cu = float(np.median((lo + hi) / 2)) if fr.vertical else 0.0
        cv = float(np.median((sd[1] + sd[2]) / 2))
        if fr.vertical:
            self._cv[fr.key] = 0.0
        self._fp = None if hug else self._footprint(lo, hi, olo, ohi, sd, cu, cv)
        centre, base = self._limb(bodies, fr, tf, half, cu, cv, obstacles)
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)
        bu0, bu1, bv0, bv1 = [], [], [], []
        for t in tf:
            c, r = centre(t), base(t, ang) + pad
            bu0.append((c[0] + r * ca).min()), bu1.append((c[0] + r * ca).max())
            bv0.append((c[1] + r * sa).min()), bv1.append((c[1] + r * sa).max())
        bu0, bu1, bv0, bv1 = map(np.array, (bu0, bu1, bv0, bv1))
        if hug:
            U0, U1, V0, V1 = (Curve(tf, e, deg) for e in (bu0, bu1, bv0, bv1))
        else:
            U0 = self._side_curve(tq, lo, olo, tf, bu0, deg, -1)
            U1 = self._side_curve(tq, hi, ohi, tf, bu1, deg, +1)
            V0 = self._side_curve(sd[0], sd[1], sd[3], tf, bv0, deg, -1)
            V1 = self._side_curve(sd[0], sd[2], sd[4], tf, bv1, deg, +1)
        if shell == "drawn":
            cU, cV = (lambda t: centre(t)[0]), (lambda t: centre(t)[1])
            dd = min(deg, 2)
            U0 = self._drawn_curve(tq, lo, olo, hi, ohi, cU, dd)
            U1 = self._drawn_curve(tq, hi, ohi, lo, olo, cU, dd)
            V0 = self._drawn_curve(sd[0], sd[1], sd[3], sd[2], sd[4], cV, dd)
            V1 = self._drawn_curve(sd[0], sd[2], sd[4], sd[1], sd[3], cV, dd)

        # hems: planes in (u, v) through the outline ends of every view; hidden hems tucked under covers
        hem = {"b": [], "t": []}
        for view in ("Front", "Back"):
            e = edges[view]
            if e is None:
                continue
            u, tb, ob, tt, ot = e
            tb, tt = np.where(ob, tb - self.tuck, tb), np.where(ot, tt + self.tuck, tt)
            V = V0 if view == "Front" else V1
            hem["b"].append(np.c_[u, V(tb), tb])
            hem["t"].append(np.c_[u, V(tt), tt])
        if fr.vertical:
            e = col_edges(d, side_view, [pid], 0, over)
            if e is not None:
                v, tb, ob, tt, ot = e
                tb, tt = np.where(ob, tb - self.tuck, tb), np.where(ot, tt + self.tuck, tt)
                U = U1 if half >= 0 else U0
                hem["b"].append(np.c_[U(tb), v, tb])
                hem["t"].append(np.c_[U(tt), v, tt])
        if not hem["b"]:
            return
        bot, top = np.vstack(hem["b"]), np.vstack(hem["t"])
        tmid = 0.5 * (t0 + t1)
        c0 = centre(tmid)
        sym = (half == 0 and fr.vertical)
        slope = 1.2 if fr.vertical else self.limb_hem_slope
        pb = self._hem_fit(bot, c0, sym, max_slope=slope)
        pt = self._hem_fit(top, c0, sym, max_slope=slope)

        side_ex = {}

        def excess(t):
            key = round(float(t), 3)
            if key not in side_ex:
                c, r = centre(key), base(key, ang) + pad
                uu, vv = c[0] + r * ca, c[1] + r * sa
                side_ex[key] = (max(0.0, float(U1(key)) - uu.max()), max(0.0, uu.min() - float(U0(key))),
                                max(0.0, float(V1(key)) - vv.max()), max(0.0, vv.min() - float(V0(key))))
            return side_ex[key]

        def ring_uv(t, a, grow=0.0):
            q = np.array([math.cos(a), math.sin(a)])
            if shell == "drawn":
                c, r = superellipse((float(U0(t)), float(V0(t))), (float(U1(t)), float(V1(t))), q, exponent)
                r += grow
                return c + r * q, c, q, r
            c = centre(t)
            e1_, e0_, f1_, f0_ = excess(t)
            r = (float(base(t, a)) + pad + grow + e1_ * max(q[0], 0) ** 2 + e0_ * max(-q[0], 0) ** 2
                 + f1_ * max(q[1], 0) ** 2 + f0_ * max(-q[1], 0) ** 2)
            return c + r * q, c, q, r

        # plate sector: keep the part of the ring that both drawings cover
        loop = True
        th = np.arange(slices) * 2 * np.pi / slices
        if open_under is not None:
            am = math.acos(-open_under)
            th = np.linspace(-am, am, max(6, int(round(slices * am / math.pi))) + 1)
            loop = False
        elif sector:
            vis_u = np.r_[lo[~olo], hi[~ohi]] if (~olo).any() or (~ohi).any() else np.r_[lo, hi]
            ulo_d = float(np.median(lo[~olo])) if (~olo).sum() > 3 else float(np.median(lo))
            uhi_d = float(np.median(hi[~ohi])) if (~ohi).sum() > 3 else float(np.median(hi))
            vlo_d, vhi_d = float(np.median(sd[1])), float(np.median(sd[2]))
            fine = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            m = self.sector_margin
            keep = []
            for a in fine:
                uv, _, _, _ = ring_uv(tmid, a)
                keep.append(ulo_d - m <= uv[0] <= uhi_d + m and vlo_d - m <= uv[1] <= vhi_d + m)
            keep = np.array(keep)
            if keep.any() and not keep.all():
                # largest circular run of kept angles
                k2 = np.r_[keep, keep]
                best, run, start, bstart = 0, 0, 0, 0
                for i, kk in enumerate(k2):
                    run = run + 1 if kk else 0
                    if run == 1:
                        start = i
                    if run > best and run <= len(keep):
                        best, bstart = run, start
                if 6 <= best < len(keep):
                    a0 = fine[bstart % len(fine)]
                    span = best * (2 * np.pi / len(fine))
                    n = max(6, int(round(slices * span / (2 * np.pi))) + 1)
                    th = a0 + np.linspace(0, span, n)
                    loop = False

        # heights: bottom/top hem planes evaluated around the ring
        tb_a, tt_a = [], []
        for a in th:
            uv, _, _, _ = ring_uv(tmid, a)
            tb_a.append(self._hem_eval(pb, uv, c0, sym))
            tt_a.append(self._hem_eval(pt, uv, c0, sym))
        tb_a = np.maximum(np.array(tb_a), t0 - self.tuck)
        tt_a = np.minimum(np.array(tt_a), t1 + self.tuck)
        if not fr.vertical:
            # along a limb, a piece that the next piece covers stops just under that piece's start
            # (a gauntlet cuff over the vambrace end, a vambrace over the sleeve end): no double layer,
            # and no flare to clear the limb beyond its real end
            for q in OVER.get(pid, ()):
                qs = self._drawn_start(q, half)
                if qs is not None and 0.5 * (t0 + t1) < qs < t1 + self.tuck:     # covers our far end
                    tt_a = np.minimum(tt_a, qs + self.limb_overlap)
        dome_h = 0.0
        if end == "dome":
            # the dome must start where the body inside ends (minus half a radius, since that end is
            # itself rounded) and reach past it by the stand-off - not shrink while the body is full width
            inside = [t for t in tf if len(self.fsection(bodies, fr, t, half, obstacles)) >= 3]
            body_end = max(inside) if inside else t1
            r_end = float(np.mean([ring_uv(min(t1, body_end), a)[3] for a in th]))
            base_t = min(t1, body_end) - 0.5 * r_end
            dome_h = max(t1 - base_t, body_end + pad - base_t, 0.02)
            tt_a = np.full(len(th), base_t)
        if end == "dome_prox":
            r_end = float(np.mean([ring_uv(t0, a)[3] for a in th]))
            dome_h = min(0.8 * r_end, 0.4 * float(np.mean(tt_a - tb_a)))
            tb_a = np.full(len(th), t0 + dome_h)
        if end == "flat":
            tb_a = np.full(len(th), float(np.min(tb_a)))
        tt_a = np.maximum(tt_a, tb_a + 0.03)
        n_r = max(3, int(round(float(np.mean(tt_a - tb_a)) / spacing)) + 1)

        def heights(ts):
            return np.array([tb_a + t * (tt_a - tb_a) for t in ts])

        # clearance safety pass (the outline already clears body + under-pieces; this catches the rest)
        Tc = heights(np.linspace(0, 1, 3 * n_r - 2))
        need_t, need = [], []
        for i in range(len(Tc)):
            for k, a in enumerate(th):
                t = float(Tc[i, k])
                uv, c, q, r = ring_uv(t, a)
                pts = self.fsection(bodies, fr, t, half, obstacles)
                br = hull_radius(pts, c, q) if len(pts) >= 3 else 0.0
                need_t.append(t)
                need.append(br + pad - r if br > 0 else 0.0)
        g = envelope(need_t, need, deg=2)

        T = heights(np.linspace(0, 1, n_r))
        grid = np.zeros((n_r, len(th), 3))
        for i in range(n_r):
            for k, a in enumerate(th):
                t = float(T[i, k])
                uv, c, q, r = ring_uv(t, a, float(g(t)))
                if taper and not loop and i in (0, n_r - 1):
                    uv = c + (uv - c) * (1 - taper)
                grid[i, k] = fr.world(t, uv[0], uv[1])
        C = len(th)
        pts = [grid.reshape(-1, 3)]
        faces = []
        for i in range(n_r - 1):
            for k in range(C if loop else C - 1):
                kn = (k + 1) % C
                faces.append([i * C + k, i * C + kn, (i + 1) * C + kn, (i + 1) * C + k])
        nv = n_r * C
        if end in ("dome", "dome_prox"):
            row, sgn = (n_r - 1, 1.0) if end == "dome" else (0, -1.0)
            last = T[row]
            cen = [ring_uv(float(t), a, float(g(float(t))))[1] for t, a in zip(last, th)]
            prev = list(range(row * C, (row + 1) * C))
            for phi in (math.radians(35), math.radians(65)):
                ring = []
                for k, a in enumerate(th):
                    t = float(last[k])
                    uv, c, q, r = ring_uv(t, a, float(g(t)))
                    uv2 = cen[k] + (uv - cen[k]) * math.cos(phi)
                    ring.append(fr.world(t + sgn * dome_h * math.sin(phi), uv2[0], uv2[1]))
                pts.append(np.array(ring))
                cur = list(range(nv, nv + C))
                nv += C
                faces += [[prev[k], prev[(k + 1) % C], cur[(k + 1) % C], cur[k]] for k in range(C if loop else C - 1)]
                prev = cur
            capc = np.mean(cen, axis=0)
            pts.append(fr.world(float(np.mean(last)) + sgn * dome_h, capc[0], capc[1])[None, :])
            apex = nv
            nv += 1
            faces += [[prev[k], prev[(k + 1) % C], apex] for k in range(C if loop else C - 1)]
        if end == "flat" and loop:
            faces.append(list(range(C))[::-1])
        pts = np.vstack(pts)

        def axis_of(p):
            t, _, _ = fr.local(p[None, :])
            cc = ring_uv(float(t[0]), 0.0)[1]
            return fr.world(float(t[0]), cc[0], cc[1])

        faces = _orient_outward(pts, faces, axis_of)
        pts, faces = solidify_rims(pts, faces, thick)
        tag = "" if not half else (".L" if half > 0 else ".R")
        self._add(f"{pid:02d} {NAMES[pid]}{tag}", pts, faces)

    @staticmethod
    def _drawn_curve(z, v, occ, ov, oocc, centre, deg):
        """A drawn edge vs station from its visible rows; if that side is mostly hidden, the other side
        mirrored about the limb centre (pieces are taken as symmetric across their own axis)."""
        vis = ~occ
        if vis.sum() >= 6 and vis.mean() >= 0.3:
            return Curve(z[vis], v[vis], deg)
        ok = ~oocc if (~oocc).sum() >= 6 else np.ones(len(z), bool)
        c = np.array([float(centre(t)) for t in z[ok]])
        return Curve(z[ok], 2 * c - ov[ok], deg)

    @staticmethod
    def _hem_cols(uv, c0, sym):
        du, dv = uv[:, 0] - c0[0], uv[:, 1] - c0[1]
        return np.c_[np.ones(len(du)), du, dv] if not sym else np.c_[np.ones(len(du)), du, dv, du * du]

    def _hem_fit(self, P, c0, sym, k=2.5, iters=4, max_slope=1.2):
        """Hem as a low-order surface t(u, v): a plane, plus a symmetric u^2 bow for centre pieces."""
        A = self._hem_cols(P[:, :2], c0, sym)
        z = P[:, 2]
        m = np.ones(len(z), bool)
        p = np.linalg.lstsq(A, z, rcond=None)[0]
        for _ in range(iters):
            r = z - A @ p
            s = 1.4826 * np.median(np.abs(r[m])) + 1e-4
            m2 = np.abs(r) < k * s
            if m2.sum() < A.shape[1] + 4 or np.array_equal(m2, m):
                break
            m = m2
            p = np.linalg.lstsq(A[m], z[m], rcond=None)[0]
        p[1:3] = np.clip(p[1:3], -max_slope, max_slope)
        if sym:
            p[3] = np.clip(p[3], -4.0, 4.0)
        return p

    def _hem_eval(self, p, uv, c0, sym):
        return float((self._hem_cols(np.atleast_2d(uv), c0, sym) @ p)[0])

    def _limb(self, bodies, fr, tf, half, cu, cv, obstacles, n_ang=64):
        """The body (plus under-pieces) inside a tube, as smooth functions of the station t.

        centre(t) -> (u, v), a low-order fit of section centroids; radius(t, angle) -> the smallest
        few-harmonic outline that stays outside the section's convex hull (an LP), coefficients
        interpolated in t."""
        ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
        K = self.harmonics
        A = np.c_[np.ones(n_ang), *[f(k * ang) for k in range(1, K + 1) for f in (np.cos, np.sin)]]
        ts, cs, secs = [], [], []
        for t in tf:
            p = self.fsection(bodies, fr, t, half, obstacles)
            if fr.vertical and len(p) and self._fp is None:
                p = p[(np.abs(p[:, 0] - cu) < 0.25) & (np.abs(p[:, 1] - cv) < 0.25)]
            if len(p) < 6:
                continue
            ts.append(t)
            cs.append(p.mean(0))
            secs.append(p)
        if len(ts) < 3:
            c0 = np.array([cu, cv])
            return (lambda t: c0), (lambda t, a: np.full(np.shape(a), 0.05))
        ts, cs = np.array(ts), np.array(cs)
        dg = 2 if np.ptp(ts) > 0.2 else 1
        CU, CV = Curve(ts, cs[:, 0], dg), Curve(ts, cs[:, 1], dg)
        coef, tc = [], []
        for t, p in zip(ts, secs):
            c = np.array([float(CU(t)), float(CV(t))])
            r = np.array([hull_radius(p, c, np.array([math.cos(a), math.sin(a)])) for a in ang])
            ok = r > 0
            if ok.mean() < 0.75:          # a sliver, or the centre is outside it: not a usable station
                continue
            res = linprog(c=A[ok].sum(0), A_ub=-A[ok], b_ub=-r[ok], bounds=[(None, None)] * A.shape[1],
                          method="highs")
            if res.success:
                w = res.x
            else:
                w = np.linalg.lstsq(A[ok], r[ok], rcond=None)[0]
                w[0] += max(0.0, float(np.max(r[ok] - A[ok] @ w)))
            coef.append(w)
            tc.append(t)
        if len(tc) < 2:
            c0 = np.array([cu, cv])
            return (lambda t: c0), (lambda t, a: np.full(np.shape(a), 0.05))
        coef, ts = np.array(coef), np.array(tc)

        def centre(t):
            return np.array([float(CU(t)), float(CV(t))])

        def radius(t, a):
            w = np.array([np.interp(t, ts, coef[:, j]) for j in range(coef.shape[1])])
            a = np.asarray(a, float)
            terms = [np.ones_like(a)] + [f(k * a) for k in range(1, K + 1) for f in (np.cos, np.sin)]
            return sum(wj * tt for wj, tt in zip(w, terms))

        return centre, radius

    @staticmethod
    def _side_curve(z, v, occ, zf, body, deg, out):
        """One side's edge vs station, fitted over the piece's whole length `zf`.

        Where the side is drawn and visible: the outermost of the drawn edge and body + stand-off, so a
        partly hidden drawing can only make the piece bigger, never pull it into the body.
        Where the drawing says nothing (hidden, or beyond the drawn rows): body + stand-off, so the
        piece hugs the limb instead of carrying the last drawn width on past a taper."""
        vis = ~occ
        bz = np.interp(z, zf, np.where(np.isnan(body), np.nanmedian(body), body))
        zs, es = list(z[vis]), list((np.maximum(v[vis] * out, bz[vis] * out)) * out)
        if len(zs):
            silent = np.min(np.abs(zf[:, None] - np.asarray(zs)[None, :]), axis=1) > 0.012
        else:
            silent = np.ones(len(zf), bool)
        silent &= ~np.isnan(body)
        zs += list(zf[silent])
        es += list(body[silent])
        if len(zs) < deg + 3:
            return Curve(z, v, min(deg, 1))
        return Curve(np.asarray(zs), np.asarray(es), deg)

    # ---- hanging cloth panel (tabard): straight-ish edges, low-order hem, gentle wrap, one stand-off
    # hanging cloth: folds per panel and their depth at the hem (zero at the top, where it is held)
    folds = 2
    fold_depth = 0.014

    def panel(self, pid, view, half, cols=9, rows=7, thick=0.004, gap=0.006, under=(10, 11, 13)):
        d, over = self.d, OVER.get(pid, set())
        dirn = -1 if view == "Front" else 1
        sp = spans(d, view, [pid], half, over)
        ss = side_strips(d, "Side_Without_Arms", [pid], over)
        ed = col_edges(d, view, [pid], half, over, trim=0.0)
        if sp is None or ss is None or ed is None:
            return
        z, lo, hi, olo, ohi = sp
        ax0, ax1 = np.abs(lo), np.abs(hi)
        inner = Curve(z, np.minimum(ax0, ax1), 2)
        outer = Curve(z, np.maximum(ax0, ax1), 1)
        ztop = float(z.max()) + (self.tuck if ed[4].mean() > 0.5 else 0.0)
        hem = robust_polyfit(np.abs(ed[0]), ed[1], 2)              # z of the hem against |x|
        # from the side: the panel's leading face and how far its outer edge wraps back
        sz, two = ss[0], ss[5]
        lead = ss[1] if dirn < 0 else ss[4]
        back = ss[2] if dirn < 0 else ss[3]
        ok = two if two.sum() >= 6 else np.ones(len(sz), bool)
        depth = Curve(sz[ok], lead[ok], 1)
        wrap = Curve(sz[ok], np.abs(back[ok] - lead[ok]), 1)

        def x_at(u, zz):
            return half * (inner(zz) + u * (outer(zz) - inner(zz)))

        # hem height per column: solve z = hem(|x(u, z)|) by fixed-point
        us = np.linspace(0, 1, cols)
        zh = np.full(cols, float(z.min()))
        for _ in range(6):
            zh = np.polyval(hem, np.abs(x_at(us, zh)))
        zh = np.minimum(zh, ztop - 0.03)

        P = []
        for t in np.linspace(0, 1, rows):
            row = []
            for j, u in enumerate(us):
                zz = zh[j] + t * (ztop - zh[j])
                x = float(x_at(u, zz))
                y = float(depth(zz)) - dirn * float(wrap(zz)) * u * u
                # folds: vertical ridges pushed outward, growing from nothing at the top to full at the hem
                ridge = 0.5 * (1 - math.cos(2 * math.pi * self.folds * u))
                y += dirn * self.fold_depth * (1 - t) ** 1.5 * ridge
                row.append((x, y, zz))
            P.append(row)
        P = np.array(P)

        # obstacles: the body plus the pieces this panel lies over
        obs = [self.section("body", zz) for zz in np.linspace(zh.min() - 0.02, ztop, 24)]
        for name, (pv, _) in self.parts.items():
            if int(name[:2]) in under:
                obs.append(pv)
        obs = np.vstack(obs)
        need_z, need = [], []
        for p in P.reshape(-1, 3):
            w = obs[(np.abs(obs[:, 0] - p[0]) < 0.03) & (np.abs(obs[:, 2] - p[2]) < 0.03)]
            if len(w):
                lim = (w[:, 1].min() - gap) if dirn < 0 else (w[:, 1].max() + gap)
                need.append((p[1] - lim) * -dirn)
                need_z.append(p[2])
        g = envelope(need_z, need)
        P[..., 1] += dirn * g(P[..., 2])
        pts, faces = shell(P, thickness=thick, outward=(0, dirn, 0))
        self._add(f"{pid:02d} Tabard {view.lower()} " + ("L" if (half > 0) == (view == "Front") else "R"), pts, faces)

    def build(self, only=None):
        """Build every piece from its template, each after the pieces it lies over (UNDER)."""
        for pid in self.ORDER:
            if only and pid not in only:
                continue
            if pid == 12:
                for view in ("Front", "Back"):
                    for half in (-1, 1):
                        self.panel(12, view, half, under=tuple(UNDER[12]))
                continue
            spec = dict(self.TUBES[pid])
            halves = (1, -1) if spec.pop("half", False) else (0,)
            for half in halves:
                self.tube(pid, half, **spec)
        return self.parts
