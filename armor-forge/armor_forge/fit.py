"""ArmorForge v2: build armor pieces from registered part-ID masks and the real mannequin geometry.

Everything that can be measured is measured from the masks: row profiles give the width/depth at each
height, the top and bottom edge at every angle around a piece comes from the outline in the view that
sees it, and the real body decides the clearance. Per-part settings are only artistic choices
(how many segments, how boxy, how thick, how far a plate wraps).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, QhullError

VIEWS = ("Front", "Back", "Side", "Side_Without_Arms")


# ----------------------------------------------------------------------------- drawings
class Profile:
    """Smoothed per-row [lo, hi] spans, evaluated at any height (clamped outside the drawn rows)."""

    def __init__(self, z, limits, smooth=1.0):
        o = np.argsort(z)
        self.z = np.asarray(z, float)[o]
        lim = np.asarray(limits, float)[o]
        if smooth:
            lim = gaussian_filter1d(lim, smooth, axis=0, mode="nearest")
        self.f = PchipInterpolator(self.z, lim, axis=0)

    def at(self, z):
        return self.f(np.clip(z, self.z[0], self.z[-1]))

    @property
    def extent(self):
        return float(self.z[0]), float(self.z[-1])


class Drawing:
    def __init__(self, labels, meta):
        self.lab = labels
        self.ppm = meta["pixels_per_meter"] / 4
        self.ox, self.oy = meta["world_origin_px"][0] / 4, meta["world_origin_px"][1] / 4
        self._c = {}

    def world(self, view, rows, cols):
        u = (cols + 0.5 - self.ox) / self.ppm
        z = (self.oy - rows - 0.5) / self.ppm
        if view == "Back":
            u = -u
        return u, z

    def pixels(self, view, ids, half=0):
        rows, cols = np.nonzero(np.isin(self.lab[view], ids))
        u, z = self.world(view, rows, cols)
        if half and view in ("Front", "Back"):
            k = u * half > 0
            u, z = u[k], z[k]
        return u, z

    def profile(self, view, ids, half=0, smooth=1.25):
        key = (view, tuple(np.atleast_1d(ids)), half, smooth)
        if key in self._c:
            return self._c[key]
        g = self.lab[view]
        sel = np.isin(g, ids)
        cols = np.arange(g.shape[1])
        u = (cols + 0.5 - self.ox) / self.ppm
        if view == "Back":
            u = -u
        if half and view in ("Front", "Back"):
            sel = sel & (u * half > 0)[None, :]
        z, lim = [], []
        for r in np.flatnonzero(sel.any(axis=1)):
            hit = u[sel[r]]
            if len(hit) < 2:
                continue
            z.append((self.oy - r - 0.5) / self.ppm)
            lim.append([hit.min(), hit.max()])
        p = Profile(z, lim, smooth) if len(z) > 3 else None
        self._c[key] = p
        return p

    def zextent(self, ids, half=0):
        zs = []
        for v in ("Front", "Back"):
            _, z = self.pixels(v, ids, half)
            if len(z):
                zs.append(z)
        z = np.concatenate(zs) if zs else np.array([])
        return (float(np.percentile(z, 0.2)), float(np.percentile(z, 99.8))) if len(z) else None

    def edge_z(self, view, ids, x, band=0.006):
        """Lowest and highest drawn row of the part at world x (front/back view)."""
        g = self.lab[view]
        u = x if view != "Back" else -x
        c = int(round(u * self.ppm + self.ox - 0.5))
        w = max(1, int(band * self.ppm))
        c0, c1 = max(c - w, 0), min(c + w + 1, g.shape[1])
        if c1 <= c0:
            return None
        rows = np.flatnonzero(np.isin(g[:, c0:c1], ids).any(axis=1))
        if not len(rows):
            return None
        return (self.oy - rows.max() - 0.5) / self.ppm, (self.oy - rows.min() - 0.5) / self.ppm


# ----------------------------------------------------------------------------- body
class Body:
    def __init__(self, g):
        self.v = np.asarray(g["vertices"], float)
        self.e = np.asarray(g["edges"])
        self.faces = g["faces"]

    def section(self, d, axis=(0, 0, 1), origin=(0, 0, 0)):
        pos = (self.v - origin) @ np.asarray(axis, float)
        p0, p1 = pos[self.e[:, 0]], pos[self.e[:, 1]]
        hit = ((p0 <= d) & (p1 > d)) | ((p1 <= d) & (p0 > d))
        t = (d - p0[hit]) / (p1[hit] - p0[hit])
        a, b = self.v[self.e[hit, 0]], self.v[self.e[hit, 1]]
        return a + (b - a) * t[:, None]


def hull_radius(pts, center, q):
    """Distance from `center` along unit `q` to the convex hull of 2D points (0 if outside/degenerate)."""
    if len(pts) < 3:
        return 0.0
    try:
        eq = ConvexHull(pts).equations
    except QhullError:
        return 0.0
    s = eq[:, :-1] @ center + eq[:, -1]
    den = eq[:, :-1] @ q
    up, dn = den > 1e-6, den < -1e-6
    if not up.any() or np.any((~up & ~dn) & (s > 1e-6)):
        return 0.0
    entry = float(np.max(-s[dn] / den[dn], initial=0.0))
    exit_ = float(np.min(-s[up] / den[up]))
    return exit_ if exit_ >= entry else 0.0


def superellipse(lo, hi, q, n):
    """Centre and radius along unit q of an asymmetric superellipse spanning the box lo..hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    span = np.maximum(np.where(q < 0, c - lo, hi - c), 1e-3)
    return c, float(np.sum(np.abs(q / span) ** n) ** (-1 / n))


# ----------------------------------------------------------------------------- shells
def shell(grid, loop=False, thickness=0.006, cap=False, outward=None):
    """Quad grid (rows x cols x 3) -> (verts, faces) with a rim wall of `thickness` along every open edge."""
    grid = np.asarray(grid, float)
    R, C = grid.shape[:2]
    pts = grid.reshape(-1, 3)
    faces = []
    for i in range(R - 1):
        for j in range(C if loop else C - 1):
            jn = (j + 1) % C
            faces.append([i * C + j, i * C + jn, (i + 1) * C + jn, (i + 1) * C + j])
    if outward is not None:
        n = _vnormals(pts, faces)
        if np.sum(n @ np.asarray(outward, float)) < 0:
            faces = [f[::-1] for f in faces]
    else:
        faces = _orient_away_from_centre(pts, faces)
    if cap:
        faces.append(list(range(C))[::-1])
        faces.append(list(range((R - 1) * C, R * C)))
    if thickness:
        n = _vnormals(pts, faces)
        edge = {}
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1]):
                k = (min(a, b), max(a, b))
                if k in edge:
                    del edge[k]
                else:
                    edge[k] = (a, b)
        if edge and not cap:
            m = len(pts)
            pts = np.vstack([pts, pts - thickness * n])
            faces += [[b, a, a + m, b + m] for a, b in edge.values()]
    return pts, faces


def _orient_away_from_centre(pts, faces):
    """Flip all faces together if, on average, they face toward the shell's own centroid."""
    c = pts.mean(0)
    tot = 0.0
    for f in faces:
        p = pts[f]
        nrm = np.cross(p[1] - p[0], p[2] - p[0]) if len(f) >= 3 else np.zeros(3)
        tot += float(nrm @ (p.mean(0) - c))
    return [f[::-1] for f in faces] if tot < 0 else faces


def _vnormals(pts, faces):
    n = np.zeros_like(pts)
    for f in faces:
        p = pts[f]
        q = np.roll(p, -1, axis=0)
        n[f] += [((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
                 ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
                 ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum()]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


@dataclass
class Spec:
    slices: int = 16
    spacing: float = 0.04        # ring spacing, m
    exponent: float = 2.3        # superellipse boxiness
    pad: float = 0.008           # stand-off from the body, m
    thick: float = 0.006         # rim wall thickness, m
    arc: tuple | None = None     # (theta0, theta1) degrees: plate wraps only this part of the ring
    cap: bool = False            # closed ends (helmet, boots)
    body: str = "body"           # which mannequin piece to clear
    side: bool = False           # build per leg/arm (L and R)
    grow: float = 0.0            # extra size on top of the drawing
    taper: float = 0.0           # shrink the end rings (rounded domes)


SPECS = {
    1: Spec(22, 0.035, 2.4, 0.009, 0.0, None, True, "head"),
    2: Spec(18, 0.03, 2.2, 0.010, 0.006, None, False, "armless"),
    3: Spec(28, 0.045, 2.35, 0.012, 0.007, None, False, "armless"),
    4: Spec(16, 0.04, 2.3, 0.016, 0.007, (-100, 100), False, "body", True, 0.004, 0.10),
    5: Spec(16, 0.035, 2.3, 0.012, 0.005, (-100, 100), False, "body", True, 0.003, 0.05),
    6: Spec(16, 0.05, 2.1, 0.008, 0.005, None, False, "body", True),
    7: Spec(16, 0.045, 2.4, 0.009, 0.006, None, False, "body", True),
    9: Spec(28, 0.035, 2.7, 0.010, 0.005, None, False, "armless"),
    10: Spec(14, 0.035, 2.5, 0.012, 0.0065, (-105, 105), False, "body", True),
    11: Spec(14, 0.035, 2.5, 0.010, 0.0065, (-105, 105), False, "body", True),
    13: Spec(16, 0.04, 2.3, 0.012, 0.0045, None, False, "body", True),
    14: Spec(14, 0.035, 2.3, 0.016, 0.009, (-150, -30), False, "body", True, 0.004),
    15: Spec(18, 0.06, 2.25, 0.008, 0.006, None, False, "body", True),
    16: Spec(18, 0.04, 3.0, 0.008, 0.0, None, True, "feet", True),
}
NAMES = {1: "Helmet", 2: "Gorget", 3: "Cuirass", 4: "Pauldron", 5: "Shoulder lame", 6: "Padded sleeve",
         7: "Vambrace", 8: "Gauntlet", 9: "Belt", 10: "Upper tasset", 11: "Lower tasset", 12: "Tabard",
         13: "Thigh plate", 14: "Knee guard", 15: "Greave", 16: "Armored boot"}
ARM_IDS = {4, 5, 6, 7}


class Fitter:
    def __init__(self, drawing, bodies):
        self.d = drawing
        self.b = {k: Body(v) for k, v in bodies.items()}
        self.parts = {}

    # ---- helpers
    def _add(self, name, pts, faces):
        self.parts[name] = (np.asarray(pts), [list(map(int, f)) for f in faces])

    def _bounds(self, ids, half, z, sag):
        lo, hi = [], []
        for v in ("Front", "Back"):
            p = self.d.profile(v, ids, half)
            if p is not None:
                a, b = p.at(z)
                lo.append(a)
                hi.append(b)
        sp = self.d.profile(sag, ids)
        return (min(lo), max(hi)), sp.at(z)

    # ---- vertical-axis parts: helmet, gorget, cuirass, belt, tassets, knee, greave, cuisse, boot
    def radial(self, pid, half=0):
        s = SPECS[pid]
        sag = "Side_Without_Arms"
        ids = [pid, 11] if pid == 11 else [pid]
        if pid == 11:
            ids = [10, 11]
        if self.d.profile(sag, ids) is None:
            return
        ze = self.d.zextent(ids, half)
        if ze is None:
            return
        z0, z1 = ze
        mid = 0.5 * (z0 + z1)
        (xl, xh), (yf, yb) = self._bounds(ids, half, mid, sag)
        cx, ax = (xl + xh) / 2, (xh - xl) / 2
        n_r = max(4, int(round((z1 - z0) / s.spacing)) + 1)
        ts = np.linspace(0, 1, n_r)
        th = np.arange(s.slices) * 2 * np.pi / s.slices
        if s.arc:
            th = np.radians(np.linspace(s.arc[0], s.arc[1], max(6, int(s.slices))))
            if half < 0:
                th = np.pi - th[::-1]          # mirror the wrap for the right side
        # top/bottom edge of the piece at every angle, read from the outline that sees it
        bot, top = np.full(len(th), z0), np.full(len(th), z1)
        for k, a in enumerate(th if not s.cap else []):
            view = "Front" if math.sin(a) <= 0 else "Back"
            x = cx + 0.85 * ax * math.cos(a)
            e = self.d.edge_z(view, ids, x) or self.d.edge_z("Front" if view == "Back" else "Back", ids, x)
            bot[k], top[k] = e if e else (z0, z1)
        if not s.cap:
            bot = np.clip(gaussian_filter1d(bot, 2.5, mode="nearest"), z0 - 0.01, z1)
            top = np.clip(gaussian_filter1d(top, 2.5, mode="nearest"), z0, z1 + 0.01)
        body = self.b.get(s.body)
        grid = []
        for t in ts:
            row = []
            for k, a in enumerate(th):
                z = bot[k] + t * (top[k] - bot[k])
                (xl, xh), (yf, yb) = self._bounds(ids, half, z, sag)
                q = np.array([math.cos(a), math.sin(a)])
                c, r = superellipse((xl - s.grow, yf - s.grow), (xh + s.grow, yb + s.grow), q, s.exponent)
                if body is not None:
                    pts = body.section(z)
                    if half:
                        pts = pts[pts[:, 0] * half > 0]
                    pts = pts[(pts[:, 0] > xl - 0.06) & (pts[:, 0] < xh + 0.06)]
                    br = hull_radius(pts[:, :2], c, q)
                    if br > 0:
                        r = max(r, br + s.pad)
                xy = c + r * q
                row.append((xy[0], xy[1], z))
            grid.append(row)
        grid = np.array(grid)
        if s.taper:
            for i, f in ((0, 1 - s.taper), (-1, 1 - s.taper)):
                c = grid[i].mean(0)
                grid[i] = c + (grid[i] - c) * f
        pts, faces = shell(grid, loop=not s.arc, thickness=s.thick, cap=s.cap,
                           outward=None)
        tag = "" if not half else (".L" if half > 0 else ".R")
        self._add(f"{pid:02d} {NAMES[pid]}{tag}", pts, faces)

    # ---- arm parts: rings along the limb axis
    def limb_frame(self, pid, half):
        u, z = self.d.pixels("Front", [pid], half)
        ub, zb = self.d.pixels("Back", [pid], half)
        P = np.c_[np.r_[u, -ub * 0 + (-ub)], np.r_[z, zb]] if False else np.c_[np.r_[u, -ub], np.r_[z, zb]]
        P = P[P[:, 0] * half > 0] if False else P
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        a = vt[0]
        if a[0] * half < 0:
            a = -a
        s = (P - c) @ a
        lo, hi = np.percentile(s, [0.3, 99.7])
        return c, a, lo, hi

    def arm(self, pid, half):
        s = SPECS[pid]
        # use the part's own pixels to find the limb direction (x, z plane)
        ua, za = self.d.pixels("Front", [pid], half)
        ub, zb = self.d.pixels("Back", [pid], half)
        P = np.c_[np.r_[ua, ub * -1 * -1], np.r_[za, zb]]
        # the Back view reports world-x already (see Drawing.world)
        P = np.c_[np.r_[ua, ub], np.r_[za, zb]]
        if len(P) < 50:
            return
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        A = vt[0]
        if A[0] * half < 0:
            A = -A
        A = A / np.linalg.norm(A)
        U = np.array([-A[1], A[0]])        # in-plane perpendicular
        if U[1] < 0:
            U = -U                          # points up
        t_all = (P - c) @ A
        t0, t1 = np.percentile(t_all, [0.3, 99.7])
        sd_u, sd_z = self.d.pixels("Side", [pid])
        n_r = max(4, int(round((t1 - t0) / s.spacing)) + 1)
        body = self.b[s.body]
        th = np.arange(s.slices) * 2 * np.pi / s.slices
        if s.arc:
            th = np.radians(np.linspace(s.arc[0], s.arc[1], s.slices))
        ts = np.linspace(t0, t1, n_r)
        meas = []
        for t in ts:
            m = np.abs(t_all - t) < 0.012
            if m.sum() < 5:
                m = np.abs(t_all - t) < 0.03
            w = (P[m] - c) @ U if m.any() else np.array([-0.05, 0.05])
            lo, hi = np.percentile(w, [1.5, 98.5])
            ctr = c + A * t
            zh = 0.012 * abs(A[1]) + max((hi - lo) / 2, 0.03) * abs(U[1])
            mm = np.abs(sd_z - ctr[1]) <= zh
            yl, yh = np.percentile(sd_u[mm], [1.5, 98.5]) if mm.sum() > 5 else (np.nan, np.nan)
            meas.append((lo, hi, yl, yh))
        meas = np.array(meas)
        for j in range(4):
            col = meas[:, j]
            ok = ~np.isnan(col)
            meas[:, j] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok]) if ok.any() else (-0.08 if j == 2 else 0.08)
        meas = gaussian_filter1d(meas, 1.5, axis=0, mode="nearest")
        grid = []
        for t, (lo, hi, yl, yh) in zip(ts, meas):
            ctr = c + A * t
            ring = []
            # body section in the plane perpendicular to the limb axis through this ring
            axis3 = np.array([A[0], 0, A[1]])
            org3 = np.array([ctr[0], 0, ctr[1]])
            pts = body.section(0.0, axis3, org3)
            for a in th:
                q = np.array([math.cos(a), math.sin(a)])          # (along U, along y)
                cc, r = superellipse((lo - s.grow, yl - s.grow), (hi + s.grow, yh + s.grow), q, s.exponent)
                if len(pts):
                    rel = pts - org3
                    pu = rel[:, 0] * U[0] + rel[:, 2] * U[1]
                    keep = (pts[:, 0] * half > 0.14) & (np.abs(pu) < 0.16)
                    if keep.sum() >= 3:
                        br = hull_radius(np.c_[pu[keep], rel[keep, 1]], cc, q)
                        if br > 0:
                            r = max(r, br + s.pad)
                u_, y_ = cc + r * q
                p2 = ctr + U * u_
                ring.append((p2[0], y_, p2[1]))
            grid.append(ring)
        grid = np.array(grid)
        if s.taper:
            for i in (0, -1):
                cc = grid[i].mean(0)
                grid[i] = cc + (grid[i] - cc) * (1 - s.taper)
        # orient the shell outward, away from the limb axis line
        pts_, faces = shell(grid, loop=not s.arc, thickness=s.thick, outward=None)
        self._add(f"{pid:02d} {NAMES[pid]}" + (".L" if half > 0 else ".R"), pts_, faces)
        self.frames = getattr(self, "frames", {})
        self.frames[(pid, half)] = (c, A, t0, t1)

    # ---- gauntlet: clip the mannequin's own hand and give it clearance
    def gauntlet(self, half):
        fr = self.frames.get((7, half))
        if fr is None:
            return
        c, A, t0, t1 = fr
        body = self.b["body"]
        d = (body.v[:, 0] - c[0]) * A[0] + (body.v[:, 2] - c[1]) * A[1]
        limit = t1
        pts, faces, look = [], [], {}
        for f in body.faces:
            f = list(f)
            if max(d[f]) <= limit or min(body.v[f, 0] * half) < 0.5:
                continue
            gate = []
            for a, b in zip(f, f[1:] + f[:1]):
                if d[a] >= limit:
                    gate.append(body.v[a])
                if (d[a] - limit) * (d[b] - limit) < 0:
                    fr_ = (limit - d[a]) / (d[b] - d[a])
                    gate.append(body.v[a] + fr_ * (body.v[b] - body.v[a]))
            if len(gate) < 3:
                continue
            ix = []
            for v in gate:
                k = tuple(v.round(7))
                if k not in look:
                    look[k] = len(pts)
                    pts.append(v)
                ix.append(look[k])
            faces.append(ix if half > 0 else ix[::-1])
        if not pts:
            return
        pts = np.array(pts)
        mid = pts.mean(0)
        pts = mid + (pts - mid) * np.array([1.03, 1.10, 1.04])
        self._add(f"08 Gauntlet" + (".L" if half > 0 else ".R"), pts, faces)

    # ---- hanging hip plates: rows by height, columns from the front edge round to the back edge
    def tasset(self, pid, half):
        ids = [10, 11] if pid == 11 else [10]
        sag = self.d.profile("Side_Without_Arms", ids, 0, 1.0)
        fr, bk = self.d.profile("Front", ids, half), self.d.profile("Back", ids, half)
        ze = self.d.zextent(ids, half)
        if sag is None or fr is None or bk is None or ze is None:
            return
        z0, z1 = ze
        if pid == 10:
            z1 = min(z1, 1.192)
        body = self.b["body"]
        rows = max(5, int(round((z1 - z0) / 0.03)) + 1)
        cols = 14
        grid = []
        for z in np.linspace(z0 + 0.002, z1, rows):
            ya, yb = sag.at(z)
            f_lo, f_hi = np.abs(fr.at(z))
            b_lo, b_hi = np.abs(bk.at(z))
            in_f, in_b = min(f_lo, f_hi), min(b_lo, b_hi)
            outer = max(max(f_lo, f_hi), max(b_lo, b_hi)) + (0.008 if pid == 10 else 0.0)
            sect = body.section(z)
            sect = sect[(sect[:, 0] * half > 0) & (np.abs(sect[:, 0]) < 0.36)]
            row = []
            for u in np.linspace(0, 1, cols):
                y = ya + u * (yb - ya)
                edge = in_f * (1 - u) + in_b * u
                xr = edge + (outer - edge) * (math.sin(math.pi * u) ** 0.67)
                near = sect[np.abs(sect[:, 1] - y) < 0.028]
                if len(near):
                    xr = max(xr, float(np.abs(near[:, 0]).max()) + 0.011)
                low = getattr(self, "_lower", {}).get(half)
                if pid == 10 and low is not None:
                    dist, k = low[0].query([y, z])
                    if dist < 0.03:
                        xr = max(xr, low[1][k] + 0.010)
                row.append((half * xr, y, z))
            grid.append(row)
        if pid == 11:
            from scipy.spatial import cKDTree
            g = np.array(grid).reshape(-1, 3)
            self._lower = getattr(self, "_lower", {})
            self._lower[half] = (cKDTree(g[:, 1:]), np.abs(g[:, 0]))
        pts, faces = shell(grid, thickness=0.0065, outward=(half, 0, 0))
        self._add(f"{pid:02d} {NAMES[pid]}" + (".L" if half > 0 else ".R"), pts, faces)

    # ---- knee cop: a panel across the front of the knee, curled back at the flanks
    def knee(self, half):
        fr = self.d.profile("Front", [14], half)
        sag = self.d.profile("Side_Without_Arms", [14], 0, 1.1)
        if fr is None or sag is None:
            return
        z0, z1 = fr.extent
        body = self.b["body"]
        grid = []
        for z in np.linspace(z0 + 0.005, z1 - 0.004, 15):
            xa, xb = fr.at(z)
            ymin, ymax = sag.at(z)
            cup = float(np.clip(ymax - ymin, 0.055, 0.115))
            sect = body.section(z)
            sect = sect[sect[:, 0] * half > 0]
            row = []
            for u in np.linspace(0, 1, 11):
                x = xa + (xb - xa) * u
                flank = abs(2 * u - 1)
                y = ymin + cup * flank ** 1.8 + 0.005 * math.sin(math.pi * u)
                near = sect[np.abs(sect[:, 0] - x) < 0.03]
                if len(near):
                    y = min(y, float(near[:, 1].min()) - 0.012)
                row.append((x, y, z))
            grid.append(row)
        pts, faces = shell(grid, thickness=0.009, outward=(0, -1, 0))
        self._add(f"14 {NAMES[14]}" + (".L" if half > 0 else ".R"), pts, faces)

    # ---- tabard: four cloth panels draped over the body from the front/back outlines
    def tabard(self):
        body = self.b["body"]
        sag = self.d.profile("Side_Without_Arms", [12], 0, 1.6)
        for view, dirn in (("Front", -1), ("Back", 1)):
            for half in (-1, 1):
                pr = self.d.profile(view, [12], half)
                if pr is None or sag is None:
                    continue
                z0, z1 = pr.extent
                grid = []
                for z in np.linspace(z0 + 0.002, z1, 17):
                    lo, hi = pr.at(z)
                    sect = body.section(z)
                    y_out = float(sag.at(z)[0 if dirn < 0 else 1])
                    row = []
                    for u in np.linspace(0, 1, 11):
                        x = lo + (hi - lo) * u
                        w = abs(x) / max(abs(lo), abs(hi), 1e-3)         # 0 at the split, 1 at the outer edge
                        near = sect[np.abs(sect[:, 0] - x) < 0.04]
                        if len(near):
                            y_in = (near[:, 1].min() - 0.012) if dirn < 0 else (near[:, 1].max() + 0.012)
                        else:
                            y_in = dirn * 0.12
                        y = y_out + (y_in - y_out) * w ** 1.4
                        y += dirn * (0.005 * math.sin(3 * math.pi * u) + 0.0028 * math.sin(8 * math.pi * u + 2.0 * z))
                        row.append((x, y, z))
                    grid.append(row)
                pts, faces = shell(grid, thickness=0.0032, outward=(0, dirn, 0))
                self._add(f"12 Tabard {view.lower()} " + ("L" if (half > 0) == (view == "Front") else "R"), pts, faces)

    def build(self):
        for pid in (1, 2, 3, 9):
            self.radial(pid)
        for pid in (13, 15, 16):
            for half in (1, -1):
                self.radial(pid, half)
        for half in (1, -1):
            self.knee(half)
            self.tasset(11, half)
            self.tasset(10, half)
        for pid in (6, 7, 5, 4):
            for half in (1, -1):
                self.arm(pid, half)
        for half in (1, -1):
            self.gauntlet(half)
        self.tabard()
        return self.parts
