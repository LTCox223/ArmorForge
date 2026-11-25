"""Molded pieces: a clean quad cage shaped to the drawn outline, with features on real edge loops.

Step 1 (shellfit) reuses the body's faces, so pieces inherit its uneven grid and can only puff out.
Here a piece starts from its own cage (rings around a centre plus an all-quad cap, no poles) and
every vertex is placed on a smooth hull built from the drawing:

  * width at each height from the front/back view, front and back depth from the side view,
  * cross-sections are superellipses through those extents (exponent sets round vs boxy),
  * the open hem follows the bottom of the drawn piece.

Feature-aligned loops: before the in-between rings are placed, each meridian is scanned for the
edges of detail colours (visor slit, trim bands). Rings are put exactly on those edges, so a band
or slit becomes a strip of whole faces, then extruded (trim) or inset and pushed in (slit). The hem
always gets a trim band.

Coordinates as in shellfit: metres, Z up, character faces -Y, +X is character left.
"""
from __future__ import annotations

import numpy as np

from .shellfit import BASE, mirror  # noqa: F401  (BASE re-exported for callers)

DETAIL_SLIT = 20
DETAIL_TRIM = 21


# ----------------------------------------------------------------------------- outline profiles
def _smooth(a, k=2):
    if k <= 0 or len(a) < 3:
        return a
    w = np.ones(2 * k + 1) / (2 * k + 1)
    pad = np.pad(a, k, mode="edge")
    return np.convolve(pad, w, mode="valid")


class Outline:
    """Per-height extents of one piece: half-width a(z) (front/back) and front/back depth (side)."""

    def __init__(self, views, pid, ids=None, smooth=2):
        ids = list(ids or [pid])
        self.pid = pid
        f, s = views["front"], views["side"]
        rows = np.arange(f.labels.shape[0])
        z = (f.r0 / f.f - rows - 0.5) * f.f / f.s
        width = np.full(len(rows), np.nan)
        for name in ("front", "back"):
            v = views[name]
            lab = np.isin(v.labels, ids)
            cols = np.arange(lab.shape[1])
            u = np.abs(((cols + 0.5) * v.f - v.c0) / v.s)
            for r in np.flatnonzero(lab.any(1)):
                width[r] = np.nanmax([width[r], u[lab[r]].max()])
        lab = np.isin(s.labels, ids)
        cols = np.arange(lab.shape[1])
        y = ((cols + 0.5) * s.f - s.c0) / s.s
        rows_s = np.arange(lab.shape[0])
        zs = (s.r0 / s.f - rows_s - 0.5) * s.f / s.s
        yf = np.full(len(rows_s), np.nan)
        yb = np.full(len(rows_s), np.nan)
        for r in np.flatnonzero(lab.any(1)):
            yf[r], yb[r] = y[lab[r]].min(), y[lab[r]].max()
        okw, oks = ~np.isnan(width), ~np.isnan(yf)
        if okw.sum() < 3 or oks.sum() < 3:
            raise ValueError(f"piece {pid} is not drawn in both the front and side views")
        self.zw, self.w = z[okw][::-1], _smooth(width[okw][::-1], smooth)
        self.zs = zs[oks][::-1]
        self.yf, self.yb = _smooth(yf[oks][::-1], smooth), _smooth(yb[oks][::-1], smooth)
        self.z_lo = max(self.zw.min(), self.zs.min())
        self.z_hi = min(self.zw.max(), self.zs.max())

    def extents(self, z):
        """-> half-width, y front (min), y back (max); zero-size outside the drawn range."""
        z = np.asarray(z, float)
        a = np.interp(z, self.zw, self.w)
        yf = np.interp(z, self.zs, self.yf)
        yb = np.interp(z, self.zs, self.yb)
        # above the top: nothing. Below the bottom of either view: walls continue straight down
        # (np.interp holds the end values); the hem cut decides where the piece stops.
        a = np.where(z > self.z_hi, 0.0, a)
        return a, yf, yb


def inside(p, outline, n=2.5):
    """True where points lie inside the superellipse hull of the outline."""
    a, yf, yb = outline.extents(p[:, 2])
    ym = 0.5 * (yf + yb)
    b = np.where(p[:, 1] < ym, ym - yf, yb - ym)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.abs(p[:, 0] / a) ** n + np.abs((p[:, 1] - ym) / b) ** n
    return (a > 0) & (b > 0) & (q <= 1.0)


def hull_exit(center, dirs, outline, n=2.5, t_max=0.5, iters=40):
    """Distance along each ray from center to where it leaves the hull (bisection; star-shaped)."""
    lo = np.zeros(len(dirs))
    hi = np.full(len(dirs), t_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ins = inside(center + dirs * mid[:, None], outline, n)
        lo = np.where(ins, mid, lo)
        hi = np.where(ins, hi, mid)
    return lo


def direction(theta, elev):
    """theta: azimuth from the front (-Y) toward +X; elev: angle above the horizontal."""
    ce = np.cos(elev)
    return np.stack([ce * np.sin(theta), -ce * np.cos(theta), np.sin(elev)], -1)


# ----------------------------------------------------------------------------- hem
def bottom_edge(views, ids, x=None, y=None, view="front"):
    """Lowest drawn z of the piece at column x (front/back) or y (side). NaN if not drawn there."""
    v = views[view]
    u = x if v.axis == 0 else y
    col = np.floor((v.c0 + v.s * v.sign * np.asarray(u)) / v.f).astype(int)
    lab = np.isin(v.labels, ids)
    out = np.full(len(col), np.nan)
    for i, c in enumerate(col):
        if 0 <= c < lab.shape[1]:
            r = np.flatnonzero(lab[:, c])
            if len(r):
                out[i] = (v.r0 - (r.max() + 0.5) * v.f) / v.s
    return out


# ----------------------------------------------------------------------------- detail scanning
def detail_at(points, normals, details, views, symmetric=True):
    """Detail id under each point, read in the view its normal faces most (front/back/side)."""
    out = np.zeros(len(points), int)
    pick = np.argmax(np.stack([-normals[:, 1], normals[:, 1], np.abs(normals[:, 0])], 1), 1)
    for k, name in enumerate(("front", "back", "side")):
        m = pick == k
        if not m.any() or name not in details:
            continue
        v = views[name]
        p = points[m].copy()
        if name == "side":
            p[:, 0] = np.abs(p[:, 0])
        col, row = v.project(p)
        hits = _sample(details[name], col, row)
        if symmetric and v.axis == 0:
            col2, row2 = v.project(mirror(p))
            hits = np.maximum(hits, _sample(details[name], col2, row2))
        out[m] = hits
    return out


def _sample(grid, col, row):
    c = np.floor(col).astype(int)
    r = np.floor(row).astype(int)
    ok = (c >= 0) & (c < grid.shape[1]) & (r >= 0) & (r < grid.shape[0])
    out = np.zeros(len(c), int)
    out[ok] = grid[r[ok], c[ok]]
    return out


def track_transitions(per_meridian, tol=0.06, smooth=4):
    """Link detail-edge positions (s in 0..1 along each meridian) into rings around the piece.

    per_meridian: list (one per meridian, in order around) of sorted s values.
    Returns (n_tracks, K) array of s per meridian; gaps are filled by interpolating around the loop.
    """
    K = len(per_meridian)
    tracks = []                      # each: dict meridian -> s
    for k, ss in enumerate(per_meridian):
        used = set()
        for s in ss:
            best, bd = None, tol
            for ti, tr in enumerate(tracks):
                if ti in used or (k - 1) not in tr:
                    continue
                d = abs(tr[k - 1] - s)
                if d < bd:
                    best, bd = ti, d
            if best is None:
                tracks.append({k: s})
                used.add(len(tracks) - 1)
            else:
                tracks[best][k] = s
                used.add(best)
    # drop tiny fragments (noise). A track seen on most meridians is a full ring (gaps are
    # interpolated around the loop); a shorter one is a partial ring: NaN where it was not seen, and
    # ring_params relaxes it evenly between its neighbours there.
    tracks = [t for t in tracks if len(t) >= 2]
    out = []
    for t in tracks:
        ks = np.array(sorted(t))
        vals = np.array([t[k] for k in ks])
        if len(ks) >= 0.75 * K:
            xs = np.concatenate([ks - K, ks, ks + K])
            row = np.interp(np.arange(K), xs, np.tile(vals, 3))
            for _ in range(smooth):          # views disagree a little: even the ring out
                row = 0.5 * row + 0.25 * (np.roll(row, 1) + np.roll(row, -1))
            out.append(row)
        else:
            row = np.full(K, np.nan)
            row[ks] = vals
            out.append(row)
    if not out:
        return np.zeros((0, K))
    out = np.array(out)
    order = np.argsort(np.nanmean(out, 1))
    return out[order]


def ring_params(feature_s, spacing_s, min_gap=0.015):
    """All ring positions along every meridian: 0 (hem), the feature rings, 1 (cap edge), plus evenly
    spaced filler rings between them. Same count on every meridian. -> (R, K)."""
    K = feature_s.shape[1]
    rows = np.array([np.zeros(K)] + [r for r in feature_s] + [np.ones(K)])
    # partial rings: where a ring was not seen, space it evenly between the known rings around it
    for k in range(K):
        col = rows[:, k]
        known = ~np.isnan(col)
        i = np.arange(len(col))
        col[~known] = np.interp(i[~known], i[known], col[known])
    rows = np.maximum.accumulate(np.clip(rows, 0, 1), axis=0)
    out = [rows[0]]
    for a, b in zip(rows[:-1], rows[1:]):
        gap = (b - a).max()
        if gap < min_gap:
            # merge near-duplicate feature rings
            continue
        n = max(1, int(np.ceil(gap / spacing_s)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    return np.array(out)


def fit_bands(MP, MD, theta, details, views, min_cover=0.3, split=0.025):
    """Trim bands as smooth curves around the piece. -> list of (z_lo[K], z_hi[K])."""
    K = len(theta)
    samples = []                         # (k, z_lo, z_hi, weight)
    for name in ("front", "back", "side"):
        if name not in details:
            continue
        v = views[name]
        for k in range(K):
            d = MD[k, MD.shape[1] // 2]
            w = {"front": -d[1], "back": d[1], "side": abs(d[0])}[name]
            if w < 0.35:
                continue
            p = MP[k].copy()
            if name == "side":
                p[:, 0] = np.abs(p[:, 0])
            col, row = v.project(p)
            lab = _sample(details[name], col, row)
            if v.axis == 0:
                c2, r2 = v.project(mirror(p))
                lab = np.maximum(lab, _sample(details[name], c2, r2))
            on = lab == DETAIL_TRIM
            edges = np.flatnonzero(np.diff(on.astype(int)))
            starts = [i + 1 for i in edges if not on[i]] + ([0] if on[0] else [])
            for st in sorted(starts):
                en = st
                while en + 1 < len(on) and on[en + 1]:
                    en += 1
                samples.append((k, p[st, 2], p[en, 2], w))
    if not samples:
        return []
    samples = np.array(samples)
    order = np.argsort(0.5 * (samples[:, 1] + samples[:, 2]))
    samples = samples[order]
    mids = 0.5 * (samples[:, 1] + samples[:, 2])
    cuts = np.flatnonzero(np.diff(mids) > split) + 1
    bands = []
    A = np.stack([np.ones(K), np.cos(theta), np.cos(2 * theta)], 1)
    for grp in np.split(samples, cuts):
        if len(np.unique(grp[:, 0])) < min_cover * K:
            continue
        k = grp[:, 0].astype(int)
        w = np.sqrt(grp[:, 3])
        out = []
        for col in (1, 2):
            coef, *_ = np.linalg.lstsq(A[k] * w[:, None], grp[:, col] * w, rcond=None)
            out.append(A @ coef)
        lo, hi = np.minimum(out[0], out[1]), np.maximum(out[0], out[1])
        bands.append((lo, hi))
    return bands


# ----------------------------------------------------------------------------- cage
def dome_cage(outline, views, ids, details=None, n_exp=2.5, K=32, ring_spacing=0.03,
              cap_elev=np.radians(62), center_frac=0.55, min_dist_fn=None, hem_band_h=0.014):
    """Build the molded dome (helmet-type piece).

    Returns dict(verts, faces, face_tags) where face_tags marks 'slit', 'trim', 'hem' faces.
    min_dist_fn(center, dirs) -> minimum distance along each ray (body + gap), optional.
    """
    zc = outline.z_lo + center_frac * (outline.z_hi - outline.z_lo)
    _, yf, yb = outline.extents(np.array([zc]))
    center = np.array([0.0, 0.5 * (yf[0] + yb[0]), zc])
    # azimuths symmetric about the front so the build is mirror-symmetric
    theta = (np.arange(K) + 0.5) / K * 2 * np.pi - np.pi

    def surf(th, el):
        d = direction(th, el)
        t = hull_exit(center, d.reshape(-1, 3), outline, n_exp).reshape(d.shape[:-1])
        if min_dist_fn is not None:
            t = np.maximum(t, min_dist_fn(center, d.reshape(-1, 3)).reshape(t.shape))
        return center + d * t[..., None], d

    # hem elevation per meridian: where the surface reaches the drawn bottom edge there
    els = np.radians(np.linspace(-80, 60, 141))
    P, D = surf(theta[:, None] + 0 * els[None], els[None] + 0 * theta[:, None])
    hem_el = np.zeros(K)
    for k in range(K):
        p = P[k]
        front = p[:, 1] < center[1]
        zb = np.where(front, bottom_edge(views, ids, x=p[:, 0], view="front"),
                      bottom_edge(views, ids, x=p[:, 0], view="back"))
        zs = bottom_edge(views, ids, y=p[:, 1], view="side")
        zh = np.nanmax(np.stack([zb, zs]), 0)
        ok = ~np.isnan(zh) & (p[:, 2] >= zh)
        hem_el[k] = els[np.argmax(ok)] if ok.any() else els[0]
    # the hem as a smooth symmetric curve: z = a + b cos(theta) + c cos(2 theta) + d cos(3 theta),
    # fitted to the drawn bottom edge (robust to a few odd meridians), then back to elevations
    z_raw = np.array([np.interp(hem_el[k], els, P[k, :, 2]) for k in range(K)])
    A = np.stack([np.ones(K), np.cos(theta), np.cos(2 * theta), np.cos(3 * theta)], 1)
    w = np.ones(K)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A * w[:, None], z_raw * w, rcond=None)
        res = z_raw - A @ coef
        sc = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / np.maximum(1.0, np.abs(res) / (2.5 * sc))
    z_fit = A @ coef
    for k in range(K):
        zz = np.maximum.accumulate(P[k, :, 2])
        hem_el[k] = np.interp(z_fit[k], zz, els)

    # sample every meridian densely: s = 0 (hem) .. 1 (cap edge)
    ss = np.linspace(0, 1, 200)
    EL_f = hem_el[:, None] + ss[None, :] * (cap_elev - hem_el[:, None])
    MP, MD = surf(np.broadcast_to(theta[:, None], EL_f.shape), EL_f)      # (K, 200, 3)

    def s_at_z(k, z):
        zz = np.maximum.accumulate(MP[k, :, 2])
        return float(np.interp(z, zz, ss))

    # trim bands: edge heights read in every view that faces a meridian, then fitted as a smooth
    # symmetric curve around the piece (z = a + b cos(theta) + c cos(2 theta)), so the band is one clean
    # pair of rings whatever view each part of it came from
    bands = fit_bands(MP, MD, theta, details, views) if details else []
    z_hem = MP[:, 0, 2]
    rows = [np.array([s_at_z(k, z_hem[k] + hem_band_h) for k in range(K)])]
    band_rows = []
    for lo, hi in bands:
        if np.mean(lo - z_hem) < hem_band_h + 0.01:
            continue                                  # the hem band covers it
        r_lo = np.array([s_at_z(k, lo[k]) for k in range(K)])
        r_hi = np.array([s_at_z(k, hi[k]) for k in range(K)])
        rows += [r_lo, r_hi]
        band_rows.append((r_lo, r_hi))
    # slit edges (local, front view only): tracked across the meridians that see them as partial rings
    if details and "front" in details:
        v = views["front"]
        per = []
        for k in range(K):
            if MD[k, MD.shape[1] // 2][1] > -0.35:
                per.append([])
                continue
            col, row = v.project(MP[k])
            c2, r2 = v.project(mirror(MP[k]))
            on = np.maximum(_sample(details["front"], col, row), _sample(details["front"], c2, r2)) == DETAIL_SLIT
            ch = np.flatnonzero(np.diff(on.astype(int)))
            per.append([0.5 * (ss[i] + ss[i + 1]) for i in ch])
        for tr in track_transitions(per, tol=0.05, smooth=0):
            if np.isnan(tr).all():
                continue
            if np.isnan(tr).any():
                rows.append(tr)
    rows = sorted(rows, key=lambda r: np.nanmean(r))
    feature_s = np.array(rows)
    length = np.linalg.norm(np.diff(MP[K // 2], axis=0), axis=1).sum()
    S = ring_params(feature_s, ring_spacing / max(length, 1e-6))
    R = S.shape[0]
    EL = hem_el[None, :] + S * (cap_elev - hem_el[None, :])
    ring_pts, ring_dirs = surf(np.broadcast_to(theta, EL.shape), EL)
    # snap the hem ring onto the drawn bottom edge (vertical slide)
    verts = [ring_pts.reshape(-1, 3)]
    idx = np.arange(R * K).reshape(R, K)
    faces, tags = [], []
    for r in range(R - 1):
        for k in range(K):
            k2 = (k + 1) % K
            faces.append([idx[r, k], idx[r, k2], idx[r + 1, k2], idx[r + 1, k]])
            tags.append("hem" if r == 0 else "")      # ring 0..1 is the hem band
    # cap: an m x m quad grid whose border matches the top ring (needs K = 4m)
    m = K // 4
    g = np.linspace(-1, 1, m + 1)
    U, Vg = np.meshgrid(g, g, indexing="ij")
    rho = np.maximum(np.abs(U), np.abs(Vg))
    # azimuth of each grid point: position along the square's border, scaled out to it
    ang = np.arctan2(U, -Vg)                       # 0 at the front (-Y side of the square)
    el_cap = cap_elev + (1 - rho) * (np.pi / 2 - cap_elev)
    cap_pts, _ = surf(ang, el_cap)
    base = R * K
    cap_idx = -np.ones((m + 1, m + 1), int)
    # border points of the square, matched to the top ring: sort both by azimuth, best rotation
    border = [(i, j) for i in range(m + 1) for j in range(m + 1) if rho[i, j] > 1 - 1e-9]
    b_ang = np.array([ang[i, j] for i, j in border])
    border = [border[q] for q in np.argsort(b_ang)]
    b_ang = np.sort(b_ang)
    cost = [np.abs(np.angle(np.exp(1j * (b_ang - np.roll(theta, -sh))))).sum() for sh in range(K)]
    sh = int(np.argmin(cost))
    for q, (i, j) in enumerate(border):
        cap_idx[i, j] = idx[R - 1, (q + sh) % K]
    inner = []
    nxt = base
    for i in range(1, m):
        for j in range(1, m):
            cap_idx[i, j] = nxt
            inner.append(cap_pts[i, j])
            nxt += 1
    allv = np.vstack([verts[0]] + ([np.array(inner)] if inner else []))
    for i in range(m):
        for j in range(m):
            q = [cap_idx[i, j], cap_idx[i + 1, j], cap_idx[i + 1, j + 1], cap_idx[i, j + 1]]
            faces.append(q)
            tags.append("")
    allv, faces = _orient_outward(allv, faces, center)
    # detail tags from the face centres
    fc = np.array([allv[f].mean(0) for f in faces])
    fn = fc - center
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    # trim bands: the ring strips lying between each band's two rings
    for r_lo, r_hi in band_rows:
        for r in range(1, R - 1):
            mid = 0.5 * (S[r] + S[r + 1])
            if ((mid > r_lo) & (mid < r_hi)).mean() > 0.5:
                for k in range(K):
                    tags[r * K + k] = "trim"
    if details:
        lab = detail_at(fc, fn, details, views)
        # slits are local: face by face
        for i, l in enumerate(lab):
            if l == DETAIL_SLIT and tags[i] == "":
                tags[i] = "slit"
    return dict(verts=allv, faces=faces, tags=tags, center=center, rings=R, K=K)


def _orient_outward(verts, faces, center):
    out = []
    for f in faces:
        p = verts[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if n @ (p.mean(0) - center) < 0:
            f = f[::-1]
        out.append(list(f))
    return verts, out


def symmetrize(verts, tol=1e-6):
    """Average each vertex with its mirror partner (nearest vertex to its mirrored position)."""
    from numpy.linalg import norm
    m = mirror(verts)
    out = verts.copy()
    for i, p in enumerate(m):
        j = int(np.argmin(norm(verts - p, axis=1)))
        q = 0.5 * (verts[i] + mirror(verts[j][None])[0])
        out[i] = q
    out[np.abs(out[:, 0]) < tol, 0] = 0.0
    return out
