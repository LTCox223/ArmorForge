"""Close-helm shape from a concept's front/side/back outlines (shape and curvature only).

The helmet is a loft of horizontal slices. Each slice's plan (top-down) shape is:
  * back half:  a superellipse (round),
  * front half: in the visor zone a boxy superellipse whose depth falls off toward the sides, so the
    face is two flat-ish planes meeting at a centre ridge (the visor "beak"); above the brow it blends
    to round.
Slice size comes from the outlines: half-width a(z) from front/back, front y_f(z) and back y_b(z) from
the side. The brow height is where the side profile steps back. The crown closes along a front-to-back
ridge line (no pole), the bottom is an open hem following the drawn bottom edge.

No body, no features: this is about getting the form right.
Coordinates: metres, Z up, the helmet faces -Y, +X is the wearer's left.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

HELMET_RGB = (244, 93, 89)


# ----------------------------------------------------------------------------- outlines
def load_mask(path, rgb=HELMET_RGB):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGBA"))
    return (a[..., 0] == rgb[0]) & (a[..., 1] == rgb[1]) & (a[..., 2] == rgb[2]) & (a[..., 3] > 128)


def _median(a, k):
    pad = np.pad(a, k, mode="edge")
    return np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(a))])


def _gauss(a, sigma):
    if sigma <= 0:
        return a
    k = int(3 * sigma) + 1
    x = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(a, k, mode="edge"), w, mode="valid")


@dataclass
class Profile:
    z: np.ndarray        # ascending heights (m)
    lo: np.ndarray       # min u per height
    hi: np.ndarray       # max u per height

    def at(self, z):
        return np.interp(z, self.z, self.lo), np.interp(z, self.z, self.hi)


def profile(mask, c0, r0, s=1600.0, sign=1.0, median_px=6, sigma_px=5, step_px=4):
    """Per-row extents of a mask, cleaned: median (kills pixel bumps, keeps steps) then light blur."""
    rows = np.flatnonzero(mask.any(1))
    rows = rows[(rows - rows[0]) % step_px == 0]
    lo, hi, z = [], [], []
    for r in rows:
        cols = np.flatnonzero(mask[r])
        u = (cols + 0.5 - c0) / s * sign
        lo.append(u.min())
        hi.append(u.max())
        z.append((r0 - r - 0.5) / s)
    z, lo, hi = np.array(z)[::-1], np.array(lo)[::-1], np.array(hi)[::-1]
    k = max(1, median_px // step_px)
    sg = sigma_px / step_px
    return Profile(z, _gauss(_median(lo, k), sg), _gauss(_median(hi, k), sg))


def bottom_edge(mask, c0, r0, s=1600.0, sign=1.0, sigma_px=24):
    """Lowest drawn z per column, as (u ascending, z), smoothed."""
    cols = np.flatnonzero(mask.any(0))
    u = (cols + 0.5 - c0) / s * sign
    z = np.array([(r0 - np.flatnonzero(mask[:, c]).max() - 0.5) / s for c in cols])
    o = np.argsort(u)
    return u[o], _gauss(_median(z[o], 8), sigma_px)


@dataclass
class Outlines:
    width: Profile        # half-width, symmetric (front and back views averaged)
    side: Profile         # lo = front (min y), hi = back (max y)
    bottom_front: tuple
    bottom_back: tuple
    bottom_side: tuple

    @property
    def z_top(self):
        return min(self.width.z.max(), self.side.z.max())


def load_outlines(mask_dir, cams_json, s=1600.0):
    cams = json.load(open(cams_json))
    f = load_mask(os.path.join(mask_dir, "Front.png"))
    b = load_mask(os.path.join(mask_dir, "Back.png"))
    sd = load_mask(os.path.join(mask_dir, "Side.png"))
    pf = profile(f, *cams["front"], s)
    pb = profile(b, *cams["back"], s, sign=-1.0)
    # symmetric half-width: mean of |left| and |right| in each view, then the larger of the two views
    z = np.union1d(pf.z, pb.z)
    hf = 0.5 * (np.abs(np.interp(z, pf.z, pf.lo)) + np.interp(z, pf.z, pf.hi))
    hb = 0.5 * (np.abs(np.interp(z, pb.z, pb.lo)) + np.interp(z, pb.z, pb.hi))
    inside_f = (z >= pf.z.min()) & (z <= pf.z.max())
    inside_b = (z >= pb.z.min()) & (z <= pb.z.max())
    half = np.where(inside_f & inside_b, np.maximum(hf, hb), np.where(inside_f, hf, hb))
    width = Profile(z, -half, half)
    side = profile(sd, *cams["side"], s)
    return Outlines(width, side, bottom_edge(f, *cams["front"], s), bottom_edge(b, *cams["back"], s, -1.0),
                    bottom_edge(sd, *cams["side"], s))


# ----------------------------------------------------------------------------- shape
@dataclass
class HelmetParams:
    n_back: float = 2.2          # back half roundness (2 = ellipse, higher = boxier)
    n_visor: float = 4.0         # visor front half boxiness
    n_skull: float = 2.2         # front half above the brow
    beak: float = 0.45           # how far the visor face falls back toward its sides (0 = flat front)
    brow_blend: float = 0.012    # height over which the visor blends into the skull (m)
    K: int = 32                  # points around each slice (multiple of 4)
    ring_spacing: float = 0.018  # target distance between slices along the profile (m)
    smooth: float = 0.010        # outline smoothing (m); never across the brow step
    crest_width: float = 0.0     # front-view width below which the top counts as a separate crest (0 = keep the comb)
    dome_start: float = 1.0      # where between brow and top to force a rounded dome (1 = use the drawn profile)
    dome_exp: float = 2.2        # dome roundness (2 = elliptic)
    dome_shift: float = 0.0      # move the dome top back (+) or forward (-), fraction of depth
    min_visor: float = 0.06      # hem stays at least this far below the brow (m)
    hem_sigma: float = 1.5       # hem smoothing around the loop (in points)


def find_brow(o: Outlines, search=(0.35, 0.85)):
    """Height where the side profile's front steps back the most (visor top / brow)."""
    z = o.side.z
    zr = z.min() + np.array(search) * (z.max() - z.min())
    m = (z > zr[0]) & (z < zr[1])
    yf = o.side.lo
    dy = np.gradient(yf, z)
    i = np.flatnonzero(m)[np.argmax(dy[m])]
    return float(z[i])


def plan_curve(a, yf, yb, t, n_front, n_back, beak):
    """Plan shape of one slice at parameters t in [0, 2pi): t = 0 front centre, pi/2 left side,
    pi back centre. Returns x, y."""
    ym = 0.5 * (yf + yb)
    c, s = np.cos(t), np.sin(t)
    front = c > 0
    n = np.where(front, n_front, n_back)
    sx = np.sign(s) * np.abs(s) ** (2 / n)
    sy = np.abs(c) ** (2 / n)
    x = a * sx
    depth = np.where(front, ym - yf, yb - ym)
    fall = np.where(front, 1 - beak * np.abs(sx), 1.0)          # visor face falls back at the sides
    y = np.where(front, ym - depth * sy * fall, ym + depth * sy)
    return x, y


def resample_closed(x, y, K):
    """K points at equal arc length, anchored at the front centre (index 0) and symmetric."""
    # use one half (front centre -> back centre on +x side) and mirror
    n = len(x)
    half = n // 2 + 1
    hx, hy = x[:half], y[:half]
    seg = np.hypot(np.diff(hx), np.diff(hy))
    cum = np.concatenate([[0], np.cumsum(seg)])
    tgt = np.linspace(0, cum[-1], K // 2 + 1)
    rx, ry = np.interp(tgt, cum, hx), np.interp(tgt, cum, hy)
    X = np.concatenate([rx, -rx[-2:0:-1]])
    Y = np.concatenate([ry, ry[-2:0:-1]])
    return X, Y


def hem_heights(o: Outlines, x, y):
    """Bottom edge at plan points: front points read the front view, back points the back view,
    side-facing points the side view; blended by how much each view sees the point."""
    ym = np.median(y)
    zf = np.interp(x, *o.bottom_front)
    zb = np.interp(x, *o.bottom_back)
    zs = np.interp(y, *o.bottom_side)
    ang = np.arctan2(x, -(y - ym))
    wf = np.clip(np.cos(ang), 0, None) ** 2
    wb = np.clip(-np.cos(ang), 0, None) ** 2
    ws = np.sin(ang) ** 2
    return (wf * zf + wb * zb + ws * zs) / (wf + wb + ws)


def _smooth_split(z, v, at, sigma_m):
    """Gaussian smoothing in z, done separately below and above `at` (keeps a step there)."""
    dz = np.median(np.diff(z))
    sg = sigma_m / dz
    out = v.copy()
    lo = z < at
    if lo.sum() > 3:
        out[lo] = _gauss(v[lo], sg)
    if (~lo).sum() > 3:
        out[~lo] = _gauss(v[~lo], sg)
    return out


def shape_profiles(o: Outlines, p: "HelmetParams"):
    """Clean, design-level profiles on a common z grid: heavy smoothing (but not across the brow),
    crest split off the top, dome rounded. -> dict of arrays + scalars."""
    brow = find_brow(o)
    z_lo = max(o.width.z.min(), o.side.z.min())
    z = np.linspace(z_lo, o.z_top, 300)
    a = np.abs(o.width.at(z)[1])
    yf, yb = o.side.at(z)
    # crest: the narrow strip on top of the dome (front view narrower than crest_width)
    wide = np.flatnonzero(a > p.crest_width)
    z_dome = z[wide.max()] if len(wide) else o.z_top
    crest_h = o.z_top - z_dome
    # heavy smoothing, separately below / above the brow
    a = _smooth_split(z, a, brow, p.smooth)
    yf = _smooth_split(z, yf, brow, p.smooth)
    yb = _smooth_split(z, yb, brow, p.smooth)
    # dome: above z_round, replace each extent with a superellipse arc that closes at the dome top
    z_round = brow + p.dome_start * (z_dome - brow)
    k = np.searchsorted(z, z_round)
    t = np.clip((z - z_round) / max(z_dome - z_round, 1e-6), 0, 1)
    arc = (1 - t ** p.dome_exp) ** (1 / p.dome_exp)
    ym_top = 0.5 * (yf[k] + yb[k]) + p.dome_shift * (yb[k] - yf[k])
    a = np.where(z > z_round, a[k] * arc, a)
    yf = np.where(z > z_round, ym_top + (yf[k] - ym_top) * arc, yf)
    yb = np.where(z > z_round, ym_top + (yb[k] - ym_top) * arc, yb)
    keep = z <= z_dome
    return dict(z=z[keep], a=a[keep], yf=yf[keep], yb=yb[keep], brow=brow, z_dome=z_dome,
                crest_h=crest_h, ym_top=ym_top)


def smooth_loop(v, sigma):
    """Gaussian smoothing around a closed loop (sigma in points)."""
    k = int(3 * sigma) + 1
    x = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    w /= w.sum()
    n = len(v)
    pad = np.concatenate([v[-k:], v, v[:k]])
    out = np.convolve(pad, w, mode="valid")
    return 0.5 * (out + out[(-np.arange(n)) % n])        # keep it mirror-symmetric


def smooth_hem(theta, z_raw, order=3, iters=4):
    """Robust low-order cosine series around the helmet (symmetric)."""
    A = np.stack([np.cos(j * theta) for j in range(order + 1)], 1)
    w = np.ones(len(theta))
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * w[:, None], z_raw * w, rcond=None)
        res = z_raw - A @ coef
        sc = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / np.maximum(1.0, np.abs(res) / (2.0 * sc))
    return A @ coef


def slicer(P, p: HelmetParams, K=None):
    """-> function z -> (X, Y): the plan (top-down) shape of the helmet at height z, K points."""
    K = K or p.K
    zz, a_, yf_, yb_, brow = P["z"], P["a"], P["yf"], P["yb"], P["brow"]
    t_fine = np.linspace(0, 2 * np.pi, 721)[:-1]

    def slice_at(z):
        a = np.interp(z, zz, a_)
        yf, yb = np.interp(z, zz, yf_), np.interp(z, zz, yb_)
        w = np.clip((brow - z) / p.brow_blend + 0.5, 0, 1)          # 1 in the visor, 0 above the brow
        nf = w * p.n_visor + (1 - w) * p.n_skull
        x, y = plan_curve(a, yf, yb, t_fine, nf, p.n_back, p.beak * w)
        return resample_closed(x, y, K)
    return slice_at


def build(o: Outlines, p: HelmetParams = HelmetParams()):
    """-> verts (N, 3), quad faces, info dict."""
    K = p.K
    P = shape_profiles(o, p)
    brow, z_top = P["brow"], P["z_dome"]
    zz = P["z"]
    z_lo = zz[0]
    a_, yf_, yb_ = P["a"], P["yf"], P["yb"]
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(zz), np.hypot(np.diff(a_), np.diff(yf_))))])
    n_rings = max(4, int(np.ceil(arc[-1] / p.ring_spacing)))
    slice_at = slicer(P, p)

    # ring parameter u in [0, 1]: u = 0 at the hem (varies around), 1 at the top ridge
    us = np.interp(np.linspace(0, arc[-1], n_rings + 1), arc, np.linspace(0, 1, len(zz)))
    # hem: from a slice near the bottom, read the drawn bottom edge around it
    x0, y0 = slice_at(z_lo + 0.25 * (brow - z_lo))
    hem = hem_heights(o, x0, y0)
    theta = np.arctan2(x0, -(y0 - np.median(y0)))
    hem = smooth_loop(hem, p.hem_sigma)
    hem = np.clip(hem, z_lo, brow - p.min_visor)
    rings = []
    for u in us[:-1]:
        zk = hem + u * (z_top - hem)            # per point height for this ring
        pts = np.zeros((K, 3))
        # evaluate the slice at each point's own height
        for k in range(K):
            X, Y = slice_at(zk[k])
            pts[k] = (X[k], Y[k], zk[k])
        rings.append(pts)
    # top: the ridge line, left and right halves meet at x = 0
    # close the dome: the last ring's left and right halves meet on the centre line
    ridge = np.zeros((K // 2 + 1, 3))
    yf, yb = np.interp(z_top, zz, yf_), np.interp(z_top, zz, yb_)
    ridge[:, 1] = np.linspace(yf, yb, K // 2 + 1)
    ridge[:, 2] = z_top
    rings = np.array(rings)
    R = len(rings)
    verts = np.vstack([rings.reshape(-1, 3), ridge])
    idx = np.arange(R * K).reshape(R, K)
    rid = R * K + np.array([min(k, K - k) for k in range(K + 1)])
    faces = []
    for r in range(R - 1):
        for k in range(K):
            k2 = (k + 1) % K
            faces.append([idx[r, k], idx[r, k2], idx[r + 1, k2], idx[r + 1, k]])
    for k in range(K):
        k2 = (k + 1) % K
        a_, b_ = rid[k], rid[k + 1]
        if a_ == b_:
            continue
        faces.append([idx[R - 1, k], idx[R - 1, k2], b_, a_])
    # outward winding: check one face against the centre
    c = verts.mean(0)
    f0 = verts[faces[len(faces) // 2]]
    nrm = np.cross(f0[1] - f0[0], f0[2] - f0[0])
    if nrm @ (f0.mean(0) - c) < 0:
        faces = [f[::-1] for f in faces]
    # merge the degenerate end quads at the ridge ends into triangles
    out = []
    for f in faces:
        g = []
        for v in f:
            if not g or g[-1] != v:
                g.append(v)
        if len(g) > 1 and g[0] == g[-1]:
            g.pop()
        if len(set(g)) >= 3:
            out.append(g)
    # edges to crease for subdivision: the visor's centre line (front meridian below the brow) and
    # the ring nearest the brow (the visor's top edge)
    brow_ring = int(np.argmin([abs(np.mean(rings[r, :, 2]) - brow) for r in range(R)]))
    crease = [(idx[r, 0], idx[r + 1, 0]) for r in range(R - 1) if rings[r + 1, 0, 2] <= brow + 0.005]
    crease += [(idx[brow_ring, k], idx[brow_ring, (k + 1) % K]) for k in range(K)
               if rings[brow_ring, k, 1] < np.median(rings[brow_ring, :, 1])]
    return verts, out, dict(brow=brow, z_top=z_top, rings=R, K=K, crest_h=P["crest_h"],
                            profiles=P, crease=crease)
