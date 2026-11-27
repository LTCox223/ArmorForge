"""Cuirass: one barrel round the torso with a flat top band, a neck opening and two armholes.

Structure (after a hand-made base): rings round the torso from the waist up to the shoulders, a flat
band over the shoulders ("yoke") inset to the neck opening, and an armhole cut in each side on mesh
edges (bottom edge on a ring at the armpit, front/back edges on columns, top edge one ring below the
yoke, leaving a strap over the shoulder). Built as the wearer's left half (x >= 0, centre columns on x = 0) for a Mirror modifier.

Measurements (artboard masks, metres):
  * width a(z): the cuirass colour in the front and back views; above the armpit, where pauldrons
    hide the sides, the sides carry on straight up;
  * front y_f(z): the side view's front edge of the cuirass (the breastplate is visible there);
  * back y_b(z): the side view's back edge where it shows, else the torso's back plus the same
    clearance as at the front (the arm hides most of it);
  * armpit height: where the drawn sleeve meets the cuirass in the front view; armhole front/back
    extent: the sleeve in the side view at that height.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import helmet as hm

S = 1600.0


@dataclass
class CuirassParams:
    cols: int = 10              # columns on the half ring (front centre -> back centre)
    ring_step: float = 0.045    # target ring spacing (m)
    n_front: float = 2.6        # plan roundness, front half (2 = ellipse, higher = boxier)
    n_back: float = 3.0
    yoke: float = 0.07          # width of the flat band over the shoulders (m)
    neck_drop: float = 0.05     # the neck opening is this much lower at the front than the back (m)
    clearance: float = 0.02     # minimum gap to the torso (m)
    smooth: float = 0.02        # profile smoothing (m)
    armhole_pad: float = 0.01   # armhole a bit larger than the sleeve (m)
    armhole_side: float = 0.8   # armhole columns only where the ring faces sideways (x >= this x max x)


@dataclass
class Profiles:
    z: np.ndarray
    a: np.ndarray
    yf: np.ndarray
    yb: np.ndarray
    z_bot: float
    z_top: float
    z_armpit: float
    arm_y: tuple                # armhole front / back (y)


def _rows(mask, cam, sign=1.0):
    r, c = np.nonzero(mask)
    return (c + .5 - cam[0]) / S * sign, (cam[1] - r - .5) / S


def _gauss(v, sigma):
    if sigma <= 0:
        return v
    k = int(3 * sigma) + 1
    x = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(v, k, mode="edge"), w, mode="valid")


def measure(front, back, side, sleeve_front, sleeve_side, cams, body, p=CuirassParams(), step=0.005):
    """Profiles from the cuirass colour (front, back, side) and the sleeve colour (front, side).
    body: plate.BodyProfile of the torso (for the hidden back)."""
    fx, fz = _rows(front, cams["front"])
    bx, bz = _rows(back, cams["back"], -1.0)
    z_bot = float(np.percentile(fz, 1))
    # the top: the shoulders (away from the neck), not the collar under the gorget
    shoulder = np.abs(fx) > 0.08
    z_top = float(np.percentile(fz[shoulder], 99)) if shoulder.any() else float(fz.max())
    z = np.arange(z_bot, z_top + 1e-9, step)
    ax_, az_ = np.concatenate([np.abs(fx), np.abs(bx)]), np.concatenate([fz, bz])
    a = np.array([np.percentile(ax_[np.abs(az_ - zz) < step], 99) if (np.abs(az_ - zz) < step).any() else np.nan
                  for zz in z])
    # armpit: lowest point where the sleeve touches the cuirass (front view)
    import scipy.ndimage as ndi
    touch = sleeve_front & ndi.binary_dilation(front, iterations=6)
    tx, tz = _rows(touch, cams["front"])
    z_arm = float(tz.min()) if len(tz) else z_bot + 0.6 * (z_top - z_bot)
    a = np.where(np.isnan(a), np.nanmedian(a), a)
    below = z <= z_arm
    a = np.where(below, a, a[below][-1] if below.any() else a)       # straight sides above the armpit
    a = _gauss(a, p.smooth / step)
    # depth from the side view
    sy, sz = _rows(side, cams["side"])
    yc_body = np.interp(z, body.z, body.yc)
    yf = np.full_like(z, np.nan)
    yb = np.full_like(z, np.nan)
    for i, zz in enumerate(z):
        k = np.abs(sz - zz) < step
        if not k.any():
            continue
        y = sy[k]
        f = y[y < yc_body[i]]
        b = y[y > yc_body[i]]
        if len(f):
            yf[i] = f.min()
        if len(b):
            yb[i] = b.max()
    body_f = np.interp(z, body.z, body.yc - body.b)
    body_b = np.interp(z, body.z, body.yc + body.b)
    ok = ~np.isnan(yf)
    yf = np.interp(z, z[ok], yf[ok]) if ok.any() else body_f - p.clearance
    yf = np.minimum(_gauss(yf, p.smooth / step), body_f - p.clearance)
    clear = np.maximum(body_f - yf, p.clearance)
    yb_body = body_b + clear
    yb = np.where(np.isnan(yb), yb_body, np.maximum(yb, body_b + p.clearance))
    yb = _gauss(yb, p.smooth / step)
    # armhole front/back: the sleeve in the side view just above the armpit
    ly, lz = _rows(sleeve_side, cams["side"])
    k = (lz > z_arm) & (lz < z_arm + 0.12)
    if k.sum() > 20:
        arm_y = (float(np.percentile(ly[k], 5)) - p.armhole_pad, float(np.percentile(ly[k], 95)) + p.armhole_pad)
    else:
        mid = float(np.interp(z_arm, z, 0.5 * (yf + yb)))
        arm_y = (mid - 0.08, mid + 0.08)
    return Profiles(z, a, yf, yb, z_bot, z_top, z_arm, arm_y)


def _half_ring(a, yf, yb, n_front, n_back, cols):
    """cols + 1 points from the front centre (x = 0) round the +x side to the back centre, at equal
    arc length."""
    t = np.linspace(0, np.pi, 361)
    x, y = hm.plan_curve(a, yf, yb, t, n_front, n_back, 0.0)
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], cols + 1)
    X, Y = np.interp(s, cum, x), np.interp(s, cum, y)
    X[0] = X[-1] = 0.0
    return X, Y


def build(pr: Profiles, p=CuirassParams()):
    """-> verts, faces, groups ("steel" | "trim"), info. Half mesh (x >= 0)."""
    C = p.cols
    # ring heights: even spacing, with one ring exactly at the armpit
    n_lo = max(2, int(round((pr.z_armpit - pr.z_bot) / p.ring_step)))
    n_hi = max(2, int(round((pr.z_top - pr.z_armpit) / p.ring_step)))
    zs = np.concatenate([np.linspace(pr.z_bot, pr.z_armpit, n_lo + 1), np.linspace(pr.z_armpit, pr.z_top, n_hi + 1)[1:]])
    r_arm = n_lo
    rings = []
    for z in zs:
        a = float(np.interp(z, pr.z, pr.a))
        yf = float(np.interp(z, pr.z, pr.yf))
        yb = float(np.interp(z, pr.z, pr.yb))
        X, Y = _half_ring(a, yf, yb, p.n_front, p.n_back, C)
        rings.append(np.stack([X, Y, np.full_like(X, z)], 1))
    rings = np.array(rings)                                   # (R, C+1, 3)
    R = len(rings)
    # neck ring: the top ring inset by the yoke width toward the centre, lower at the front
    top = rings[-1]
    yc = 0.5 * (top[0, 1] + top[-1, 1])
    d = top[:, :2] - [0.0, yc]
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    neck = top.copy()
    neck[:, :2] = top[:, :2] - p.yoke * dn
    neck[0, 0] = neck[-1, 0] = 0.0
    theta = np.linspace(0, np.pi, C + 1)
    neck[:, 2] = pr.z_top - p.neck_drop * 0.5 * (1 + np.cos(theta))
    verts = np.vstack([rings.reshape(-1, 3), neck]).copy()
    idx = np.arange(R * (C + 1)).reshape(R, C + 1)
    nidx = R * (C + 1) + np.arange(C + 1)
    # armhole columns: the side columns whose y at the armpit lies in the sleeve's range
    ya = rings[r_arm, :, 1]
    xa = rings[r_arm, :, 0]
    side = xa >= p.armhole_side * xa.max()             # the part of the ring that faces sideways
    cols_in = [c for c in range(C + 1) if side[c] and pr.arm_y[0] <= ya[c] <= pr.arm_y[1]]
    if len(cols_in) < 2:
        mid = int(np.argmax(rings[r_arm, :, 0]))
        cols_in = [mid - 1, mid + 1]
    c0, c1 = min(cols_in), max(cols_in)
    # the armhole stops one ring short of the top: a strap of plate stays over the shoulder
    hole = set((r, c) for r in range(r_arm, max(r_arm + 1, R - 2)) for c in range(c0, c1))
    # round the armhole: its edge points onto the ellipse that fits the cut (side view, y-z)
    rows_h = sorted({r for r, _ in hole})
    r_lo, r_hi = rows_h[0], rows_h[-1] + 1
    edge = [(r, c) for r in range(r_lo, r_hi + 1) for c in (c0, c1)]
    edge += [(r, c) for r in (r_lo, r_hi) for c in range(c0 + 1, c1)]
    E = np.array([verts[idx[r, c]] for r, c in edge])
    yc_, zc_ = 0.5 * (E[:, 1].min() + E[:, 1].max()), 0.5 * (E[:, 2].min() + E[:, 2].max())
    ry, rz = 0.5 * (E[:, 1].max() - E[:, 1].min()), 0.5 * (E[:, 2].max() - E[:, 2].min())
    for (r, c) in edge:
        v = verts[idx[r, c]]
        ang = np.arctan2((v[2] - zc_) / max(rz, 1e-6), (v[1] - yc_) / max(ry, 1e-6))
        verts[idx[r, c], 1] = yc_ + ry * np.cos(ang)
        verts[idx[r, c], 2] = zc_ + rz * np.sin(ang)
    faces, groups, tags = [], [], []
    for r in range(R - 1):
        for c in range(C):
            if (r, c) in hole:
                continue
            faces.append([idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c]])
            tags.append((r, c))
    for c in range(C):                       # yoke: top ring -> neck ring
        faces.append([idx[R - 1, c], idx[R - 1, c + 1], nidx[c + 1], nidx[c]])
        tags.append((R - 1, c))
    # trim: the faces round the armhole and the waist row
    for (r, c) in tags:
        near_hole = (r_arm - 1 <= r <= R - 2 and (c0 - 1 <= c <= c1)) and (r, c) not in hole
        groups.append("trim" if (near_hole or r == 0) else "steel")
    # outward winding (away from the torso's centre line)
    f0 = verts[faces[len(faces) // 3]]
    n = np.cross(f0[1] - f0[0], f0[2] - f0[0])
    cen = np.array([0.0, yc, f0[:, 2].mean()])
    if n @ (f0.mean(0) - cen) < 0:
        faces = [f[::-1] for f in faces]
    info = dict(rings=R, cols=C, r_armpit=r_arm, armhole_cols=(c0, c1), z=zs.tolist())
    return verts, faces, groups, info


def mirror_full(verts, faces):
    """Both halves as one mesh (for checks; the .blend uses a Mirror modifier)."""
    n = len(verts)
    V = np.vstack([verts, verts * [-1, 1, 1]])
    F = list(faces) + [[i + n for i in f[::-1]] for f in faces]
    return V, F
