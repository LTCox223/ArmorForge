"""Build each armor part as its own lofted shape, with cross-sections measured from its mask.

For a part we choose an axis (vertical for torso/leg pieces, the limb direction for arm pieces),
cut it into rings along that axis, and for every ring read the part's width (front/back masks) and
depth (side mask) straight from the part's own coloured pixels. Each ring is a (super)ellipse of that
size and position, so the part keeps its own outline and silhouette instead of inheriting the body's skin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import BARE, NAMES

ARM_PARTS = {"Pauldrons", "Shoulder lames", "Padded sleeves", "Vambraces", "Gauntlets"}
SPLIT_LR = {"Pauldrons", "Shoulder lames", "Padded sleeves", "Vambraces", "Gauntlets", "Upper tassets",
            "Lower tassets", "Thigh plates", "Knee guards", "Greaves", "Armored boots"}


@dataclass
class PartSpec:
    exponent: float = 2.0     # 2 = ellipse, larger = boxier (chunkier plates)
    spacing: float = 0.04     # ring spacing, m
    min_rings: int = 3
    pad: float = 0.006        # minimum stand-off from the body, m
    grow: float = 0.0         # extra outward size on top of the mask, m
    taper_ends: float = 0.0   # fraction the first/last ring shrinks (bevelled ends)


SPECS = {
    "Helmet": PartSpec(2.2, 0.04, pad=0.010),
    "Gorget": PartSpec(2.2, 0.03, pad=0.012, taper_ends=0.06),
    "Cuirass": PartSpec(2.7, 0.05, pad=0.010, taper_ends=0.04),
    "Pauldrons": PartSpec(2.3, 0.045, pad=0.018, grow=0.006, taper_ends=0.08),
    "Shoulder lames": PartSpec(2.3, 0.03, pad=0.012, grow=0.004, taper_ends=0.05),
    "Padded sleeves": PartSpec(2.0, 0.06, pad=0.004),
    "Vambraces": PartSpec(2.2, 0.05, pad=0.010, taper_ends=0.05),
    "Gauntlets": PartSpec(2.4, 0.04, pad=0.010),
    "Belt": PartSpec(2.8, 0.04, pad=0.012),
    "Upper tassets": PartSpec(2.6, 0.04, pad=0.010, taper_ends=0.05),
    "Lower tassets": PartSpec(2.6, 0.04, pad=0.008, taper_ends=0.05),
    "Split tabard": PartSpec(3.0, 0.07, pad=0.004),
    "Thigh plates": PartSpec(2.3, 0.045, pad=0.010, taper_ends=0.04),
    "Knee guards": PartSpec(2.3, 0.04, pad=0.016, grow=0.004, taper_ends=0.08),
    "Greaves": PartSpec(2.2, 0.07, pad=0.010, taper_ends=0.03),
    "Armored boots": PartSpec(2.4, 0.04, pad=0.008, taper_ends=0.04),
}


def _px_world(mask, part, cam, step=3):
    """(a, z) world coordinates of pixels labelled `part` (a = x for front/back, y for sides)."""
    rr, cc = np.nonzero(mask[::step, ::step] == part)
    if not len(rr):
        return np.zeros((0, 2))
    rr, cc = rr * step, cc * step
    a = cam.sign * (cc - cam.c0) / cam.s
    z = (cam.r0 - rr) / cam.s
    return np.stack([a, z], 1)


def _smooth(x, k=1):
    if len(x) < 3:
        return x
    y = x.copy()
    for _ in range(k):
        y = np.convolve(np.pad(y, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    return y


def _fill(x):
    """Linear-interpolate NaNs; if everything is NaN return None."""
    ok = ~np.isnan(x)
    if not ok.any():
        return None
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _axis_frame(part, body_pts, side):
    if part in ARM_PARTS:
        # arm direction from the base-mesh region: shoulder -> hand, in the x-z plane
        c = body_pts - body_pts.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        d = vt[0]
        if d[0] * side < 0:
            d = -d
        A2 = d / np.linalg.norm(d)
    else:
        A2 = np.array([0.0, 1.0])
    U2 = np.array([A2[1], -A2[0]])
    if U2[0] * side < 0 and part in ARM_PARTS:
        U2 = -U2
    if part not in ARM_PARTS:
        U2 = np.array([1.0, 0.0])
    return A2, U2


def fit_part(part, side, v, faces, labels, masks, cams, spec, seg=16):
    """Return (verts, polys) for one part side (side = +1 left/x>0, -1 right, 0 whole part)."""
    pi = NAMES.index(part)
    fsel = labels == pi
    if side:
        cx = v[faces].mean(1)[:, 0]
        fsel &= (cx * side) > 0
    if not fsel.any():
        return None
    body = v[np.unique(faces[fsel])]

    def sel_side(pts_xz):
        if not side:
            return pts_xz
        return pts_xz[(pts_xz[:, 0] * side) > 0.0]

    # pixels of this part in the front/back views (x, z) and the relevant side view (y, z)
    fr = np.vstack([sel_side(_px_world(masks["front"], pi, cams["front"])),
                    sel_side(_px_world(masks["back"], pi, cams["back"]))])
    sv = "side" if part in ARM_PARTS else "side_noarm"
    sd = _px_world(masks[sv], pi, cams[sv])
    if len(fr) >= 50:
        # face labels spill a little at the seams, so trust the mask box and drop stray body points
        lo, hi = fr.min(0) - 0.03, fr.max(0) + 0.03
        keep = ((body[:, 0] > lo[0]) & (body[:, 0] < hi[0]) & (body[:, 2] > lo[1]) & (body[:, 2] < hi[1]))
        if keep.sum() >= 10:
            body = body[keep]
    ref = fr if len(fr) >= 50 else body[:, [0, 2]]
    A2, U2 = _axis_frame(part, ref, side or 1)
    O2 = ref.mean(0)
    # body extents as the floor
    bx = body[:, [0, 2]] - O2
    bs = bx @ A2
    if len(fr):
        sproj = (fr - O2) @ A2
        s0, s1 = np.percentile(sproj, [0.3, 99.7])
    else:
        s0, s1 = bs.min(), bs.max()
    s0, s1 = min(s0, bs.min()), max(s1, bs.max())
    n = max(spec.min_rings, int(round((s1 - s0) / spec.spacing)) + 1)
    ss = np.linspace(s0, s1, n)
    h = (s1 - s0) / max(n - 1, 1) * 0.75 + 1e-4
    cu = np.full(n, np.nan); au = np.full(n, np.nan)
    cw = np.full(n, np.nan); bw = np.full(n, np.nan)
    bcu = np.zeros(n); bau = np.zeros(n); bcw = np.zeros(n); bbw = np.zeros(n)
    frs = (fr - O2) @ A2 if len(fr) else np.zeros(0)
    fru = (fr - O2) @ U2 if len(fr) else np.zeros(0)
    bu = bx @ U2
    for i, s in enumerate(ss):
        m = np.abs(frs - s) <= h
        if m.sum() >= 6:
            lo, hi = np.percentile(fru[m], [1, 99])
            cu[i], au[i] = (lo + hi) / 2, (hi - lo) / 2
        mb = np.abs(bs - s) <= max(h, 0.03)
        if mb.sum() >= 3:
            bcu[i], bau[i] = (bu[mb].min() + bu[mb].max()) / 2, (bu[mb].max() - bu[mb].min()) / 2
            bcw[i], bbw[i] = (body[mb, 1].min() + body[mb, 1].max()) / 2, (body[mb, 1].max() - body[mb, 1].min()) / 2
        else:
            bcu[i], bau[i] = (bu.min() + bu.max()) / 2, (bu.max() - bu.min()) / 2
            bcw[i], bbw[i] = (body[:, 1].min() + body[:, 1].max()) / 2, (body[:, 1].max() - body[:, 1].min()) / 2
        zc = O2[1] + s * A2[1]
        zh = h * abs(A2[1]) + max(au[i] if not np.isnan(au[i]) else bau[i], 0.02) * abs(U2[1])
        ms = np.abs(sd[:, 1] - zc) <= zh if len(sd) else np.zeros(0, bool)
        if ms.sum() >= 6:
            lo, hi = np.percentile(sd[ms, 0], [1, 99])
            cw[i], bw[i] = (lo + hi) / 2, (hi - lo) / 2
    # fall back to the body where the mask has nothing to say, then enforce the stand-off
    for arr, bod in ((cu, bcu), (au, bau), (cw, bcw), (bw, bbw)):
        f = _fill(arr)
        arr[:] = f if f is not None else bod
    au = np.maximum(au + spec.grow, bau + spec.pad)
    bw = np.maximum(bw + spec.grow, bbw + spec.pad)
    cu, au, cw, bw = (_smooth(x, 1) for x in (cu, au, cw, bw))
    if spec.taper_ends:
        for k in (0, -1):
            au[k] *= 1 - spec.taper_ends
            bw[k] *= 1 - spec.taper_ends

    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    e = 2.0 / spec.exponent
    sx, sy = np.sign(c) * np.abs(c) ** e, np.sign(s_) * np.abs(s_) ** e
    verts = []
    for i in range(n):
        base = O2 + ss[i] * A2 + cu[i] * U2            # (x, z) of the ring centre
        for k in range(seg):
            p2 = base + au[i] * sx[k] * U2
            verts.append([p2[0], cw[i] + bw[i] * sy[k], p2[1]])
    verts = np.array(verts)
    polys = []
    for i in range(n - 1):
        for k in range(seg):
            a, b = i * seg + k, i * seg + (k + 1) % seg
            polys.append([a, b, b + seg, a + seg])
    # end caps (triangle fans to a centre point)
    cen0 = verts[:seg].mean(0)
    cen1 = verts[-seg:].mean(0)
    verts = np.vstack([verts, cen0, cen1])
    c0i, c1i = len(verts) - 2, len(verts) - 1
    for k in range(seg):
        polys.append([c0i, k, (k + 1) % seg])
        o = (n - 1) * seg
        polys.append([c1i, o + (k + 1) % seg, o + k])
    return verts, polys


def build_parts(v, faces, labels, masks, cams, seg=16, specs=SPECS):
    faces = np.asarray(faces)
    out = {}
    for pi, name in enumerate(NAMES):
        if pi == BARE:
            continue
        spec = specs.get(name, PartSpec())
        sides = ((1, "L"), (-1, "R")) if name in SPLIT_LR else ((0, ""),)
        for side, tag in sides:
            r = fit_part(name, side, v, faces, labels, masks, cams, spec, seg)
            if r is not None:
                out[f"{name}.{tag}" if tag else name] = r
    return out
