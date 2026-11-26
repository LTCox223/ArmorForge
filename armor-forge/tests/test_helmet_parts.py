import collections
import json
import os

import numpy as np
import pytest

from armor_forge import helmet as hm
from armor_forge import helmet_parts as hp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACED = os.path.join(ROOT, "out", "helmet", "traced")
ART = os.path.join(ROOT, "reference", "Embermine Heavy Armor Turnaround.png")
CAMS = os.path.join(ROOT, "armor-forge", "cams.json")


def test_fit_line_ignores_clutter_and_gaps():
    rng = np.random.default_rng(0)
    z = np.concatenate([np.linspace(1.80, 1.86, 60), np.linspace(1.90, 1.95, 50)])   # gap in the line
    u = 0.10 + 0.2 * (z - 1.88)
    cu, cz = rng.uniform(0.0, 0.2, 80), rng.uniform(1.8, 1.95, 80)                 # clutter
    a0, k, n = hp.fit_line(np.r_[u, cu], np.r_[z, cz], (0.03, 0.18), np.linspace(-0.6, 0.6, 13), 1.88,
                           (1.8, 1.95), tol=0.002, step=0.001)
    assert abs(a0 - 0.10) < 0.0015 and abs(k - 0.2) < 1e-9 and n >= 110


def test_band_edge_moves_to_outer_edge():
    b = np.linspace(0, 1, 400)
    off = np.tile(np.linspace(-0.006, 0.006, 8), 50)
    shift = hp.band_edge(0.5 + off, b, 0.5, 0.0, 0.0, (0, 1), outward=1.0)
    assert 0.004 < shift <= 0.006 + 1e-9
    assert hp.band_edge(0.5 + off, b, 0.5, 0.0, 0.0, (0, 1), outward=-1.0) < -0.004


def simple_lines():
    return hp.VisorLines(rear=(-0.05, 0.0, 1.9), top_side=(1.95, 0.0, -0.2), sides=(0.1, 0.0, 1.9),
                         top_front=(1.95, 0.0))


def test_build_visor_is_a_wedge_grid():
    side_front = hm.Profile(np.array([1.7, 2.1]), np.array([-0.2, -0.2]), np.array([0.1, 0.1]))
    flat = (np.array([-0.3, 0.3]), np.array([1.8, 1.8]))
    p = hp.VisorParams(cols=4, rows=5)
    v, f, crease = hp.build_visor(simple_lines(), side_front, flat, flat, p)
    assert v.shape == ((2 * p.cols + 1) * (p.rows + 1), 3) and len(f) == 2 * p.cols * p.rows
    assert np.allclose(v[:, 2].min(), 1.8) and np.allclose(v[:, 2].max(), 1.95)
    ridge = v[np.isclose(v[:, 0], 0)]
    assert np.allclose(ridge[:, 1], -0.2)                                  # ridge on the side outline's front
    corner = v[np.isclose(np.abs(v[:, 0]), 0.1)]
    assert np.allclose(corner[:, 1], -0.05)                                # rear corners on the rear edge
    mid = v[np.isclose(np.abs(v[:, 0]), 0.05)]
    assert np.allclose(mid[:, 1], -0.125)                                  # flat faces in between
    assert len(crease) == p.rows
    edges = collections.Counter()
    for q in f:
        for a, b in zip(q, q[1:] + q[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2
    # faces point forward (away from the wedge's inside)
    q = v[f[0]]
    assert np.cross(q[1] - q[0], q[2] - q[0])[1] < 0


def test_skull_beak_passes_behind_the_visor_corner():
    lines = simple_lines()
    fn = hp.skull_beak(lines, gap=0.004)
    a, yf, yb, n = 0.13, -0.19, 0.13, 6.0
    beak = fn(1.85, a, yf, yb, n)
    t = np.linspace(0, 2 * np.pi, 3601)[:-1]
    x, y = hm.plan_curve(a, yf, yb, t, n, 2.2, beak)
    k = np.argmin(np.abs(x[:900] - (0.1 - 0.004)))
    assert abs(y[k] - (-0.05 + 0.004)) < 0.003
    assert fn(2.05, a, yf, yb, n) == 0.0                                    # round above the visor


@pytest.mark.skipif(not (os.path.exists(os.path.join(TRACED, "trace.json")) and os.path.exists(ART)),
                    reason="Embermine traced masks / concept not present")
def test_embermine_parts():
    from armor_forge import concept as cc
    cams = json.load(open(CAMS))
    meta = json.load(open(os.path.join(TRACED, "trace.json")))
    regs = {v: cc.Registration(**meta[v]["registration"]) for v in ("front", "side", "back")}
    masks = {v: hm.load_mask(os.path.join(TRACED, f"{cc.MASK_FILES[v]}.png")) for v in ("front", "back", "side")}
    parts = hp.build_parts(masks["front"], masks["back"], masks["side"], cc.load_rgb(ART), regs, cams)
    L = parts.lines
    assert 0.08 < L.w(1.85) < 0.12                      # visor half-width
    assert 0.10 < L.y_rear(1.85) - (-0.19) < 0.18       # visor depth, front to rear edge
    assert 1.90 < L.z_top_front(0) < 1.95
    iou = hp.silhouette_iou(parts, masks, cams)
    assert min(iou.values()) > 0.8
    # the skull never sits in front of the visor where the visor covers it
    sv = parts.skull[0]
    vv = parts.visor[0]
    assert sv[:, 1].min() > vv[:, 1].min()
