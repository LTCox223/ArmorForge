import collections
import os

import numpy as np
import pytest

from armor_forge import helmet as hm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MASKS = os.path.join(ROOT, "input", "Masks")
CAMS = os.path.join(ROOT, "armor-forge", "cams.json")


def test_plan_curve_beak_and_round_back():
    t = np.linspace(0, 2 * np.pi, 361)[:-1]
    x, y = hm.plan_curve(0.12, -0.18, 0.12, t, n_front=4.0, n_back=2.2, beak=0.45)
    assert np.isclose(y[0], -0.18) and np.isclose(x[0], 0)            # front centre is the tip
    assert np.isclose(y[180], 0.12)                                     # back centre
    assert np.isclose(np.abs(x).max(), 0.12, atol=1e-3)
    # beak: halfway out to the side, the front has fallen back from the tip
    k = np.argmin(np.abs(x[:90] - 0.06))
    assert y[k] > -0.18 + 0.02


def test_smooth_loop_is_symmetric_and_keeps_mean():
    v = np.random.default_rng(1).normal(size=32)
    out = hm.smooth_loop(v, 1.5)
    assert np.allclose(out, out[(-np.arange(32)) % 32])
    assert abs(out.mean() - v.mean()) < 0.2


@pytest.mark.skipif(not os.path.exists(os.path.join(MASKS, "Front.png")), reason="Embermine masks not present")
def test_embermine_helmet_builds_clean():
    o = hm.load_outlines(MASKS, CAMS)
    v, f, info = hm.build(o)
    assert 1.88 < info["brow"] < 1.98
    assert all(3 <= len(q) <= 4 for q in f)
    edges = collections.Counter()
    for q in f:
        for a, b in zip(q, q[1:] + q[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2
    assert np.allclose(np.sort(v[:, 0]), np.sort(-v[:, 0]), atol=1e-6)   # mirror-symmetric
