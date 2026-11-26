import collections

import numpy as np
import pytest

from armor_forge import pauldron as pd

CAMS = {v: [2048.0, 3712.0] for v in ("front", "back", "side")}
S = 1600.0
Z0, R_ARM, J = 1.5, 0.07, np.array([0.25, 0.0, 1.5])


def grid():
    n = 4096
    u = (np.arange(n) + 0.5 - 2048) / S
    z = (3712 - np.arange(n) - 0.5) / S
    return u[None, :], z[:, None]


@pytest.fixture(scope="module")
def scene():
    u, z = grid()
    # arm along +x at z = Z0, y = 0
    arm_f = (u > 0.15) & (u < 0.6) & (np.abs(z - Z0) < R_ARM)
    arm_s = (u ** 2 + (z - Z0) ** 2) < R_ARM ** 2
    # cap: upper part of a sphere of radius 0.14 around the shoulder joint
    rc = 0.14
    cap_f = ((u - J[0]) ** 2 + (z - Z0) ** 2 < rc ** 2) & (z > Z0 + 0.03) & (u > 0.12)
    cap_b = ((-u - J[0]) ** 2 + (z - Z0) ** 2 < rc ** 2) & (z > Z0 + 0.03) & (-u > 0.12)
    cap_s = (u ** 2 + (z - Z0) ** 2 < rc ** 2) & (z > Z0 + 0.03)
    # lame: upper half of a tube of radius 0.11 for x in [0.36, 0.44]
    rl = 0.11
    lam_f = (u > 0.36) & (u < 0.44) & (z > Z0) & (z < Z0 + rl)
    lam_b = (-u > 0.36) & (-u < 0.44) & (z > Z0) & (z < Z0 + rl)
    lam_s = (u ** 2 + (z - Z0) ** 2 < rl ** 2) & (z > Z0)
    return dict(arm=(arm_f, arm_s), cap=pd.Views(cap_f, cap_b, cap_s, CAMS),
                lame=pd.Views(lam_f, lam_b, lam_s, CAMS))


def test_axis_from_views(scene):
    ax = pd.axis_from_views(*scene["arm"], CAMS)
    assert np.allclose(ax.d, [1, 0, 0], atol=0.01)
    assert abs(ax.origin[2] - Z0) < 0.002 and abs(ax.origin[1]) < 0.002
    assert np.allclose(ax.up, [0, 0, 1], atol=0.01)
    r = pd.arm_radius(scene["arm"][0], CAMS, ax)
    assert abs(r - R_ARM) < 0.01


def test_cap_and_lame(scene):
    ax = pd.axis_from_views(*scene["arm"], CAMS)
    p = pd.PauldronParams()
    P = pd.build(scene["cap"], scene["lame"], R_ARM, ax, p)
    names = [n for n, _, _ in P.pieces]
    assert names == ["Pauldron cap", "Pauldron lame 1"]
    cv, cf = P.pieces[0][1], P.pieces[0][2]
    assert np.allclose(P.info["joint"][2], Z0, atol=0.01)
    d = np.linalg.norm(cv - np.array(P.info["joint"]), axis=1)
    assert abs(np.median(d) - 0.14) < 0.015              # a dome of the drawn radius
    assert cv[:, 2].min() > Z0                           # over the shoulder, not under the arm
    assert all(len(f) == 4 for f in cf)
    edges = collections.Counter()
    for f in cf:
        for a, b in zip(f, f[1:] + f[:1]):
            edges[(min(a, b), max(a, b))] += 1
    assert max(edges.values()) == 2                      # clean manifold grid
    lv = P.pieces[1][1]
    t, up, back = ax.to_local(lv)
    rad = np.hypot(up, back)
    assert abs(np.median(rad) - 0.11) < 0.015
    assert t.max() < 0.44 - ax.origin[0] + 0.02          # lame stays where it was drawn along the arm
    mv, mf = pd.mirror(lv, P.pieces[1][2])
    assert np.allclose(mv[:, 0], -lv[:, 0])
