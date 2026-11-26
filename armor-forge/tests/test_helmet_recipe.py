import json
import os

import numpy as np
import pytest

from armor_forge import helmet as hm
from armor_forge import helmet_recipe as hr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACED = os.path.join(ROOT, "out", "helmet", "traced")
CAMS = {v: [2048.0, 3712.0] for v in ("front", "back", "side")}


def ellipse_masks(a=0.11, d=0.15, zc=1.86, h=0.17, yc=-0.02):
    """A dome-ish helmet: half-width a, half-depth d, from zc - h/2 .. zc + h (upper half rounded)."""
    S = 4096
    r = np.arange(S)
    z = (3712 - r - 0.5) / 1600
    c = (np.arange(S) + 0.5 - 2048) / 1600
    zlo, ztop = zc - 0.5 * h, zc + h
    t = np.clip((z - zc) / (ztop - zc), 0, 1)
    shrink = np.where((z >= zlo) & (z <= ztop), np.sqrt(np.clip(1 - t ** 2, 0, 1)), 0)
    front = np.abs(c)[None, :] <= (a * shrink)[:, None]
    side = np.abs(c - yc)[None, :] <= (d * shrink)[:, None]
    return front, front.copy(), side


@pytest.fixture(scope="module")
def outlines():
    f, b, s = ellipse_masks()
    return f, b, s, hm.outlines_from_masks(f, b, s, CAMS)


@pytest.mark.parametrize("mode", ["separate", "integral", "open"])
def test_split_modes(outlines, mode):
    f, b, s, o = outlines
    p = hm.HelmetParams(brow=1.88)
    seam = hm.Seam(lambda x: 1.88 + 0 * np.asarray(x, float), 0.08)
    v, faces, info = hm.build(o, p, seam=seam)
    h = hr.split(v, faces, info, hr.FaceParams(mode=mode))
    assert len(h.face.faces) > 0 and len(h.skull.faces) > 0
    assert h.weld == (mode != "separate")
    sk = h.skull.verts[h.skull.seam]
    fc = h.face.verts[h.face.seam]
    # every face-plate seam point has a skull seam point at the same place (exact when welded)
    dist = np.min(np.linalg.norm(fc[:, None] - sk[None], axis=2), axis=1)
    if mode == "separate":
        assert 0.002 < dist.max() < 0.01
    else:
        assert dist.max() < 1e-9
    # the face plate is in front and below the brow
    assert h.face.verts[:, 1].mean() < h.skull.verts[:, 1].mean()
    assert h.face.verts[:, 2].max() <= 1.88 + 0.006
    if mode == "open":
        n_full = int(info["face_mask"].sum())
        assert len(h.face.faces) < n_full           # only a rim is kept


def test_no_split_keeps_old_behaviour(outlines):
    f, b, s, o = outlines
    v, faces, info = hm.build(o, hm.HelmetParams())
    assert "face_mask" not in info and len(faces) > 100


@pytest.mark.skipif(not os.path.exists(os.path.join(TRACED, "trace.json")), reason="Embermine traced masks not present")
def test_embermine_recipe_defaults_without_art():
    cams = json.load(open(os.path.join(ROOT, "armor-forge", "cams.json")))
    m = {v: hm.load_mask(os.path.join(TRACED, f"{n}.png")) for v, n in (("front", "Front"), ("back", "Back"), ("side", "Side"))}
    h = hr.build(m["front"], m["back"], m["side"], cams, hr.FaceParams(mode="integral"))
    assert h.info["front_lines"] is None                 # no art: default seam
    iou = hr.silhouette_iou(h, m, cams)
    assert min(iou.values()) > 0.8


def test_untwist_moves_the_rear_edge_by_the_turn():
    from armor_forge import helmet_parts as hp
    L = hp.VisorLines(rear=(-0.05, 0.0, 1.85), top_side=(1.93, 0.2, -0.19), sides=(0.10, 0.0, 1.85),
                      top_front=(1.92, 0.3))
    t = np.radians(-4.2)
    U = hr.untwist_lines(L, t)
    # drawn u = y cos - x sin at the near corner x = w: y = (u + w sin) / cos
    assert np.isclose(U.y_rear(1.85), (-0.05 + 0.10 * np.sin(t)) / np.cos(t))
    assert U.y_rear(1.85) < L.y_rear(1.85)               # the true rear edge is further forward
    assert hr.untwist_lines(L, 0.0) is L
