import numpy as np

from armor_forge import concept as cc

HELM = (244, 93, 89)
OTHER = (54, 117, 222)


def synthetic(true_reg, art_shape=(1273, 1236), size=1024, cam=(512.0, 928.0), grow=4):
    """A mask artboard (helmet disk on an 'other' body block) and a concept sheet with the same
    figure painted dark on a light panel through true_reg; the painted helmet is `grow` px (mask) wider."""
    rgba = np.zeros((size, size, 4), np.uint8)
    r, c = np.mgrid[:size, :size]
    body = (np.abs(c - cam[0]) < 120) & (r > 400) & (r < 900)
    head = (c - cam[0]) ** 2 + (r - 330) ** 2 < 80 ** 2
    rgba[body] = (*OTHER, 255)
    rgba[head & ~body] = (*HELM, 255)
    art = np.full(art_shape + (3,), 236, np.float32)
    ys, xs = np.mgrid[:art_shape[0], :art_shape[1]] + 0.5
    mc, mr = true_reg.to_mask(xs, ys)
    fig_body = (np.abs(mc - cam[0]) < 120) & (mr > 400) & (mr < 900)
    fig_head = (mc - cam[0]) ** 2 + (mr - 330) ** 2 < (80 + grow) ** 2
    art[fig_body | fig_head] = 70
    return rgba, art


def test_trace_follows_the_art_but_not_into_other_pieces():
    reg = cc.Registration(0.5, 10.0, 20.0, 1.0)
    rgba, art = synthetic(reg)
    m, info = cc.trace_part(art, rgba, reg, HELM, reach_m=0.01, ppm=1600.0, thin_art_px=1.0, margin_m=0.01)
    r, c = np.mgrid[:m.shape[0], :m.shape[1]]
    # grew to the painted radius on top (84 px), stopped at the body block below
    top = r[m].min()
    assert abs(top - (330 - 84)) <= 2
    assert not (m & (rgba[..., 2] == OTHER[2])).any()
    assert info["added_px"] > 0


def test_register_recovers_offset():
    # mask artboard at 400 px/m (a quarter of the real 4096 one) to keep the test small
    init = cc.initial_registration("front", (1273, 1236), (512.0, 928.0), ppm=400.0)
    true = cc.Registration(init.scale * 1.004, init.bx + 3.0, init.by - 2.0, 1.0)
    rgba, art = synthetic(true, grow=0)
    reg = cc.register("front", art, rgba, (512.0, 928.0), ds=2, ppm=400.0)
    # scale and offset trade off a little; what matters is where the figure lands on the art
    pts_c, pts_r = np.array([512.0, 392.0, 632.0, 512.0]), np.array([250.0, 600.0, 600.0, 900.0])
    got, want = np.array(reg.to_art(pts_c, pts_r)), np.array(true.to_art(pts_c, pts_r))
    assert np.abs(got - want).max() < 0.8          # concept pixels
    assert reg.iou > 0.95
