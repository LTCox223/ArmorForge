import numpy as np

from armor_forge import nomask as nm


def test_regular_lattice_ignores_text_rows():
    rows = [68.5, 76.5, 103.0, 193.0, 282.5, 373.0, 463.0, 552.5, 642.5, 732.5, 823.0]
    d, chain = nm.regular_lattice(rows)
    assert abs(d - 90.0) < 0.2 and max(chain) == 823.0 and min(chain) == 103.0


def test_head_cut_finds_the_rim_step():
    # a figure 360 px/m: head (spike, wide brow, taper, rim flare) on a neck band, then shoulders
    ppm, z0 = 360.0, 900.0
    fig = np.zeros((1000, 600), bool)
    cx = 300

    def band(z_lo, z_hi, half):
        r0, r1 = int(z0 - z_hi * ppm), int(z0 - z_lo * ppm)
        fig[r0:r1, int(cx - half * ppm):int(cx + half * ppm) + 1] = True
    band(2.00, 2.09, 0.01)    # spike
    band(1.87, 2.00, 0.11)    # brow
    band(1.78, 1.87, 0.095)   # taper
    band(1.76, 1.78, 0.105)   # rim flare
    band(1.71, 1.76, 0.095)   # neck band
    band(1.40, 1.71, 0.30)    # shoulders
    cal = nm.PanelCal(ppm, z0, cx + 0.5)
    cut, info = nm.head_cut(fig, cal)
    assert abs((z0 - cut) / ppm - 1.76) < 0.006


def test_synthetic_side_keeps_target_ratio():
    front = np.zeros((nm.ARTBOARD, nm.ARTBOARD), bool)
    r0, r1 = int(nm.ORIGIN[1] - 1.95 * nm.PPM), int(nm.ORIGIN[1] - 1.80 * nm.PPM)
    half = int(0.10 * nm.PPM)
    front[r0:r1, int(nm.ORIGIN[0]) - half:int(nm.ORIGIN[0]) + half] = True
    target = (np.array([1.7, 2.1]), np.array([0.1, 0.1]), np.array([-0.15, -0.15]), np.array([0.09, 0.09]))
    side = nm.synthetic_side(front, front, target)
    cols = np.flatnonzero(side[(r0 + r1) // 2])
    y = (cols - nm.ORIGIN[0]) / nm.PPM
    assert abs(y.min() - (-0.15)) < 0.003 and abs(y.max() - 0.09) < 0.003
