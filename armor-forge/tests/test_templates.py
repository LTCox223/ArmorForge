import numpy as np

from armor_forge.templates import Curve, envelope, robust_plane, robust_polyfit


def test_robust_polyfit_ignores_spikes():
    x = np.linspace(0, 1, 60)
    y = 0.3 + 0.5 * x
    y[[5, 20, 41]] += [0.4, -0.5, 0.3]          # mask noise: a few wild rows
    c = robust_polyfit(x, y, 1)
    assert np.allclose(c, [0.5, 0.3], atol=1e-3)


def test_curve_extends_flat_outside_measured_range():
    z = np.linspace(0.2, 0.5, 30)
    c = Curve(z, 2 * z, 1)
    assert np.isclose(c(0.0), c(0.2)) and np.isclose(c(0.9), c(0.5))


def test_robust_plane_recovers_tilted_hem():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.1, 0.1, 80), rng.uniform(-0.1, 0.1, 80)
    z = 0.7 + 0.2 * x - 0.3 * y
    z[:6] += 0.15                                 # occluded / mislabeled edge samples
    p = robust_plane(x, y, z)
    assert np.allclose(p, [0.7, 0.2, -0.3], atol=1e-3)


def test_envelope_is_smallest_cover_and_never_negative():
    z = np.linspace(0, 1, 50)
    need = np.where(z > 0.7, (z - 0.7) * 0.1, -0.02)   # only the bottom of the piece is too tight
    g = envelope(z, need, deg=1)
    assert np.all(g(z) >= np.maximum(need, 0) - 1e-9)
    assert g(0.0) < 1e-6                                  # untouched where nothing was needed
    assert np.isclose(g(1.0), 0.03, atol=2e-3)
    assert np.all(envelope(z, -np.ones_like(z))(z) == 0)
