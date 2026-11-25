import numpy as np

from armor_forge import shellfit as sf


def make_views(fill=None, size=256, f=16.0):
    """Synthetic 4096-px-equivalent views at 1/f resolution, origin at the image centre bottom."""
    views = {}
    for name in sf.VIEW_FILES:
        lab = np.zeros((size, size), np.uint8)
        if fill:
            fill(name, lab)
        views[name] = sf.View(name, lab, 2048.0, 3712.0, 1600.0, f)
    return views


def grid(nx, nz, x0=0.0, x1=0.3, z0=1.0, z1=1.3, y=-0.1):
    xs, zs = np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1)
    verts = np.array([(x, y, z) for z in zs for x in xs])
    idx = np.arange(len(verts)).reshape(nz + 1, nx + 1)
    faces = [[idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]] for i in range(nz) for j in range(nx)]
    return verts, faces


def test_projection_matches_layout():
    v = make_views()["front"]
    col, row = v.project(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
    assert np.allclose(col, [2048 / 16, (2048 + 1600) / 16])
    assert np.allclose(row, [3712 / 16, (3712 - 1600) / 16])
    back = make_views()["back"]
    assert back.project(np.array([[1.0, 0, 0]]))[0][0] < 2048 / 16      # back view is mirrored


def test_assign_labels_fills_gaps_and_absorbs_specks():
    verts, faces = grid(6, 6)
    adj = sf.face_adjacency(faces)
    votes = np.zeros((len(faces), 18))
    votes[:18, 3] = 1.0          # bottom half cuirass
    votes[18:, 9] = 1.0          # top half belt
    votes[20] = 0                # unseen face -> filled
    votes[30, 9], votes[30, 4] = 0, 1.0   # one-face speck -> absorbed
    lab = sf.assign_labels(votes, adj)
    assert (lab[:18] == 3).all()
    assert lab[20] == 9 and lab[30] == 9


def test_upper_layer_wins_when_it_has_a_fair_share():
    verts, faces = grid(2, 1)
    votes = np.zeros((2, 18))
    votes[:, 6] = 1.0            # sleeve
    votes[0, 4] = 0.5            # pauldron (over sleeve) with half the votes
    lab = sf.prefer_upper_layers(np.array([6, 6]), votes, 0.35)
    assert lab.tolist() == [4, 6]


def test_hanging_votes_become_undersuit():
    verts, faces = grid(3, 3)
    votes = np.zeros((len(faces), 18))
    votes[:, 12] = 1.0
    lab = sf.assign_labels(votes, sf.face_adjacency(faces), skip_ids=[12])
    assert (lab == sf.BASE).all()


def test_lift_stops_at_outline():
    # side view: piece 14 drawn from y = -0.20 to 0 at every height
    def fill(name, lab):
        if name.startswith("side"):
            c0 = int((2048 - 0.20 * 1600) / 16)
            lab[:, c0:2048 // 16] = 14
    views = make_views(fill)
    p = np.array([[0.0, -0.10, 1.0]])
    n = np.array([[0.0, -1.0, 0.0]])          # facing forward: only the side view constrains it
    t = sf.lift_distance(p, n, views, 14, 0.3, np.array([False]))
    assert abs(t[0] - 0.10) < 0.02


def test_lift_ignores_views_that_start_on_another_piece():
    def fill(name, lab):
        if name.startswith("side"):
            lab[:, :] = 15            # greave drawn over the spot: can't tell
    views = make_views(fill)
    t = sf.lift_distance(np.array([[0.0, -0.1, 1.0]]), np.array([[0.0, -1.0, 0.0]]), views, 14, 0.05,
                         np.array([False]))
    assert t[0] == 0.05


def test_snap_moves_hem_onto_outline():
    # piece 3 drawn below z = 1.17; the selected faces stop at a grid line z = 1.15
    def fill(name, lab):
        if name == "front":
            r = int((3712 - 1.17 * 1600) / 16)
            lab[r:, :] = 3
    views = make_views(fill)
    verts, faces = grid(6, 6)                 # z rows every 0.05
    piece = [i for i, f in enumerate(faces) if verts[f].mean(0)[2] < 1.15]
    vnames = np.full(len(verts), "front", object)
    seam = np.zeros(len(verts), bool)
    out = sf.snap_boundary(verts, faces, piece, 3, views, vnames, seam, smooth_iters=0)
    top = np.isclose(verts[:, 2], 1.15) & (verts[:, 0] > 0.01) & (verts[:, 0] < 0.29)
    assert np.all(out[top, 2] > 1.155) and np.all(out[top, 2] < 1.185)
    assert np.allclose(out[~top & (verts[:, 2] < 1.14)], verts[~top & (verts[:, 2] < 1.14)])


def test_hanging_panel_follows_outline():
    # front view: tabard 12 is 0.3 m wide from z 0.7 to 1.1; side view puts it at y = -0.15
    def fill(name, lab):
        r0, r1 = int((3712 - 1.1 * 1600) / 16), int((3712 - 0.7 * 1600) / 16)
        if name in ("front", "back"):
            c0, c1 = int((2048 - 0.15 * 1600) / 16), int((2048 + 0.15 * 1600) / 16)
            lab[r0:r1, c0:c1] = 12
        else:
            c = int((2048 - 0.15 * 1600) / 16)
            lab[r0:r1, c:c + 2] = 12
    views = make_views(fill)
    verts, faces = sf.hanging_panel(views, 12, "front", lambda x, z: -0.10, gap=0.01, cell=0.05)
    assert verts is not None and len(faces) > 0
    used = verts[np.unique(np.concatenate(faces))]
    assert used[:, 0].min() >= 0 and abs(used[:, 0].max() - 0.15) < 0.02
    assert abs(used[:, 2].max() - 1.1) < 0.02 and abs(used[:, 2].min() - 0.7) < 0.02
    assert np.all(used[:, 1] <= -0.11 + 1e-6)          # at least gap in front of what's under it
    # normals point away from the body (-Y for a front panel)
    a, b, c = verts[faces[0][:3]]
    assert np.cross(b - a, c - a)[1] < 0
