import numpy as np

from armor_forge.projection import (mask_from_pixels, ortho_pixel_coords, sample_mask,
                                    view_direction)
from armor_forge.selection import (face_adjacency, face_centers_normals, grow_selection,
                                   select_faces_for_view)


def front_camera(y=-5.0):
    """Blender camera with rotation (90deg, 0, 0): looks along +Y, image-up is world +Z."""
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    m[:3, 3] = [0, y, 0]
    return m


def test_view_direction_front():
    assert np.allclose(view_direction(front_camera()), [0, 1, 0])


def test_ortho_projection_known_point():
    px, depth = ortho_pixel_coords([[0.3, 0.0, 0.5]], front_camera(), 2.0, (100, 200))
    assert np.allclose(px, [[80, 150]])  # 0.01 world units per pixel
    assert np.allclose(depth, [5.0])


def test_mask_alpha_and_luminance():
    w, h = 4, 2
    rgba = np.zeros((h, w, 4))
    rgba[..., 3] = 1.0
    rgba[0, 1] = [1, 1, 1, 1]  # opaque image: brightness decides
    m = mask_from_pixels(rgba.ravel(), w, h)
    assert m[0, 1] and m.sum() == 1
    rgba[..., 3] = 0.0
    rgba[1, 2, 3] = 1.0  # transparent image: alpha decides
    m = mask_from_pixels(rgba.ravel(), w, h)
    assert m[1, 2] and m.sum() == 1


def test_sample_mask_bounds():
    mask = np.zeros((10, 10), bool)
    mask[2, 3] = True
    px = np.array([[3.5, 2.5], [-1, 0], [3.5, 99]])
    assert list(sample_mask(mask, px)) == [True, False, False]


def grid(n):
    verts = [(x, 0.0, z) for z in range(n + 1) for x in range(n + 1)]
    faces = []
    for z in range(n):
        for x in range(n):
            a = z * (n + 1) + x
            faces.append((a, a + 1, a + n + 2, a + n + 1))
    return np.array(verts, float), faces


def test_centers_and_normals():
    verts, faces = grid(2)
    c, n = face_centers_normals(verts, faces)
    assert np.allclose(c[0], [0.5, 0, 0.5])
    assert np.allclose(np.abs(n), [[0, 1, 0]] * 4)


def test_select_respects_mask_facing_and_occlusion():
    verts, faces = grid(2)
    verts = verts - [1, 0, 1]  # center on origin
    c, n = face_centers_normals(verts, faces)
    if n[0][1] > 0:  # make normals face the front camera (-Y)
        n = -n
    cam = front_camera()
    w, h = 100, 100
    mask = np.zeros((h, w), bool)
    mask[50:, :] = True  # top half of the image only
    sel = select_faces_for_view(c, n, cam, 4.0, (w, h), mask)
    assert sel.sum() == 2  # the two top faces
    occ = np.array([True, False, False, False])
    sel2 = select_faces_for_view(c, n, cam, 4.0, (w, h), np.ones((h, w), bool), occluded=occ)
    assert sel2.sum() == 3
    sel3 = select_faces_for_view(c, -n, cam, 4.0, (w, h), np.ones((h, w), bool))
    assert sel3.sum() == 0  # back-facing faces are never picked


def test_grow_selection():
    _, faces = grid(3)
    adj = face_adjacency(faces)
    sel = np.zeros(9, bool)
    sel[4] = True  # centre face
    assert grow_selection(adj, sel, 0).sum() == 1
    assert grow_selection(adj, sel, 1).sum() == 5  # edge neighbours only
    assert grow_selection(adj, sel, 2).sum() == 9
