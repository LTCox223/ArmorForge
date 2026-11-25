"""Pure-numpy projection helpers (no Blender needed, so they are unit-testable).

Conventions match Blender:
  * a camera looks down its local -Z axis, local +Y is image "up"
  * image rows are stored bottom-to-top (row 0 is the BOTTOM row), as in bpy.types.Image.pixels
  * for an orthographic camera, ortho_scale spans the LARGER image dimension
"""
from __future__ import annotations

import numpy as np


def ortho_pixel_coords(points_world, cam_world_matrix, ortho_scale, res):
    """Project world-space points through an orthographic camera.

    points_world: (N, 3); cam_world_matrix: 4x4; res: (width, height) in pixels.
    Returns (pixels (N, 2) with y measured from the BOTTOM, depth (N,)) where depth is the
    distance in front of the camera (positive = visible side of the camera plane).
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    inv = np.linalg.inv(np.asarray(cam_world_matrix, dtype=float))
    local = np.c_[pts, np.ones(len(pts))] @ inv.T
    w, h = res
    world_per_px = ortho_scale / max(w, h)
    px = np.c_[w / 2.0 + local[:, 0] / world_per_px, h / 2.0 + local[:, 1] / world_per_px]
    return px, -local[:, 2]


def view_direction(cam_world_matrix):
    """Unit world-space direction the camera looks along (its local -Z)."""
    d = np.asarray(cam_world_matrix, dtype=float)[:3, :3] @ np.array([0.0, 0.0, -1.0])
    return d / np.linalg.norm(d)


def mask_from_pixels(flat_rgba, width, height, threshold=0.5):
    """Flat RGBA float array (bpy Image.pixels order) -> (H, W) bool mask, row 0 = bottom.

    If the image has real transparency, alpha is the mask; otherwise brightness is.
    """
    arr = np.asarray(flat_rgba, dtype=float).reshape(height, width, 4)
    alpha = arr[..., 3]
    if alpha.min() < 0.99:
        return alpha > threshold
    lum = arr[..., :3].mean(axis=-1)
    return lum > threshold


def sample_mask(mask, pixel_coords):
    """True where a pixel coordinate lands inside the mask (out of bounds = False)."""
    h, w = mask.shape
    x = np.floor(pixel_coords[:, 0]).astype(int)
    y = np.floor(pixel_coords[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros(len(pixel_coords), dtype=bool)
    out[ok] = mask[y[ok], x[ok]]
    return out
