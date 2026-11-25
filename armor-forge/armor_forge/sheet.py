"""Read the four-view orthographic reference sheets into metric silhouette masks.

Sheet layout (both the mannequin template and the generated armor sheets):
    Front | Back
    Side  | Side without arms
Each panel plots 2 m x 2 m: horizontal axis -1..+1 m, vertical axis z = 0..2 m. The printed
"px per m" caption is just text; real scale is taken from the plot rectangle, expressed here as
fractions of the image size (measured on sheets produced from the same template).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

# Plot rectangles (x -1..+1 m, z 0..2 m) as fractions of image width / height.
LAYOUT = {
    "cols": {"L": (0.0675, 0.4394), "R": (0.5602, 0.9318)},
    "rows": {"T": (0.1213, 0.4824), "B": (0.5927, 0.9538)},
}
PANELS = {"front": ("L", "T"), "back": ("R", "T"), "side": ("L", "B"), "side_noarm": ("R", "B")}


@dataclass
class ViewMask:
    """Boolean silhouette plus the mapping from panel meters to array pixels."""
    arr: np.ndarray      # (H, W) bool, row 0 = top
    col0: float          # column of u = -1 m
    row0: float          # row of z = 2 m
    pxu: float           # columns per meter
    pxv: float           # rows per meter

    def sample(self, u, z):
        """Silhouette lookup at panel coordinates (u across, z up), meters. Outside = False."""
        u, z = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(z, dtype=float))
        c = np.floor(self.col0 + (u + 1.0) * self.pxu).astype(int)
        r = np.floor(self.row0 + (2.0 - z) * self.pxv).astype(int)
        h, w = self.arr.shape
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.zeros(u.shape, dtype=bool)
        out[ok] = self.arr[r[ok], c[ok]]
        return out


def figure_mask(rgb, thresh=185, open_px=5):
    """Dark-ish figure pixels on the light panel: threshold, drop thin lines, keep the largest blob."""
    m = rgb.min(axis=2) < thresh
    m = ndi.binary_opening(m, structure=np.ones((open_px, open_px)))
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(m)


def load_sheet(path, layout=LAYOUT, thresh=185, z_extra=0.06):
    """Return {panel_name: ViewMask}. z_extra lets silhouettes run a little above the 2 m line."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(int)
    h, w, _ = img.shape
    out = {}
    for name, (cc, rr) in PANELS.items():
        c0, c1 = (round(f * w) for f in layout["cols"][cc])
        r0, r1 = (round(f * h) for f in layout["rows"][rr])
        pxu = (c1 - c0) / 2.0
        pxv = (r1 - r0) / 2.0
        top = max(0, r0 - int(round(z_extra * pxv)))
        crop = img[top:r1 + 1, c0:c1 + 1]
        out[name] = ViewMask(figure_mask(crop, thresh), col0=0.0, row0=float(r0 - top),
                             pxu=pxu, pxv=pxv)
    return out
