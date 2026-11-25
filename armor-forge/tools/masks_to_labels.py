#!/usr/bin/env python3
"""Convert colour-ID part masks (RGBA PNGs) into the 1/4-res part-ID grids fit_set.py reads.

    python tools/masks_to_labels.py --masks input/Masks --colors input/Part_Colors.json --out input

Writes <View>_labels.npy (uint8, 0 = background, else part id) for each view in fit.VIEWS.
Downsampling is nearest-neighbour (as Part_Colors.json requires) so IDs never blend.
Pixels whose colour is not in the palette are reported and left as background.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.fit import VIEWS  # noqa: E402


def palette(meta):
    pal = {}
    for p in meta["parts"]:
        rgb = p["rgb"]
        rgb = tuple(int(c) for c in (rgb.split() if isinstance(rgb, str) else rgb))
        pal[rgb] = int(p["id"])
    return pal


def to_labels(img, pal, alpha_min=128, factor=4):
    a = np.asarray(img.convert("RGBA"))
    key = (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]
    lab = np.zeros(key.shape, np.uint8)
    known = np.zeros(key.shape, bool)
    for (r, g, b), pid in pal.items():
        m = key == ((r << 16) | (g << 8) | b)
        lab[m] = pid
        known |= m
    fg = a[..., 3] >= alpha_min
    lab[~fg] = 0
    unknown = int((fg & ~known).sum())
    # nearest-neighbour: take the pixel at the centre of each factor x factor block
    o = factor // 2
    return lab[o::factor, o::factor], unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=4)
    a = ap.parse_args()
    pal = palette(json.load(open(a.colors)))
    os.makedirs(a.out, exist_ok=True)
    for v in VIEWS:
        lab, unknown = to_labels(Image.open(os.path.join(a.masks, f"{v}.png")), pal, factor=a.factor)
        np.save(os.path.join(a.out, f"{v}_labels.npy"), lab)
        ids = np.unique(lab[lab > 0]).tolist()
        print(f"{v:18s} {lab.shape} parts {ids}" + (f"  ({unknown} off-palette px ignored)" if unknown else ""))


if __name__ == "__main__":
    main()
