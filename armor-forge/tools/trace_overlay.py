"""Draw mask (blue) and traced (red) outlines of a piece over the concept, per view.

    python tools/trace_overlay.py --art ref.png --masks input/Masks --traced out/helmet/traced \
        --cams cams.json --out out/helmet/trace_overlay.png [--extra name=dir:colour ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet as hm  # noqa: E402


def outline_points(mask, reg):
    e = mask & ~ndi.binary_erosion(mask)
    r, c = np.nonzero(e)
    return reg.to_art(c + 0.5, r + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--traced", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=600)
    ap.add_argument("--half", type=float, default=0.20, help="half size of the crop (m)")
    ap.add_argument("--centre-z", type=float, default=1.86)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.traced, "trace.json")))
    art = Image.open(a.art).convert("RGB")
    tiles = []
    for v in ("front", "side", "back"):
        reg = cc.Registration(**meta[v]["registration"])
        ppm = reg.scale * 1600.0
        old = hm.load_mask(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png"))
        new = hm.load_mask(os.path.join(a.traced, f"{cc.MASK_FILES[v]}.png"))
        cams = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cams.json")))
        cx, cy = reg.to_art(cams[v][0], cams[v][1] - a.centre_z * 1600)
        if v == "side":
            cx -= 0.03 * ppm
        h = a.half * ppm
        box = (cx - h, cy - h, cx + h, cy + h)
        k = a.tile / (2 * h)
        crop = art.transform((a.tile, a.tile), Image.EXTENT, box, Image.BICUBIC)
        d = ImageDraw.Draw(crop)
        for m, col in ((old, (40, 90, 255)), (new, (235, 30, 30))):
            x, y = outline_points(m, reg)
            for X, Y in zip((x - box[0]) * k, (y - box[1]) * k):
                d.point((X, Y), fill=col)
        d.text((8, 8), f"{v}: blue = AI mask, red = traced from the art", fill=(0, 0, 0))
        tiles.append(crop)
    W = Image.new("RGB", (a.tile * len(tiles), a.tile), "white")
    for i, t in enumerate(tiles):
        W.paste(t, (i * a.tile, 0))
    W.save(a.out)


main()
