"""Helmet masks from a concept sheet with no piece masks and no side view (e.g. Thornwood).

    python tools/helmet_from_sheet.py --art "blender/Thornwood knight armor turnaround.png" \
        --front 10,880 --back 890,1760 --out out/thornwood/masks

Writes Front/Back/Side.png (Side is a stand-in built from the head target), cams.json, trace.json
and cut.png (the helmet outline and the cut line drawn on the concept). Then run build_helmet.py or
build_helmet_parts.py with --masks/--cams/--trace-meta pointing here.
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from armor_forge import concept as cc  # noqa: E402
from armor_forge import nomask as nm  # noqa: E402


def pair(s):
    a, b = s.split(",")
    return int(a), int(b)


ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True)
ap.add_argument("--front", type=pair, required=True, help="front panel column range")
ap.add_argument("--back", type=pair, required=True, help="back panel column range")
ap.add_argument("--bg-front", type=pair, default=None, help="figure-free columns for the grid (default: search)")
ap.add_argument("--bg-back", type=pair, default=None, help="figure-free columns for the grid (default: search)")
ap.add_argument("--grid-m", type=float, default=0.25, help="grid line spacing (m)")
ap.add_argument("--head-target", default=os.path.join(HERE, "head_target.json"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

rgb = cc.load_rgb(a.art)
fig = nm.figure_mask(rgb)
masks, regs, info = {}, {}, {}
for v, panel, bg in (("front", a.front, a.bg_front), ("back", a.back, a.bg_back)):
    cal = nm.calibrate_panel(rgb, *panel, bg, a.grid_m)
    cut, dbg = nm.head_cut(nm.panel_figure(fig, *panel), cal)
    masks[v] = nm.helmet_mask(rgb, cal, cut)
    regs[v] = cal.registration()
    info[v] = dict(ppm=cal.ppm, z0_row=cal.z0_row, cx=cal.cx, cut_z=(cal.z0_row - cut) / cal.ppm,
                   **{k: (cal.z0_row - r) / cal.ppm for k, r in dbg.items()})
    print(f"{v}: {cal.ppm:.1f} px/m, u=0 at column {cal.cx:.1f}, z=0 at row {cal.z0_row:.1f}; "
          f"helmet z {info[v]['cut_z']:.3f} .. {info[v]['top_row']:.3f} (widest {info[v]['widest_row']:.3f}, "
          f"shoulders {info[v]['shoulder_row']:.3f})")
masks["side"] = nm.synthetic_side(masks["front"], masks["back"], nm.load_head_target(a.head_target))
regs["side"] = regs["front"]          # no side panel: placeholder so tools that read trace.json work
nm.write_masks(a.out, masks, regs)
json.dump(info, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)

# check image: outline (red) and cut row (blue) on the concept, front and back
art = Image.open(a.art).convert("RGB")
tiles = []
for v in ("front", "back"):
    reg = regs[v]
    e = masks[v] & ~ndi.binary_erosion(masks[v])
    r, c = np.nonzero(e)
    x, y = reg.to_art(c + 0.5, r + 0.5)
    cx, cy = reg.to_art(nm.ORIGIN[0], nm.ORIGIN[1] - 1.9 * nm.PPM)
    h = 0.25 * reg.scale * nm.PPM
    T = 500
    k = T / (2 * h)
    crop = art.transform((T, T), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)
    d = ImageDraw.Draw(crop)
    for X, Y in zip((x - cx + h) * k, (y - cy + h) * k):
        d.point((X, Y), fill=(255, 0, 0))
    cut_y = (info[v]["z0_row"] - info[v]["cut_z"] * info[v]["ppm"] - cy + h) * k
    d.line((0, cut_y, T, cut_y), fill=(40, 90, 255))
    tiles.append(crop)
W = Image.new("RGB", (1000, 500))
W.paste(tiles[0], (0, 0))
W.paste(tiles[1], (500, 0))
W.save(os.path.join(a.out, "cut.png"))
print("wrote", a.out)
