"""Trace a piece's outline from the concept sheet (masks only as a guide); writes masks in the same frame.

    python tools/trace_concept.py --art "reference/Embermine Heavy Armor Turnaround.png" \
        --masks input/Masks --cams armor-forge/cams.json --out out/helmet/traced [--rgb 244,93,89]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet as hm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True)
ap.add_argument("--masks", required=True)
ap.add_argument("--cams", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rgb", default=",".join(map(str, hm.HELMET_RGB)))
ap.add_argument("--reach", type=float, default=0.03, help="how far (m) the outline may move off the mask")
a = ap.parse_args()
rgb = tuple(int(v) for v in a.rgb.split(","))
for v, (m, reg, info) in cc.trace_views(a.art, a.masks, a.cams, rgb, out_dir=a.out, reach_m=a.reach).items():
    old = hm.load_mask(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png"), rgb)
    s = cc.edge_offsets(old, m) * 1000
    print(f"{v:5s}: figure overlap {reg.iou:.3f}, outline moved median {np.median(s):+.1f} mm "
          f"(5-95%: {np.percentile(s, 5):+.1f} .. {np.percentile(s, 95):+.1f} mm)")
print("wrote", a.out)
