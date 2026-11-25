#!/usr/bin/env python3
"""Inflate the base mesh to match armor silhouettes, keeping its topology and curves.

    python tools/inflate_sheet.py armor_sheet.png --body-obj mannequin.obj --out inflate_out

Writes <out>/armor_inflated.obj (same faces as the body, vertices pushed outward) and preview.png.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.carve import write_obj  # noqa: E402
from armor_forge.inflate import inflate, silhouette_iou, subdivide, weld  # noqa: E402
from armor_forge.meshio import read_obj  # noqa: E402
from armor_forge.preview import contact_sheet  # noqa: E402
from armor_forge.sheet import load_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armor_sheet")
    ap.add_argument("--body-obj", required=True)
    ap.add_argument("--body-obj-zup", action="store_true")
    ap.add_argument("--out", default="inflate_out")
    ap.add_argument("--t-max", type=float, default=0.12, help="max push, m")
    ap.add_argument("--arm-t-max", type=float, default=0.06, help="max push on the arms, m")
    ap.add_argument("--gap", type=float, default=0.004)
    ap.add_argument("--smooth", type=int, default=3, help="smoothing passes on the push field")
    ap.add_argument("--subdivide", type=int, default=1,
                    help="linear quad subdivisions before inflating (more points to follow the outline)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    views = load_sheet(args.armor_sheet)
    v, f = read_obj(args.body_obj, y_up=not args.body_obj_zup)
    v, f, _ = weld(v, f)
    print(f"body: {len(v)} verts, {len(f)} faces after welding the seam")
    for _ in range(args.subdivide):
        v, f = subdivide(v, f)
    if args.subdivide:
        print(f"subdivided: {len(v)} verts, {len(f)} faces")
    nv, t = inflate(v, f, views, gap=args.gap, smooth_iters=args.smooth,
                    t_max=args.t_max, arm_t_max=args.arm_t_max)
    print(f"push distance: median {np.median(t[t > 0]) * 100:.1f} cm, "
          f"max {t.max() * 100:.1f} cm, {int((t > 0).sum())} of {len(t)} verts moved")
    before = silhouette_iou(v, f, views)
    after = silhouette_iou(nv, f, views)
    for k in after:
        print(f"  silhouette match {k:5s}: body {100 * before[k]:.1f}%  ->  inflated {100 * after[k]:.1f}%")
    quads = np.array([q for q in f if len(q) == 4])
    write_obj(os.path.join(args.out, "armor_inflated.obj"), nv, np.array(f))
    contact_sheet(nv, quads).save(os.path.join(args.out, "preview.png"))
    contact_sheet(v, quads).save(os.path.join(args.out, "preview_body.png"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
