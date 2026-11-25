#!/usr/bin/env python3
"""Carve a rough 3D armor volume from four-view reference sheets.

    python tools/carve_sheet.py armor_sheet.png mannequin_sheet.png --out carve_out

Writes <out>/armor_rough.obj (import into Blender with default OBJ settings) and
<out>/preview.png, and prints sanity checks.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.carve import (carve_envelope, carve_shell, make_grid, surface_nets,  # noqa: E402
                               visual_hull, write_obj)
from armor_forge.meshio import read_obj, sample_surface  # noqa: E402
from armor_forge.sheet import load_sheet  # noqa: E402


def render(verts, quads, az_deg, el_deg, size=(560, 940), pad=0.04, bg=(233, 238, 241)):
    az, el = np.radians(az_deg), np.radians(el_deg)
    c = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    r = np.cross(-c, [0, 0, 1.0]); r /= np.linalg.norm(r)
    up = np.cross(r, -c)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    facing = n @ c
    keep = facing > 0
    tris, p, n, facing = tris[keep], p[keep], n[keep], facing[keep]
    u, v, d = p @ r, p @ up, p @ c
    w, h = size
    s = (h * (1 - 2 * pad)) / 2.1
    x = w / 2 + u * s
    y = h * (1 - pad) - (v - 0.0) * s
    light = np.array([-0.4, -0.7, 0.6]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1) * 0.6 + 0.4 * facing * 0.5
    img = Image.new("RGB", size, bg)
    dr = ImageDraw.Draw(img)
    for i in np.argsort(d.mean(axis=1)):
        g = int(np.clip(shade[i], 0, 1) * 215)
        dr.polygon([(x[i, 0], y[i, 0]), (x[i, 1], y[i, 1]), (x[i, 2], y[i, 2])],
                   fill=(g, int(g * 1.03), int(g * 1.08)))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armor_sheet")
    ap.add_argument("mannequin_sheet")
    ap.add_argument("--out", default="carve_out")
    ap.add_argument("--voxel", type=float, default=0.01, help="voxel size in meters")
    ap.add_argument("--body-obj", help="mannequin mesh as OBJ, exported from Blender with default "
                    "settings and 'Apply Modifiers' on (meters, same pose as the sheets). "
                    "Strongly recommended: it keeps arms from turning into slabs.")
    ap.add_argument("--body-obj-zup", action="store_true",
                    help="the body OBJ is already Z-up (not Blender's default export)")
    ap.add_argument("--band", type=float, default=0.10, help="max armor distance from body, m")
    ap.add_argument("--gap", type=float, default=0.004, help="clearance above the body, m")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    armor_v = load_sheet(args.armor_sheet)
    body_v = load_sheet(args.mannequin_sheet)
    grid = make_grid(voxel=args.voxel)
    armor_hull = visual_hull(grid, armor_v)
    body_hull = visual_hull(grid, body_v)
    vox_vol = args.voxel ** 3
    print(f"grid {grid.shape}  armor hull {armor_hull.sum() * vox_vol:.3f} m3  "
          f"body hull {body_hull.sum() * vox_vol:.3f} m3")
    outside = (body_hull & ~armor_hull).sum() / body_hull.sum()
    print(f"body hull voxels outside armor hull: {100 * outside:.1f}%  (high = sheets misaligned)")

    if args.body_obj:
        bv, bf = read_obj(args.body_obj, y_up=not args.body_obj_zup)
        shell = carve_envelope(armor_hull, grid, sample_surface(bv, bf), band=args.band)
        print(f"envelope (body-band {args.band} m) volume {shell.sum() * vox_vol * 1000:.1f} litres")
    else:
        shell = carve_shell(armor_hull, body_hull, args.voxel, band=args.band, gap=args.gap)
        print(f"armor shell volume {shell.sum() * vox_vol * 1000:.1f} litres")
    xs, ys, zs = grid.axes()
    checks = {"front": (shell.any(axis=1), armor_v["front"].sample(xs[:, None], zs[None, :])),
              "back": (shell.any(axis=1), armor_v["back"].sample(-xs[:, None], zs[None, :])),
              "side (with arms)": (shell.any(axis=0), armor_v["side"].sample(ys[:, None], zs[None, :]))}
    for name, (proj, mask) in checks.items():
        print(f"  silhouette match, {name}: IoU {100 * (proj & mask).sum() / (proj | mask).sum():.1f}%")
    verts, quads = surface_nets(shell, grid)
    print(f"mesh: {len(verts)} verts, {len(quads)} quads  ({time.time() - t0:.1f}s)")
    write_obj(os.path.join(args.out, "armor_rough.obj"), verts, quads)

    views = [render(verts, quads, a, e) for a, e in ((-35, 12), (145, 12), (90, 0))]
    sheet = Image.new("RGB", (sum(v.width for v in views), views[0].height))
    x = 0
    for v in views:
        sheet.paste(v, (x, 0)); x += v.width
    sheet.save(os.path.join(args.out, "preview.png"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
