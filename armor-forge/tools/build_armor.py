#!/usr/bin/env python3
"""Build separate low-poly armor pieces from the labelled base mesh.

    python tools/build_armor.py armor_sheet.png --body-obj mannequin_yup.obj --labels label_out/face_parts_subdiv.npy --out armor_out
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge.inflate import inflate, subdivide, vertex_normals, weld  # noqa: E402
from armor_forge.labels import NAMES, PALETTE  # noqa: E402
from armor_forge.meshio import read_obj  # noqa: E402
from armor_forge.preview import render  # noqa: E402
from armor_forge.shells import build_shells, split_lr, write_obj_multi  # noqa: E402
from armor_forge.sheet import load_sheet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armor_sheet")
    ap.add_argument("--body-obj", required=True)
    ap.add_argument("--body-obj-zup", action="store_true")
    ap.add_argument("--labels", required=True, help="face_parts_subdiv.npy from label_parts.py")
    ap.add_argument("--subdivide", type=int, default=1)
    ap.add_argument("--out", default="armor_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    v, f = read_obj(a.body_obj, y_up=not a.body_obj_zup)
    v, f, _ = weld(v, f)
    for _ in range(a.subdivide):
        v, f = subdivide(v, f)
    f = np.array(f)
    nv, t = inflate(v, f, load_sheet(a.armor_sheet))
    lab = np.load(a.labels)
    assert len(lab) == len(f), "labels do not match this mesh (same --subdivide as label_parts?)"
    nrm = vertex_normals(v, f)
    parts = split_lr(build_shells(v, f, nrm, t, lab))
    tris = 0
    for name, (pv, pq) in parts.items():
        tris += 2 * len(pq)
        print(f"  {name:22s} {len(pq):5d} quads")
    print(f"total: {len(parts)} pieces, {sum(len(q) for _, q in parts.values())} quads, ~{tris} triangles")
    write_obj_multi(os.path.join(a.out, "armor_pieces.obj"), parts)

    # preview: all pieces, flat shaded, coloured by part
    V, Q, C = [], [], []
    for name, (pv, pq) in parts.items():
        base = sum(len(x) for x in V)
        V.append(pv)
        Q.append(pq + base)
        pn = name.rsplit(".", 1)[0]
        C += [PALETTE[NAMES.index(pn)]] * len(pq)
    V, Q, C = np.vstack(V), np.vstack(Q), np.array(C)
    imgs = [render(V, Q, ang, el, face_colors=C, smooth=False) for ang, el in ((-35, 12), (145, 12), (90, 0))]
    sheet = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 0))
        x += i.width
    sheet.save(os.path.join(a.out, "preview.png"))
    imgs = [render(V, Q, ang, el, smooth=False) for ang, el in ((-35, 12), (145, 12), (90, 0))]
    sheet = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 0))
        x += i.width
    sheet.save(os.path.join(a.out, "preview_plain.png"))


if __name__ == "__main__":
    main()
