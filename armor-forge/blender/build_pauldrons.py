"""Build the pauldrons as trimmed plates (cap + lame, left and right) fitted to the concept.

    blender -b -P armor-forge/blender/build_pauldrons.py -- --masks input/Masks \
        --cams armor-forge/cams.json --art "reference/Embermine Heavy Armor Turnaround.png" \
        --trace-meta out/helmet/traced/trace.json --mannequin armor-forge/mannequin.json --out out/pauldrons

Each piece (by its colour in the piece map) is a four-sided plate fitted to the piece's outline in
the front, back and side views and to its gold trim (armor_forge/plate.py), then built as flat panels
split at the creases, a raised rim and (on the cap) a lip on the neck edge. Writes
<out>/pauldrons.blend (collection "Pauldrons"; "Reference" holds the torso), renders,
concept_vs_pauldrons.png (fitted edge loop in red and creases in blue on the concept) and
pauldrons.json (fit scores and the plates).
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np
import scipy.ndimage as ndi
from mathutils import Vector
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import build_helmet as bh  # noqa: E402
from armor_forge import concept as cc  # noqa: E402
from armor_forge import helmet_parts as hp  # noqa: E402
from armor_forge import plate as pl  # noqa: E402

PIECES = [  # name, piece-map colour, layout
    ("Pauldron cap", (137, 87, 212), dict(lip=0.012)),
    ("Pauldron lame 1", (220, 157, 242), dict(creases_u=(), lip=0.0)),
]
COLOURS = {"panel": (0.33, 0.36, 0.40), "rim": (0.62, 0.48, 0.25), "lip": (0.62, 0.48, 0.25)}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--trace-meta", required=True, help="registrations (concept.trace_views output)")
    ap.add_argument("--mannequin", default=None, help="mannequin.json, for a reference torso")
    ap.add_argument("--thickness", type=float, default=0.004)
    ap.add_argument("--res", type=int, default=700)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side-turn", type=float, default=None,
                    help="side view's drawn turn (deg); default: sheet.json next to --trace-meta, else measured")
    ap.add_argument("--measure-turn", action="store_true", help="measure the turn on the cap and save it")
    ap.add_argument("--asymmetric", action="store_true", help="don't force front/back symmetry")
    return ap.parse_args(argv)


def colour_mask(path, rgb):
    a = np.asarray(Image.open(path).convert("RGBA"))
    return (a[..., 0] == rgb[0]) & (a[..., 1] == rgb[1]) & (a[..., 2] == rgb[2]) & (a[..., 3] > 128)


def gold_in_frame(art_rgb, reg, mask, pad=40):
    """Trim pixels (gold) in the artboard frame, around the piece."""
    r, c = np.nonzero(mask)
    r0, r1 = max(0, r.min() - pad), min(mask.shape[0], r.max() + pad)
    c0, c1 = max(0, c.min() - pad), min(mask.shape[1], c.max() + pad)
    rr, cc_ = np.mgrid[r0:r1, c0:c1].astype(float)
    ax, ay = reg.to_art(cc_ + .5, rr + .5)
    g = ndi.map_coordinates(hp.gold_score(art_rgb), [ay - .5, ax - .5], order=1) > 0.5
    out = np.zeros_like(mask)
    out[r0:r1, c0:c1] = g
    return out


def make_object(name, mesh, coll, thickness):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in mesh.verts], [], mesh.faces)
    me.validate()
    for key in ("panel", "rim", "lip"):
        m = bpy.data.materials.get(f"Plate {key}") or bpy.data.materials.new(f"Plate {key}")
        m.diffuse_color = (*COLOURS[key], 1)
        me.materials.append(m)
    idx = {"panel": 0, "rim": 1, "lip": 2}
    for poly, g in zip(me.polygons, mesh.groups):
        poly.material_index = idx[g]
        poly.use_smooth = False
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    for a, b in mesh.sharp:
        e = bm.edges.get((bm.verts[a], bm.verts[b]))
        if e is not None:
            e.smooth = False
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if thickness:
        so = ob.modifiers.new("Solidify", "SOLIDIFY")
        so.thickness = thickness
        so.offset = -1
    return ob


def load_torso(path):
    d = json.load(open(path)).get("GunHound Fitting Body.001")
    if d is None:
        return None
    V = np.array(d["verts"])
    F = d["faces"]
    if d.get("mirror"):
        n = len(V)
        V = np.vstack([V, V * [-1, 1, 1]])
        F = F + [[i + n for i in reversed(f)] for f in F]
    return V, F


def overlay(art, reg, cam, view, plates, centre_u, centre_z, size=500, half=0.2, turn=0.0):
    ppm = reg.scale * 1600
    cx, cy = reg.to_art(cam[0] + centre_u * 1600, cam[1] - centre_z * 1600)
    h = half * ppm
    k = size / (2 * h)
    crop = art.transform((size, size), Image.EXTENT, (cx - h, cy - h, cx + h, cy + h), Image.BICUBIC)
    d = ImageDraw.Draw(crop)

    def P(Q):
        u, z = pl.project(Q, view, turn, centre[0])
        x, y = reg.to_art(cam[0] + u * 1600, cam[1] - z * 1600)
        return [((a - cx + h) * k, (b - cy + h) * k) for a, b in zip(x, y)]
    centre = [None]
    for plate, layout in plates:
        centre[0] = plate.centre()
        Sm = plate.surface(np.linspace(0, 1, 41), np.linspace(0, 1, 41))
        for u in layout.creases_u:
            d.line(P(Sm[int(round(u * 40)), :]), fill=(30, 110, 255), width=2)
        for v in layout.creases_v:
            d.line(P(Sm[:, int(round(v * 40))]), fill=(30, 110, 255), width=2)
        loop = pl._loop(Sm)
        d.line(P(np.vstack([loop, loop[:1]])), fill=(230, 20, 20), width=2)
    return crop


def main():
    a = args()
    cams = json.load(open(a.cams))
    meta = json.load(open(a.trace_meta))
    regs = {v: cc.Registration(**meta[v]["registration"]) for v in ("front", "back", "side")}
    art_rgb = cc.load_rgb(a.art)
    sheet_path = os.path.join(os.path.dirname(os.path.abspath(a.trace_meta)), "sheet.json")
    sheet = json.load(open(sheet_path)) if os.path.exists(sheet_path) else {}
    if a.side_turn is not None:
        turn, turn_source = np.radians(a.side_turn), "given"
    elif "side_turn_deg" in sheet and not a.measure_turn:
        turn, turn_source = np.radians(sheet["side_turn_deg"]), "sheet.json"
    else:
        turn, turn_source = 0.0, "measure"
    built, report = [], {}
    prev_masks = None
    for name, rgb, lay in PIECES:
        full = {v: colour_mask(os.path.join(a.masks, f"{cc.MASK_FILES[v]}.png"), rgb) for v in ("front", "back", "side")}
        if not full["front"].any():
            continue
        above = prev_masks if built else None          # the pieces above may cover this one
        views = [pl.make_view(v, full[v], cams[v], gold_in_frame(art_rgb, regs[v], full[v]),
                              in_front=None if above is None else above[v]) for v in full]
        prev_masks = full if not built else {v: prev_masks[v] | full[v] for v in full}
        if built:      # a lame: start from the piece above, continued down the arm
            init = pl.continue_from(built[-1][1])
        else:
            init = pl.init_from_views(full["front"], full["back"], full["side"], cams)
        for vw in views:
            if vw.name == "side":
                vw.turn = turn
        measure = not built and turn_source == "measure"
        plate, t_fit = pl.fit(views, init, symmetric=not a.asymmetric, fit_turn=measure)
        if measure:
            turn = t_fit
            print(f"armor-forge: side view drawn turned {np.degrees(turn):+.1f} deg (measured on the {name.lower()})")
        _, detail = pl.score(plate, views, detail=True)
        layout = pl.PanelLayout(**lay)
        mesh = pl.build_mesh(plate, layout)
        print(f"armor-forge: {name}: " + ", ".join(f"{v} IoU {d['iou']:.3f} trim {d['trim_mm']:.1f} mm"
                                                     for v, d in detail.items() if isinstance(d, dict))
              + f"; {len(mesh.verts)} verts, {len(mesh.faces)} faces")
        built.append((name, plate, layout, mesh))
        report[name] = dict(scores=detail, plate=plate.vector().reshape(8, 3).tolist())

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    if a.mannequin:
        torso = load_torso(a.mannequin)
        if torso is not None:
            ref = bpy.data.collections.new("Reference")
            sc.collection.children.link(ref)
            me = bpy.data.meshes.new("Torso")
            me.from_pydata([tuple(v) for v in torso[0]], [], torso[1])
            mt = bpy.data.materials.new("Torso")
            mt.diffuse_color = (0.75, 0.72, 0.68, 1)
            me.materials.append(mt)
            ob = bpy.data.objects.new("Torso", me)
            ref.objects.link(ob)
    coll = bpy.data.collections.new("Pauldrons")
    sc.collection.children.link(coll)
    for name, plate, layout, mesh in built:
        make_object(f"{name} L", mesh, coll, a.thickness)
        make_object(f"{name} R", pl.mirror(mesh), coll, a.thickness)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "pauldrons.blend")))
    report["side_turn_deg"] = float(np.degrees(turn))
    report["side_turn_source"] = turn_source
    json.dump(report, open(os.path.join(a.out, "pauldrons.json"), "w"), indent=1)
    if turn_source == "measure":
        sheet["side_turn_deg"] = float(np.degrees(turn))
        sheet["side_turn_note"] = "measured on the pauldron cap (build_pauldrons.py)"
        json.dump(sheet, open(sheet_path, "w"), indent=1)
        print("armor-forge: saved the side view's turn to", sheet_path)

    cam = bh.setup_render(a.res)
    cam.data.ortho_scale = 0.45
    centre = Vector((0.28, 0.03, 1.64))
    for view in ("front", "side", "back", "three_quarter"):
        bh.render_view(cam, view, os.path.join(a.out, f"pauldron_{view}.png"), centre=centre)
    art = Image.open(a.art).convert("RGB")
    T = 450
    W = Image.new("RGB", (T * 4, T * 2), "white")
    plates = [(p_, l_) for _, p_, l_, _ in built]
    for i, (v, u) in enumerate((("front", 0.28), ("side", 0.03), ("back", -0.28))):
        W.paste(overlay(art, regs[v], cams[v], v, plates, u, 1.64, T, turn=turn), (i * T, 0))
    for i, v in enumerate(("front", "side", "back", "three_quarter")):
        W.paste(Image.open(os.path.join(a.out, f"pauldron_{v}.png")).convert("RGB").resize((T, T)), (i * T, T))
    W.save(os.path.join(a.out, "concept_vs_pauldrons.png"))
    print("armor-forge: wrote", a.out)


if __name__ == "__main__":
    main()
