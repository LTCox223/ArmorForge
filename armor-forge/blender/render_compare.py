"""Render several armor sets on the mannequin with the same calibrated cameras (headless, read-only).

    blender -b reference/Embermine_Vanguard_Fitted.blend --factory-startup -P blender/render_compare.py -- \
        --colors input/Part_Colors.json --out out/blender --obj v2=out/armor_v2.obj --obj v3=out/armor_v3.obj

The opened .blend is never saved. Its own armor (objects named "NN ...") is rendered as the set named by
--native (default "embermine"). Each --obj set is imported (Y-up OBJ, one object per piece named "NN_...").
Every piece is coloured by its part ID from Part_Colors.json; the mannequin is the underlayer colour.
Writes <out>/<set>_<camera>.png and prints JSON>> with triangle counts per set.
"""
import argparse
import json
import os
import re
import sys

import bpy

CAMERAS = ("Front calibrated", "Back calibrated", "Side calibrated", "Three-quarter front", "Three-quarter rear")
BODY = ("GunHound Fitting Body", "SkullTarget")       # shown under every set (head/feet replaced by gear)
PID = re.compile(r"^(\d\d)[ _]")


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obj", action="append", default=[], help="name=path.obj")
    ap.add_argument("--native", default="embermine")
    ap.add_argument("--res", type=int, default=900)
    return ap.parse_args(argv)


def material(pid, rgb, cache={}):
    if pid not in cache:
        m = bpy.data.materials.new(f"AF_part_{pid:02d}")
        m.diffuse_color = (*[c / 255 for c in rgb], 1.0)
        cache[pid] = m
    return cache[pid]


def paint(obj, pid, rgb):
    obj.data.materials.clear()
    obj.data.materials.append(material(pid, rgb))


def flatten_shading(obj):
    """Flat-shade every set so the comparison is geometry against geometry (no smoothing tricks)."""
    obj.data = obj.data.copy()
    obj.data.polygons.foreach_set("use_smooth", [False] * len(obj.data.polygons))
    for md in list(obj.modifiers):
        if md.type in {"SUBSURF", "MULTIRES"} or md.name.startswith("Smooth by Angle"):
            obj.modifiers.remove(md)


def tris(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    for o in objs:
        m = o.evaluated_get(dg).to_mesh()
        n += sum(len(p.vertices) - 2 for p in m.polygons)
        o.evaluated_get(dg).to_mesh_clear()
    return n


def main():
    a = args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(a.colors))
    rgb = {p["id"]: p["rgb"] for p in meta["parts"]}
    # the scene that holds the calibrated cameras
    scene = next(s for s in bpy.data.scenes if any(o.name == CAMERAS[0] for o in s.objects))
    bpy.context.window_manager.windows[0].scene = scene if bpy.context.window_manager.windows else None
    bpy.context.window.scene = scene if bpy.context.window else None

    sets = {a.native: [o for o in scene.objects if o.type == "MESH" and PID.match(o.name)]}
    for spec in a.obj:
        name, path = spec.split("=", 1)
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=os.path.abspath(path), forward_axis="NEGATIVE_Z", up_axis="Y")
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        for o in new:
            if o.name not in scene.objects:
                scene.collection.objects.link(o)
        sets[name] = new
    for objs in sets.values():
        for o in objs:
            flatten_shading(o)
            m = PID.match(o.name)
            if m:
                paint(o, int(m.group(1)), rgb.get(int(m.group(1)), (200, 200, 200)))
    for name in BODY:
        o = scene.objects.get(name)
        if o is not None:
            flatten_shading(o)              # copies the mesh first: the shared data is never touched
            paint(o, 17, rgb[17])
    body_objs = [scene.objects[n] for n in BODY if n in scene.objects]

    r = scene.render
    r.engine = "BLENDER_WORKBENCH"
    r.resolution_x = r.resolution_y = a.res
    r.resolution_percentage = 100
    r.film_transparent = False
    sh = scene.display.shading
    sh.light, sh.color_type = "STUDIO", "MATERIAL"
    sh.show_object_outline, sh.object_outline_color = True, (0.08, 0.08, 0.08)
    sh.show_cavity, sh.cavity_type = True, "WORLD"
    sh.show_backface_culling = False

    info = {"sets": {}, "cameras": {}}
    for set_name, objs in sets.items():
        info["sets"][set_name] = {"pieces": len(objs), "tris": tris(objs)}
        for other, oo in sets.items():
            for o in oo:
                o.hide_render = other != set_name
        for o in scene.objects:
            if o.type == "MESH" and o not in body_objs and not any(o in oo for oo in sets.values()):
                o.hide_render = True        # spare mannequin pieces (armless torso, old feet...)
        for o in body_objs:
            o.hide_render = False
        for cam in CAMERAS:
            c = scene.objects.get(cam)
            if c is None:
                continue
            info["cameras"][cam] = {"type": c.data.type, "ortho_scale": round(c.data.ortho_scale, 3)}
            scene.camera = c
            r.filepath = os.path.join(os.path.abspath(a.out), f"{set_name}_{cam.replace(' ', '_')}.png")
            bpy.ops.render.render(write_still=True)
    print("JSON>>" + json.dumps(info))


main()
