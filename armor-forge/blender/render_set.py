"""Render a built set (one collection) on the mannequin from fixed cameras, Workbench, read-only.

    blender -b out/step1/embermine_shells.blend -P armor-forge/blender/render_set.py -- \
        --collection "ArmorForge · Embermine" --out out/step1/renders --prefix shells

Shows the collection plus the mannequin objects given by --body (mirrored). Hides everything else.
--wire adds a dark wireframe overlay so topology is visible.
"""
import argparse
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

CAMS = {  # name: (location, look-at)
    "front": ((0.0, -6.0, 1.05), (0.0, 0.0, 1.05)),
    "three_quarter": ((3.3, -5.2, 1.6), (0.0, 0.0, 1.0)),
    "side": ((6.0, 0.0, 1.05), (0.0, 0.0, 1.05)),
    "back": ((0.0, 6.0, 1.05), (0.0, 0.0, 1.05)),
}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", nargs="+", required=True)
    ap.add_argument("--body", nargs="*", default=["GunHound Fitting Body", "GunHound Fitting Body.003",
                                                  "SkullTarget"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="set")
    ap.add_argument("--res", type=int, default=900)
    ap.add_argument("--wire", action="store_true")
    ap.add_argument("--lens", type=float, default=60.0)
    return ap.parse_args(argv)


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def main():
    a = args()
    sc = bpy.data.scenes.new("AF render")
    keep = []
    for cname in a.collection:
        c = bpy.data.collections[cname]
        sc.collection.children.link(c)
        keep += list(c.all_objects)
    grey = bpy.data.materials.new("AF body")
    grey.diffuse_color = (0.55, 0.55, 0.58, 1)
    for n in a.body:
        o = bpy.data.objects[n].copy()
        o.data = o.data.copy()
        o.data.materials.clear()
        o.data.materials.append(grey)
        for m in list(o.modifiers):
            if m.type == "ARMATURE":
                o.modifiers.remove(m)
        o.parent = None
        sc.collection.objects.link(o)
    if a.wire:
        black = bpy.data.materials.new("AF wire")
        black.diffuse_color = (0.03, 0.03, 0.03, 1)
        for o in keep:
            if o.type != "MESH":
                continue
            w = o.copy()
            w.data = o.data.copy()
            w.data.materials.clear()
            w.data.materials.append(black)
            for m in list(w.modifiers):
                if m.type == "SOLIDIFY":
                    w.modifiers.remove(m)
            wm = w.modifiers.new("wire", "WIREFRAME")
            wm.thickness = 0.002
            wm.use_even_offset = False
            sc.collection.objects.link(w)
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_cavity = True
    sh.cavity_type = "WORLD"
    sc.display.shading.show_object_outline = True
    sc.render.resolution_x = a.res
    sc.render.resolution_y = int(a.res * 1.25)
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("AF world")
    sc.world.color = (0.16, 0.16, 0.17)
    cam_data = bpy.data.cameras.new("AF cam")
    cam_data.lens = a.lens
    cam = bpy.data.objects.new("AF cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    os.makedirs(a.out, exist_ok=True)
    for name, (loc, tgt) in CAMS.items():
        cam.location = loc
        look_at(cam, tgt)
        sc.render.filepath = os.path.join(a.out, f"{a.prefix}_{name}.png")
        bpy.ops.render.render(write_still=True, scene=sc.name)
        print("saved", sc.render.filepath)


main()
