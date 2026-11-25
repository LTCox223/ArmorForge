"""Build the concept's helmet (shape only) and render it next to the concept art.

    blender -b -P armor-forge/blender/build_helmet.py -- \
        --masks input/Masks --cams armor-forge/cams.json --out out/helmet

Writes <out>/helmet.blend and renders <out>/helmet_<view>.png (front, side, back, three-quarter).
Parameters (see armor_forge/helmet.py HelmetParams) can be overridden: --param beak=0.3 n_visor=3.5
"""
import argparse
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from armor_forge import helmet as hm  # noqa: E402


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", nargs="*", default=[])
    ap.add_argument("--subd", type=int, default=1, help="Subdivision levels shown in the renders")
    ap.add_argument("--res", type=int, default=700)
    return ap.parse_args(argv)


def make_object(verts, faces, info, subd=1, name="Helmet", thickness=0.004):
    """Mesh object with crease edges, Solidify (inward) and Subdivision, linked to the scene."""
    import bmesh
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    me.shade_smooth()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    cl = bm.edges.layers.float.get("crease_edge") or bm.edges.layers.float.new("crease_edge")
    for i, j in info["crease"]:
        e = bm.edges.get((bm.verts[i], bm.verts[j]))
        if e is not None:
            e[cl] = 1.0
    bm.to_mesh(me)
    bm.free()
    mat = bpy.data.materials.get("Helmet steel") or bpy.data.materials.new("Helmet steel")
    mat.diffuse_color = (0.33, 0.36, 0.40, 1)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    so = ob.modifiers.new("Solidify", "SOLIDIFY")
    so.thickness = thickness
    so.offset = -1
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = subd
    return ob


VIEWS = {"front": (0, -3, 0), "side": (3, 0, 0), "back": (0, 3, 0), "three_quarter": (-2.0, -2.2, 0.9),
         "top": (0, 0, 3)}
CENTRE = Vector((0, -0.03, 1.90))


def setup_render(res):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_specular_highlight = True
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.93, 0.94, 0.96)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 0.42
    return cam


def render_view(cam, view, path, centre=CENTRE):
    sc = bpy.context.scene
    cam.location = centre + Vector(VIEWS[view])
    if view == "top":   # looking down, front of the helmet at the bottom of the image
        cam.rotation_euler = (0, 0, 0)
    else:
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)


def load(a):
    o = hm.load_outlines(a.masks, a.cams)
    p = hm.HelmetParams()
    for kv in a.param:
        k, v = kv.split("=")
        setattr(p, k, type(getattr(p, k))(v))
    return o, p


def main():
    a = args()
    o, p = load(a)
    verts, faces, info = hm.build(o, p)
    print(f"armor-forge: helmet {len(verts)} verts, {len(faces)} faces, brow at z={info['brow']:.3f}, "
          f"{info['rings']} rings x {info['K']}")
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    make_object(verts, faces, info, a.subd)
    os.makedirs(a.out, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out, "helmet.blend")))
    cam = setup_render(a.res)
    for name in ("front", "side", "back", "three_quarter"):
        render_view(cam, name, os.path.join(a.out, f"helmet_{name}.png"))
    print("armor-forge: wrote", a.out)


if __name__ == "__main__":
    main()
