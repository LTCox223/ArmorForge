# armor-forge

Turn concept art of armor on a mannequin into fitted, pre-rigged armor shells in Blender.

The idea: armor on a character is a shell around a body you already have. So instead of
reconstructing 3D from a picture, we take the picture's *outline* of each armor piece, find the
body surface underneath it, lift that surface off the skin, and give it thickness. The result
already shares the body's skin weights and rig. You then refine it by hand.

## Body-shell set from a piece map (current approach, step 1)

The AI makes the art; the scripts make the pieces. Input is a flat-colour piece map (four views,
one colour per piece, like `input/Masks` + `input/Part_Colors.json`). Every piece is built from a
clean mesh and only its vertices move, so no new topology is invented:

| Piece kind | Built from | Pieces (Embermine) |
|---|---|---|
| `shell` | the mannequin's own faces under the drawing, hem snapped to the outline, lifted along normals to the outline (visual hull) | plates, sleeves, belt, gauntlets, boots |
| `hanging` | a quad grid following the outline row by row, one depth per row clear of what is underneath | tabard |
| `base` | the whole body except the sections a piece replaces (head, hands, feet) | undersuit |

    python tools/masks_to_labels.py --masks ../input/Masks --colors ../input/Part_Colors.json --out ../input
    blender -b ../blender/Embermine_Modular_Geometry.blend -P blender/build_shell_set.py -- \
        --labels ../input --cams cams.json --colors ../input/Part_Colors.json \
        --out ../out/step1/embermine_shells.blend

Pieces land in a new collection `ArmorForge · <name>` (half body with Mirror + Solidify modifiers, a
flat material per piece). The opened .blend is never overwritten. Per-piece settings (kind, skin gap,
max lift, thickness) are in `PIECES` in `armor_forge/shellfit.py`; the layer order (what lies over what)
is `UNDER` there.

Check a build against the drawing and a hand-made kit:

    blender -b ../out/step1/embermine_shells.blend -P blender/compare_to_kit.py -- \
        --set "ArmorForge · Embermine" --kit-scene "Embermine modular fitting" \
        --labels ../input --cams cams.json --colors ../input/Part_Colors.json --out ../out/step1/compare
    blender -b ../out/step1/embermine_shells.blend -P blender/render_set.py -- \
        --collection "ArmorForge · Embermine" --out ../out/step1/renders --prefix shells --lens 85

Embermine result (`out/step1/`): outline match (IoU with the drawn silhouette) 0.92 front/back, 0.87
side; the hand-made kit scores 0.91 / 0.90. Kit surfaces lie 6-14 mm (mean) from the built ones.
Known gaps: no surface detail (visors, trim, buckles, spikes); helmet is a smooth dome; the tabard is
a flat sheet (no side flare); shoulder lames and gorget are small; pieces are not rigged (the
mannequin's Armature modifiers point at no armature).

## Helmet from the concept (shape only, current focus)

    blender -b -P blender/build_helmet.py -- --masks ../input/Masks --cams cams.json --out ../out/helmet

Builds just the helmet, no mannequin and no surface features, as a loft of horizontal slices:
outlines from the full-resolution masks, smoothed hard but never across the brow step; plan shape
round at the back, boxy with a centre "beak" in the visor zone, blending to round above the brow;
crown closes along the comb ridge; hem smoothed around the loop. Visor centre line and brow are
creased for the Subdivision modifier. Tweak with `--param beak=0.3 n_visor=3.5 ...`
(`HelmetParams` in `armor_forge/helmet.py`). Output: `helmet.blend`, renders, and
`overlay.png` (the built outline traced in red on the concept art).

**Outlines traced from the art** (`armor_forge/concept.py`). The masks only guide; the edge comes
from the concept sheet itself. Each view's mask artboard is registered onto its concept panel
(scale + offset fitted on the whole figure), the art's darkness is resampled smoothly into the mask
frame (sub-pixel edges), and the piece is taken where the art is painted, within a reach band around
its mask and never into another piece's colour. Thin grid lines are opened away. Output masks are in
the original frame, so `build_helmet.py --masks <traced>` uses them unchanged.

    python tools/trace_concept.py --art "../reference/Embermine Heavy Armor Turnaround.png" \
        --masks ../input/Masks --cams cams.json --out ../out/helmet/traced
    python tools/trace_overlay.py --art "../reference/Embermine Heavy Armor Turnaround.png" \
        --masks ../input/Masks --traced ../out/helmet/traced --out ../out/helmet/trace_overlay.png

On Embermine the masks were already within 1-2 mm of the painted edge wherever the helmet meets the
background; where it meets the neck and gorget the edge is not a silhouette and stays as masked.

**Helmet as separate parts from the gold trim** (`armor_forge/helmet_parts.py`,
`blender/build_helmet_parts.py`). Depth comes from matching heights between the orthographic views:
the front view gives a part's width at each height, the side view where it starts and ends
front-to-back. The concept's gold trim marks the part edges:

1. gold pixels (warm, lit, yellowish) are picked out of the front and side views and mapped into the
   artboard frame;
2. the visor frame's straight edges are fitted through the gold (tolerates gaps and clutter) and
   moved to the outer edge of each gold band: front view outer sides `|x| = w(z)` and top V, side
   view rear edge `y = y_r(z)` and top edge;
3. **visor**: the slit panel shows in both the front and the side view, so its faces are steeply
   angled: a wedge from the centre ridge (side outline's front) back to the rear edge at `x = +-w`;
   top and bottom of each column from both views; ridge creased;
4. **skull**: the outline with the visor's area cut from the front view (that becomes the face
   opening's edge) and cropped at the neck-guard rim in the side view; its plan matches the visor's
   wedge just behind it, fading to a round dome above the visor; anything left in front of the visor
   is pushed behind it.

    blender -b -P blender/build_helmet_parts.py -- --masks ../out/helmet/traced --cams cams.json \
        --art "../reference/Embermine Heavy Armor Turnaround.png" \
        --trace-meta ../out/helmet/traced/trace.json --out ../out/helmet_parts

Output: `helmet_parts.blend` (objects "Helmet visor" and "Helmet skull"), renders, `lines.png` (the
fitted edges on the concept), `lines.json` (edges + silhouette IoU) and `concept_vs_parts.png`.
Embermine: IoU against the traced outlines 0.93 front, 0.92 side, 0.85 back; visor 187 verts, skull
625. Not yet: the brow band and neck-guard rim as their own bands, the comb as its own crest (the dome
top is pointed because the comb is part of the outline), eye slot and slits.

**Sheets with no masks and no side view** (`armor_forge/nomask.py`, `tools/helmet_from_sheet.py`).
For a sheet like Thornwood (front and back panels on a metric grid only): the scale comes from the
evenly spaced grid lines (`--grid-m`, bottom line = 0 m), `u = 0` from the figure's mirror axis, the
helmet is the figure above the rim (the biggest inward step of the central width between the head's
widest row and the shoulders), and the missing side view is a stand-in that keeps the head target's
depth-to-width ratio (`head_target.json`, SkullTarget sliced). The masks land in the usual artboard
frame, so `build_helmet.py` reads them:

    python tools/helmet_from_sheet.py --art "../blender/Thornwood knight armor turnaround.png" \
        --front 10,880 --back 890,1760 --out ../out/thornwood/masks
    blender -b -P blender/build_helmet.py -- --masks ../out/thornwood/masks \
        --cams ../out/thornwood/masks/cams.json --out ../out/thornwood/helmet

Thornwood result (`out/thornwood/concept_vs_helmet.png`): front/back silhouette right (crest,
brow, rim), side profile guessed; the small flame spikes are smoothed away, the visor's lower tips
(which hang over the gorget) are cut off, and the parts builder can't run (it needs a side view).

**Pick the curvature by eye** (`blender/helmet_variants.py`). The outlines don't decide how round or
boxy the helmet is between the views; this renders a 3 x 3 grid over any two `HelmetParams`, each cell
a three-quarter render plus plan cuts at visor and skull heights, with the concept beside it:

    blender -b -P blender/helmet_variants.py -- --masks ../out/helmet/traced --cams cams.json \
        --art "../reference/Embermine Heavy Armor Turnaround.png" \
        --trace-meta ../out/helmet/traced/trace.json \
        --rows beak=0.2,0.45,0.7 --cols n_visor=2.5,4,6 --out ../out/helmet/variants_visor.png

## Molded pieces (step 2 prototype: helmet)

For the pieces that have to look good, a piece starts from its own clean cage instead of the body:
rings around a centre plus an all-quad cap (no poles), each vertex placed on a smooth hull built from
the drawing (width from front/back, depth from the side, superellipse cross-sections). Features sit
on real edge loops:

* detail colours in the piece map (`<View>_details.npy`: 20 = visor slit, 21 = trim band) are read
  along each meridian; trim bands are fitted as smooth curves around the piece and get exactly two
  rings, slit edges become partial rings across the meridians that see them;
* every open hem gets a trim band automatically;
* trim bands are extruded outward, slits are inset and pushed in; Solidify inward; an optional
  Subdivision Surface modifier is added switched off.

    blender -b ../blender/Embermine_Modular_Geometry.blend -P blender/build_molded.py -- \
        --labels ../input --cams cams.json --colors ../input/Part_Colors.json \
        --out ../out/step2/embermine_molded.blend

The Embermine detail grids in `input/` are stand-ins made with `blender/project_details.py` from the
hand-made helmet (its slit and trim faces projected into the views); in the real pipeline the image
AI paints these colours. Result: `out/step2/helmet_compare.png` (all quads, manifold, ~430 verts).

## Rough-in from four-view sheets (earlier approach)

Inputs: a gridded four-view sheet (Front, Back, Side, Side without arms; each panel 2 m x 2 m,
x -1..+1 m, z 0..2 m) of the AI-drawn armor, the same sheet rendered for the bare mannequin, and
the mannequin as an OBJ.

    pip install -r requirements.txt
    python tools/carve_sheet.py armor_sheet.png mannequin_sheet.png \
        --body-obj mannequin.obj --out carve_out

* Export `mannequin.obj` from Blender with default OBJ settings and *Apply Modifiers* on (body,
  skull stand-in and feet, mirrored).
* The tool reads the silhouettes at metric scale (it ignores the printed "px per m" caption and
  uses the plot rectangle), carves a 3D volume from the four views, limits it to a band (default
  10 cm) around the body surface, and writes `carve_out/armor_rough.obj` plus `preview.png`.
  Import the OBJ into Blender with default settings.
* It prints a silhouette-match score per view (how well the volume reproduces the sheet).
* Result: one fused, chunky armor volume. Cut it into pieces, retopo and refine by hand.
  Next step: a flat-color-coded sheet so each piece is carved separately.

`armor_forge/sheet.py` holds the sheet layout fractions. If a future sheet template changes its
layout, update `LAYOUT` there.

## Inflate the base mesh (keeps the body's curves and topology)

    python tools/inflate_sheet.py armor_sheet.png --body-obj mannequin.obj --out inflate_out

Pushes every vertex of the base mesh outward along its own normal until it would leave an armor
silhouette (front, back, side), then smooths the push distances. The output has the base mesh's
faces and vertex order, so its curves, UVs and weights can be kept; only positions change.
`--subdivide 1` (default) adds a linear subdivision first so the surface can follow the outline.
Silhouettes alone give a smooth, padded look; hard plate edges need the per-piece masks.

## Alternative: mask-to-shell (surface fitting; not yet run in Blender)

## Workflow

1. **Render the mannequin** (fixed orthographic front/back camera, rest pose):

       blender -b MyFile.blend -P blender/render_mannequin.py

   Writes `mannequin_renders/front.png`, `back.png`, `camera.json` next to the .blend.
   Change the body object with `-- --body "Object Name"`.

2. **Generate concept art on those renders.** Feed `front.png`/`back.png` to your image tool as the
   pose/shape reference (edge or depth conditioning keeps the armor aligned to the body).

3. **Make a mask per armor piece.** Same pixel size as the renders. White (or opaque, if the PNG has
   transparency) = armor. Paint it over the concept art or use a segmentation tool.

4. **Fit the shell:**

       blender -b MyFile.blend -P blender/fit_shell.py -- --name Chest \
           --front-mask masks/chest_front.png --back-mask masks/chest_back.png \
           --thickness 0.015 --gap 0.004 --grow 1

   Creates an object named `Chest`, parented to the rig, with the body's vertex groups, a Solidify
   modifier, and an Armature modifier. Then touch up shape and weights.

Prefer the GUI? Open the script from disk in Blender's Text Editor (Text > Open, don't paste it),
edit the `DEFAULTS` block at the top, and press Run.

## Good fits and bad fits

Works well: breastplates, greaves, bracers, gauntlets, tabards, anything that hugs the body.
Weak: protruding pieces (big pauldrons, spikes, horns, crests, capes). A front/back outline can't tell
depth. Model those by hand, or add a side-view step later. Put surface detail (trim, rivets,
engraving) in textures/normal maps rather than geometry.

## Audit tool (no Blender needed)

    python tools/blend_audit.py MyFile.blend

Reads the raw .blend and reports likely rig problems: Rest Position left on, helper bones that
deform, vertex groups with no bone, vertex weights that don't sum to 1, broken constraint or
driver targets. Read-only. Works on Blender 3.x/4.x files (classic header). Compressed files need
libzstd on the system.

## Layout

    armor_forge/   pure Python/numpy: sheet reading, carving, meshing, projection, selection; unit-tested
    blender/       scripts that run inside Blender (render_mannequin.py, fit_shell.py)
    tools/         standalone tools (carve_sheet.py, inflate_sheet.py, blend_audit.py)
    tests/         pytest suite for armor_forge/

Run tests: `pip install -r requirements-dev.txt && pytest`

## Status

- `armor_forge/` (sheet reading, carving, meshing) has unit tests; `tools/carve_sheet.py` was run on a real
  sheet pair. `tools/blend_audit.py` reads Blender 3.x-5.x files.
- `blender/build_shell_set.py`, `compare_to_kit.py` and `render_set.py` run headless in Blender 5.1
  (also as the `bpy` 5.1.2 module from PyPI). `blender/fit_shell.py` is superseded by
  `build_shell_set.py` and has not been run.

## Roadmap

- Side-view masks for depth, and a depth/height map to vary thickness per piece
- Bake the concept art onto the shell as a texture (projection) for surface detail
- Per-piece presets (gap/thickness) and batch fitting for whole sets
- Rig repair script (turn off Deform on non-DEF bones, normalize weights)
