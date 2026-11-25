# CLAUDE.md

armor-forge: turns concept-art masks of armor on a mannequin into fitted, pre-rigged armor shells in Blender. See `README.md` for the workflow and roadmap.

## Layout

- `armor_forge/` – pure Python/numpy math (projection, mask sampling, face selection). **No `bpy` imports here**, so it stays unit-testable.
- `blender/` – scripts that run inside Blender (`render_mannequin.py`, `fit_shell.py`, shared `_common.py`). They work headless (`blender -b file.blend -P blender/x.py -- --opt val`) and in the GUI (edit the `DEFAULTS` block).
- `tools/blend_audit.py` – standalone read-only .blend rig auditor; parses the raw file, needs no Blender.
- `tests/` – pytest suite for `armor_forge/`.

## Commands

```bash
pip install -r requirements-dev.txt
python3 -m pytest          # config in pytest.ini (pythonpath=., testpaths=tests)
```

## Conventions and gotchas

- Match Blender's conventions in the math: cameras look down local -Z, +Y is image up, image rows are stored bottom-to-top (row 0 = bottom), and an ortho camera's `ortho_scale` spans the larger image dimension.
- Keep Blender-specific code in `blender/`; put anything testable in `armor_forge/` and add tests for it.
- Blender isn't installed in the cloud environment, and `blender/*.py` have never been run inside Blender (written against the 4.3 API). Don't claim they work; say they're untested unless verified. Expect small API fixes on first real run.
- `tools/blend_audit.py` is read-only; keep it that way.
- Generated outputs (`mannequin_renders/`, `masks/`, `*.blend1`) are gitignored; don't commit them.
