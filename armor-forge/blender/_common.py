"""Shared helpers for the Blender-side scripts.

The scripts work two ways:
  * headless:   blender -b file.blend -P blender/<script>.py -- --option value
  * GUI:        open the script in Blender's Text Editor, edit DEFAULTS at the top, press Run
"""
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_path():
    """Make `import armor_forge` work inside Blender."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return ROOT


def script_args():
    """Arguments after '--' on the Blender command line (empty list in the GUI)."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def blend_dir():
    import bpy
    if not bpy.data.filepath:
        raise RuntimeError("Save the .blend first so outputs have a folder to live in.")
    return os.path.dirname(bpy.data.filepath)
