"""Load the shared example scene without changing the installed package path."""

from pathlib import Path
import runpy

make_scene = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/fdtdx/native_scene.py")
)["make_scene"]
