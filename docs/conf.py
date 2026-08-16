import os
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
METADATA = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]

project = "TAMA"
author = ", ".join(entry["name"] for entry in METADATA["authors"])
version = METADATA["version"]
release = version

extensions = ["myst_parser", "sphinx_rtd_theme"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} documentation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://tama-adjoint.readthedocs.io/en/latest/",
)
