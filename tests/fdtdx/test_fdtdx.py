"""Installed-package checks isolated by process and JAX precision/platform."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
GPU_CASES = ("main", "periodic", "gaussian", "dipole", "float32",
             "fabrication", "sparse", "pec", "pmc", "bloch", "bloch32")


def _run(script, *arguments, gpu=False):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, arguments)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
             "JAX_PLATFORMS": "cuda" if gpu else "cpu",
             "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.skipif(importlib.util.find_spec("fdtdx") is None,
                    reason="Install the fdtdx extra for CPU backend checks")
@pytest.mark.parametrize("name", ("specs", "functionals", "api", "adapter"))
def test_fdtdx_cpu(name):
    _run(f"check_{name}.py")


@pytest.mark.fdtdx_gpu
@pytest.mark.parametrize("case", GPU_CASES)
def test_fdtdx_gpu(case, tmp_path):
    # Explicit selection must fail if CUDA or its dependencies are unavailable.
    _run("check_backend.py", "--case", case,
         "--output", tmp_path / f"{case}.json", gpu=True)


@pytest.mark.fdtdx_gpu
def test_fdtdx_public_api_gpu(tmp_path):
    _run("check_integration.py", "--output", tmp_path / "integration.json", gpu=True)
