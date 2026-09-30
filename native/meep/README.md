# TAMA Meep native sampler

This subproject builds the unchanged Meep/MPI C++ sampler separately from the
pure Python `tama` distribution. Build it in the same conda environment that
provides `pymeep`, MPI, NumPy, the C++ compiler, CMake, Ninja and GSL.

From the repository root, after activating that environment:

```bash
python -m pip install --no-build-isolation .
python -m pip install --no-build-isolation ./native/meep
python -c "import tama; assert tama.native_sampler_available()"
```

`MEEP_CONDA_PREFIX` can select the environment explicitly; otherwise the build
uses `CONDA_PREFIX`. `./build_native_sampler.sh` performs both installations with
the selected environment's Python. Neither Meep nor this extension is required
to install TAMA's pure Python core or run its FDTDX backend.

The extension installs as `tama_meep_native.native_sampler`. Legacy
`tama.native_sampler` and backend `tama.backends.meep.native_sampler` imports
resolve to this same module, including the original native API capability check.
Builds are tied to the selected Python, Meep, MPI and NumPy ABI; do not distribute
a locally built wheel as a universal binary.
