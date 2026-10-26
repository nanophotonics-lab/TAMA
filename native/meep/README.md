# TAMA Meep native sampler

This subproject builds the Meep/MPI C++ sampler separately from the
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
to install TAMA's pure Python core.

The extension installs as `tama_meep_native.native_sampler`. Compatibility
`tama.native_sampler` and backend `tama.backends.meep.native_sampler` imports
resolve to this same module, including the native API capability check.
Builds are tied to the selected Python, Meep, MPI and NumPy ABI; do not distribute
a locally built wheel as a universal binary.

The native sampler stores identical tensor forward samples once while retaining
every material-gradient contribution. Tensor plan signatures can have more rows
than their history width and include each contribution's history-column index.

Cylindrical near-to-far source transposes use radial quadrature and the actual
cylindrical Yee-cell volume.

Adjoint accumulation uses the discrete Yee forward difference and preserves
source endpoints within floating-point roundoff.

The sampler supports independent design-region offsets, Cartesian Mirror
sampling, source transposes, and near-to-far source folding. Real symmetric
dielectric MaterialGrid gradients support internal projection and subpixel
averaging. Tensor interface averaging uses libctlgeom's adaptive quadrature.

The Python package requires native API 17. Install the Python core and extension
together. Incompatible native API versions are rejected.
