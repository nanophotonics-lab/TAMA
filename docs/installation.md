# Installation

## Python package

TAMA is a pure Python package containing both backend implementations. Installing
it alone does not install an FDTD engine or build a native module:

```bash
python -m pip install .
python -c "import tama; print(tama.__version__)"
```

Run these commands from the root of a cloned repository or unpacked source
distribution. Build Meep's native sampler from the included `native/meep` directory
as shown below.

## Meep

The existing conda environment definition retains PyMeep 1.34.0 MPICH,
MPICH 4.3.2, mpi4py 4.1.2, the compiler and native build dependencies.
Python 3.11, 3.12 and 3.13 are covered by the CI matrix, including the original
NumPy 1.26 compatibility lane.

```bash
conda env create -f environment.yml
conda activate tama
python -m pip install .
python -m pip install --no-build-isolation ./native/meep
python -c "import tama; print(tama.get_backend('meep').native_sampler_available())"
python -c "import tama_meep_native.native_sampler as n; print(n.__file__)"
```

The last path must point to a compiled extension. `build_native_sampler.sh`
provides the combined installation helper. Rebuild the native package after
changing PyMeep, MPI or the compiler environment, even if version strings match.
Local native wheels are environment-specific; they are not portable Meep wheels.
OpenMPI, macOS and native Windows builds are outside the validated native lane.

The `meep` extra depends on `tama-meep-native==0.6.0a1`; it does not install
PyMeep from PyPI. Build the native distribution locally before using the extra.

## FDTDX

Use Python 3.12 or 3.13. The exact dependencies are FDTDX 0.6.2,
JAX/JAXlib 0.11.0 and Equinox 0.13.8, with NumPy >=2 and SciPy >=1.13.
JAX's Python requirement prevents selecting this extra on Python 3.11;
the Meep Python 3.11 support remains available.

```bash
python -m pip install '.[fdtdx]'
# Linux/WSL CUDA 12 setup, if not already installed in this environment:
python -m pip install 'jax[cuda12]==0.11.0'
JAX_PLATFORMS=cuda python -c "import jax, tama; print(jax.devices()); print(tama.get_backend('fdtdx'))"
```

The tested GPU lane uses the CUDA 12 plugin/PJRT version 0.11.0 and an NVIDIA
driver suitable for that runtime. This command does not install the host driver.
CPU evaluation is available for small tests; CPU offload requires an actual
CUDA device. CUDA on native Windows is not validated; use Linux or WSL.
FDTDX selection never falls back to Meep or silently accepts CPU for a GPU test.

Float64/complex128 scenes require `jax.config.update('jax_enable_x64', True)`
before creating native arrays. Use `False` before constructing float32/complex64
scenes; mixed-precision field/source caches are rejected.

## Migrating an existing installation

Reinstall the core package and install the new native package in the existing
Meep environment. Root-level Meep classes and old Python submodule paths remain
available as aliases. Old extensions under `tama/native_sampler.so` are not the
new native package: verify the `tama_meep_native` path shown above.

For FDTDX, install the core with its `fdtdx` extra. The former standalone
`tama_fdtdx` package is not needed by this version. Scene and target conventions
are described in [backend migration](backends.md) and [FDTDX](fdtdx.md).
