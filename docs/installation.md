# Installation

## Python package

TAMA's Python package provides the Meep adjoint interface. Installing it alone
does not install Meep or build the native sampler:

```bash
python -m pip install .
python -c "import tama; print(tama.__version__)"
```

Run these commands from the root of a cloned repository or unpacked source
distribution. Build Meep's native sampler from the included `native/meep` directory
as shown below.

## Meep

The conda environment and automated CI use PyMeep 1.34.0 MPICH,
MPICH 4.3.2, mpi4py 4.1.2, the compiler and native build dependencies.
Python 3.11, 3.12 and 3.13 are covered by the CI matrix, including the
NumPy 1.26 compatibility lane. This conda lane remains on 1.34.0 until the
Meep 1.35.0 package is available on conda-forge.

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

### Meep 1.35.0

To use Meep 1.35.0, build the
[meep-1.35.0.tar.gz release archive](https://github.com/NanoComp/meep/releases/download/v1.35.0/meep-1.35.0.tar.gz)
with the [upstream build instructions](https://meep.readthedocs.io/en/latest/Build_From_Source/).
Install its Python module, headers and library into the same dependency environment.
Enable MPI and use that environment's MPI compiler and libraries. The commands
above for `environment.yml` alone install Meep 1.34.0.

After installing Meep 1.35.0, verify the imported version and rebuild TAMA's native
module against it:

```bash
python -c "import meep; assert meep.__version__ == '1.35.0'; print(meep.__file__)"
python -m pip install .
MEEP_CONDA_PREFIX="$CONDA_PREFIX" python -m pip install \
  --no-build-isolation --no-deps --force-reinstall ./native/meep
python -c "import tama; assert tama.get_backend('meep').native_sampler_available()"
```

A source installation can leave older conda package metadata in place. Use the
imported version and linked `libmeep` to identify the engine used for a benchmark.
Keep the release archive and build options with the benchmark record.

## Migrating an existing installation

Reinstall the core package and install the separate native package in the existing
Meep environment. Root-level Meep classes and compatibility Python submodule
paths are available as aliases. Verify that the native module loads from
`tama_meep_native`, rather than `tama/native_sampler.so`.
