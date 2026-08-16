# Installation

## Environment

The validated source-build environment is listed below. All Python, PyMeep,
MPI, compiler, and TAMA components must come from the same conda environment
because the native extension links directly to Meep and MPI.

| Component | Validated support |
|---|---|
| Operating system | Ubuntu 26.04 x86_64 |
| Python | CPython 3.11, 3.12, or 3.13 |
| Meep | conda-forge PyMeep 1.34.0 `mpi_mpich` build |
| MPI | MPICH 4.3.2 with mpi4py 4.1.2 |
| Distribution | Source archive/sdist built inside the target environment |
| Validated execution | Serial and MPI with 2/4/8 ranks on Python 3.11-3.13 |

OpenMPI, macOS, and Windows are outside the validated support matrix for this
release. Prebuilt binary wheels are not distributed, and a locally built wheel
must not be copied between conda environments. Use a conda environment with
MPI-enabled Meep and the Python dependencies used by the example scripts.

From a source checkout, create the validated environment with:

```bash
conda env create -f environment.yml
conda activate tama
```

The equivalent manual environment creation command is:

```bash
conda create -n <env-name> --override-channels -c conda-forge \
  python=3.11 \
  "pymeep=1.34.0=mpi_mpich_*" \
  mpi4py=4.1.2 mpich=4.3.2 cxx-compiler cmake ninja pip \
  "scikit-build-core>=1.0" gsl \
  nlopt scipy matplotlib autograd numpy pytest
conda activate <env-name>
```

For an existing environment:

```bash
conda activate <env-name>
conda install --override-channels -c conda-forge \
  "pymeep=1.34.0=mpi_mpich_*" \
  mpi4py=4.1.2 mpich=4.3.2 cxx-compiler cmake ninja pip \
  "scikit-build-core>=1.0" gsl \
  nlopt scipy matplotlib autograd numpy pytest
```

Check the environment:

```bash
python -c "import meep as mp; print(mp.__version__)"
which python
which mpic++
```

## Install From Source

From a source checkout or unpacked source release, install TAMA into the
active environment:

```bash
cd <tama-source-directory>
python -m pip install --no-build-isolation .
python -c "import tama; print(tama.__version__); print(tama.native_sampler_available())"
```

## Native Sampler

TAMA includes a C++ extension for field sampling, exact point-monitor
transposes, and native MaterialGrid-transpose gradients. This extension is
required for all TAMA runtime use. The normal package installation builds
it automatically with CMake using Python, NumPy, MPI, and Meep from the active
conda environment.

Verify that the installed package can find the native sampler:

```bash
python -c "import tama; print(tama.__version__); print(tama.native_sampler_available())"
python -c "import importlib.util; print(importlib.util.find_spec('tama.native_sampler').origin)"
```

After a successful build, `native_sampler_available()` should print `True` and
the module origin should end with the platform's compiled-extension suffix
(for example, `.so` on Linux). If package configuration cannot find Meep or
MPI, activate the conda environment containing the MPI-enabled pymeep build and
install the package again.

Importing TAMA requires the compiled extension. Rebuild TAMA after updating or
rebuilding PyMeep, even when the Meep version string is unchanged.

## Basic Imports

After installation, scripts can import TAMA from any working directory in
the same Python environment.

```python
import meep as mp
import numpy as np
import tama as tm
```
