# TAMA: Time-domain Adjoint with Memory-efficient Acceleration

[![CI](https://github.com/nanophotonics-lab/TAMA/actions/workflows/ci.yml/badge.svg)](https://github.com/nanophotonics-lab/TAMA/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/tama-adjoint/badge/?version=latest)](https://tama-adjoint.readthedocs.io/en/latest/)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL_v2%2B-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v0.5.0--alpha.2-orange.svg)](https://github.com/nanophotonics-lab/TAMA/releases/tag/v0.5.0-alpha.2)

TAMA (Time-domain Adjoint with Memory-efficient Acceleration, imported as
`tama`) is a time-domain adjoint optimization package built on Meep. Users
create Meep sources, geometry, materials, and simulations directly, while
TAMA provides native Yee-grid design gradients; point, regional-field,
signed-flux, and eigenmode objectives; broadband temporal filtering; sparse
field-history reconstruction; periodic fabrication filtering; and MPI-aware
execution.

## Status

The current pre-release is `v0.5.0-alpha.2` (Python package version
`0.5.0a2`). APIs, examples, and native-sampler build details may change
before the first stable release.

This is the first public pre-release in this repository.

The native `DesignGrid` path supports 2D TMz/TEz, 3D scalar-permittivity
designs using the full `Ex`/`Ey`/`Ez` gradient contraction, and Meep
cylindrical simulations with `m=-1`, `0`, or `+1`. Full-vector refers to the
fields used in that contraction; it does not mean anisotropic or
tensor-permittivity optimization.

## Documentation

The [complete user manual](https://tama-adjoint.readthedocs.io/en/latest/) is also
available in the repository under [`docs/`](docs/index.md):

- [installation and validated environment](docs/installation.md)
- [project layout](docs/project-layout.md)
- [design gradients and grid coordinates](docs/design-gradients.md)
- [periodic fabrication filter and VJP](docs/fabrication.md)
- [objective conventions and targets](docs/targets.md)
- [`TDAObjective`](docs/tda-objective.md)
- [`MultiTDAObjective`](docs/multi-tda-objective.md)
- [MPI chunk topology and source safety](docs/mpi.md)
- [running and testing](docs/running-and-testing.md)
- [troubleshooting](docs/troubleshooting.md)

To build the HTML manual without importing TAMA, Meep, or the native
extension, use pip 25.1 or newer:

```bash
python -m pip install --group docs
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Install From Source

TAMA is validated on Linux with CPython 3.11-3.13 and the conda-forge
MPI-enabled PyMeep 1.34.0 MPICH build. Python, PyMeep, MPI, the compiler, and
TAMA must come from the same conda environment because the native extension
links directly to Meep and MPI.

Create the supplied environment and install the source tree:

```bash
conda env create -f environment.yml
conda activate tama
python -m pip install --no-build-isolation .
python -c "import tama; print(tama.__version__); print(tama.native_sampler_available())"
```

For development, replace the install command with:

```bash
python -m pip install --no-build-isolation -e .
```

After a successful build, `native_sampler_available()` must print `True`.
Rebuild TAMA after updating or rebuilding PyMeep, even if the Meep version
string is unchanged. See the [installation guide](docs/installation.md) for
the complete support matrix, manual environment command, and native-library
diagnostics.

## First Runs

Run one projected-gradient iteration of the bundled 3D example:

```bash
python examples/tda_3d_projected_gradient.py --iterations 1
```

Build the reduced normal-incidence Gamma-periodic RGB-router example without
running FDTD, or run one optimization iteration:

```bash
python examples/tda_3d_gamma_rgb_router.py --dry-run
python examples/tda_3d_gamma_rgb_router.py --iterations 1 --output rgb.npz
```

Run the cylindrical point-target example in serial or MPI:

```bash
python examples/tda_cylindrical_mode.py --target point
mpirun -np 2 python examples/tda_cylindrical_mode.py --target point
```

The objective guides document the complete setup for
[`TDAObjective`](docs/tda-objective.md) and
[`MultiTDAObjective`](docs/multi-tda-objective.md).

## Testing

Run the serial test suite from the repository root:

```bash
python -m pytest -q
```

Run MPI marker groups with their matching rank counts:

```bash
mpirun -np 2 python -m pytest -q -p no:cacheprovider -m mpi2
mpirun -np 4 python -m pytest -q -p no:cacheprovider -m mpi4
mpirun -np 8 python -m pytest -q -p no:cacheprovider -m mpi8
```

## License

TAMA is licensed under the GNU General Public License version 2.0 or later
(`GPL-2.0-or-later`). TAMA depends on Meep, which uses the same license. TAMA
is an independent project and is not affiliated with or endorsed by the Meep
project.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
