# TAMA: Time-domain Adjoint with Memory-efficient Acceleration

[![CI](https://github.com/nanophotonics-lab/TAMA/actions/workflows/ci.yml/badge.svg)](https://github.com/nanophotonics-lab/TAMA/actions/workflows/ci.yml)
[![License: GPL v2+](https://img.shields.io/badge/License-GPL_v2%2B-blue.svg)](../LICENSE)
[![Release](https://img.shields.io/badge/release-v0.5.0--alpha.3-orange.svg)](https://github.com/nanophotonics-lab/TAMA/releases/tag/v0.5.0-alpha.3)

TAMA (Time-domain Adjoint with Memory-efficient Acceleration, imported as
`tama`) is a time-domain adjoint optimization package built on Meep. Users create Meep
sources, geometry, materials, and simulations directly, while TAMA provides
native Yee-grid design gradients; point, regional-field, signed-flux, and
eigenmode objectives; broadband temporal filtering; sparse field-history
reconstruction; periodic fabrication filtering; and MPI-aware execution.

## Documentation

```{toctree}
:maxdepth: 2

installation
project-layout
design-gradients
fabrication
targets
tda-objective
multi-tda-objective
mpi
running-and-testing
troubleshooting
```

## Status

The current pre-release is `v0.5.0-alpha.3` (Python package version
`0.5.0a3`). APIs, examples, and native-sampler build details may change
before the first stable release.

**The native `DesignGrid` path supports 2D TMz/TEz, 3D
scalar-permittivity designs using the full
`Ex`/`Ey`/`Ez` gradient contraction, and Meep cylindrical simulations with
`m=-1`, `0`, or `+1`. Here, full-vector refers to the fields used in that
contraction; it does not mean anisotropic or tensor-permittivity optimization.**

## License

TAMA is licensed under the GNU General Public License version 2.0 or later
(`GPL-2.0-or-later`).

TAMA depends on Meep, which is distributed under the GNU General Public
License version 2.0 or later (`GPL-2.0-or-later`). TAMA is an independent
project and is not affiliated with or endorsed by the Meep project.
