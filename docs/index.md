# TAMA: Time-domain Adjoint with Memory-efficient Acceleration

TAMA supports explicit Meep and FDTDX backends. The pure Python core contains
both implementations; Meep's native sampler is built separately. Existing root
APIs continue to select Meep. This documentation describes the `0.6.0a1` alpha.

```{toctree}
:maxdepth: 2

installation
backends
fdtdx
project-layout
design-gradients
fabrication
targets
tda-objective
multi-tda-objective
mpi
running-and-testing
refactor-validation
troubleshooting
```

The design-gradient, fabrication, target, objective and MPI guides retain the
Meep numerical conventions. The FDTDX guide specifies its separate coordinate,
unit, normalization and supported-physics contract.
