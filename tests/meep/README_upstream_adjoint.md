# Meep adjoint regression ports

These tests apply supported Meep adjoint scenes to the installed TAMA package.
They retain the upstream geometry, material parameters, frequency cases,
perturbations, observables, and comparison tolerances. They do not run Meep's
adjoint solver as a substitute for TAMA's gradient.

## Run

Install TAMA and its native Meep sampler as described in
[running and testing](../../docs/running-and-testing.md). Do not add `src` to
`PYTHONPATH`. From the repository root, run:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  tests/meep/test_upstream_adjoint_solver.py \
  tests/meep/test_upstream_adjoint_port_ratio.py \
  tests/meep/test_upstream_adjoint_filters.py
```

All 42 cases also belong to the ordinary `tests/meep` collection. A numerical
mismatch is a test failure, without a skip, xfail, or relaxed tolerance.
The dense solver tests retain full time histories and require several GiB of
memory. Run them serially before considering parallel execution.

## Source and coverage

The source is Meep v1.34.0, commit
`3e7b7fee0da4a9b91b07acd9f71d35d674396563`:

- [test_adjoint_solver.py](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/python/tests/test_adjoint_solver.py):
  32 cases cover DFT fields, eigenmode coefficients, filter/projection
  backpropagation, off-diagonal permittivity, two objective functions,
  multi-frequency equivalence, and five mode-source bandwidths. Separating
  assertions by frequency or objective lets all conditions run after a failure.
- [test_adjoint_jax.py](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/python/tests/test_adjoint_jax.py):
  six waveguide cases check the mean output/input modal-power ratio, including
  the gradient of the design-dependent input power. Each uses five finite
  differences. This ports the physical gradient checks, not the JAX wrapper API.
- `test_adjoint_solver.py::test_periodic_design`: four cases check the fully
  periodic conic filter and projection with solid/void constraints and x/y
  shifts. Meep supplies the constraint functions; TAMA supplies the filter,
  projection, and vector-Jacobian products.

## TAMA adaptations

TAMA records 500 time units with `sampling_interval=1` instead of upstream
adaptive DFT decay. Fourier objectives use TAMA time histories. Modal targets
use explicit trapezoidal quadrature and a reference mode at each frequency.
These target the same physical observables, but are not bitwise copies of Meep's
DFT monitors. The module docstrings record quadrature and normalization details.
The single-frequency two-objective checks combine separate TDA evaluations;
the multi-frequency checks exercise `MultiTDAObjective`.

The filter/projection backpropagation check uses the second-order forward
difference `[4(F(p+dp)-F(p))-(F(p+2dp)-F(p))]/2`. The perturbation `dp`, mapping,
and upstream tolerance are unchanged. The extra positive perturbation remains
within the design bounds. For the single-frequency fixture, the upstream
first-order reference has relative truncation error `9.0896e-6`, above its
`5e-6` tolerance. The second-order reference agrees with a separately converged
central difference to `1.6e-8`. This is a documented change to the reference
formula, not a claim that the unmodified upstream assertion passes.

The port-ratio directions are stored in
`data/upstream_adjoint_port_ratio_directions.npy`, so Meep tests do not need
JAX at runtime. The test records the generation expression, JAX version, PRNG
settings, and SHA256, and verifies the fixture before use. Upstream did not
pin the JAX version or PRNG partitioning setting.

Only matching filter semantics are ported. Meep's `periodic_axes` selects
periodic boundaries; TAMA's option selects the axes to filter. They match here
when both axes are periodic. Single-axis, nonperiodic, unequal-resolution,
Gaussian, and cylindrical filter cases are outside this port.

These files do not cover upstream cylindrical near-to-far adjoints, the
nonzero-Bloch/Dz `test_complex_fields` case, damping, LDOS,
`test_unfilter_design`, or JAX wrapper internals. Existing TAMA tests remain
responsible for its own APIs and supported near-to-far paths.
