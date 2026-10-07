# Anisotropic materials and averaging: implementation report

Date: 2026-10-07. Branch: `feat/anisotropic-material-averaging`.
Base: `4d6bc788f87e9b5877310ebe45d2c29fd2a8ccec`.

Cartesian 2D/3D TAMA now supports density gradients between fixed real symmetric
positive-definite dielectric tensors, MaterialGrid `do_averaging`, internal
`beta`/`eta` projection, and Simulation `eps_averaging`. This is implemented in
the native solver integration and both objective classes, including forward-only
evaluation and simulation reuse. The native interface is API 13.

## Behavior and implementation

- `eps_averaging` controls global subpixel averaging. MaterialGrid interface
  averaging requires both `eps_averaging=True` and `do_averaging=True`.
  Internal projection remains active when averaging is disabled.
- The general gradient records displacement fields and contracts the derivative
  of the discrete inverse-permittivity operator with forward/adjoint fields.
  It includes Meep's staggered off-diagonal stencil, density interpolation,
  projection, and the density-dependent interface normal.
- Material derivatives use local finite differences with step `1e-5`, central
  differences in the interior and second-order one-sided differences near the
  weight bounds. They do not require one additional FDTD solve per variable.
  This is an adjoint gradient with a numerical local material Jacobian, not a
  completely analytic derivative.
- Anisotropic MaterialGrid averaging uses the interface-normal tensor mixing
  rule of [Kottke, Farjadpour, and Johnson](https://arxiv.org/abs/0708.1031).
  The same evaluator initializes the forward/adjoint material and computes
  its local Jacobian. Isotropic endpoints reduce to harmonic normal and
  arithmetic tangential averaging.
- The scalar path with averaging off and `beta=0` remains available. The general
  path already includes the endpoint tensors and does not multiply by the
  scalar `material_factor` a second time.

Code: [material tensor evaluator](native/meep/material_tensor.hpp),
[native plan and contraction](native/meep/native_sampler.cpp),
[validation and initialization](src/tama/backends/meep/sampling_grid.py).
Usage and constraints: [design gradient guide](docs/design-gradients.md).

## Numerical verification

All directional checks use the actual forward objective, a deterministic
nonuniform density field, and a normalized random direction. Central finite
differences use `h=2e-4` and `h=1e-4`. Relative error is
`abs(adjoint - FD_1e-4) / abs(FD_1e-4)`.
The tests require matching signs, finite outputs, adjoint/FD agreement within
**0.5%**, and FD step-halving agreement within **0.1%**.

| Averaged scalar / unaveraged tensor case | Adjoint/FD error |
| --- | ---: |
| 2D scalar Ez, beta=32, eta=0.42 | 0.082836% |
| 2D scalar Hz, beta=8 | 0.169102% |
| Rotated tensor Hz, MultiTDA, averaging off | 0.171324% |
| Fixed scalar curved geometry, global averaging on | 0.093784% |

| Tensor coupling case | Adjoint/FD error |
| --- | ---: |
| Fully rotated 3D tensor, averaging on | 0.168820% |
| 3D tensor with a singleton design z axis | 0.114281% |
| 2D tensor with xz/yz coupling, averaging on | 0.080091% |
| Fixed external tensor, global averaging off | 0.082074% |

| High-beta averaged case | Adjoint/FD error |
| --- | ---: |
| Scalar Hz, beta=128 | 0.176875% |
| Tensor 2D, beta=128 | 0.073842% |
| Scalar Hz, beta=256 | 0.176306% |
| Tensor 2D, beta=256 | 0.074512% |

Maximum FD step-halving change across these 12 checks: **0.00128293%**.
Sparse stride-2 versus dense gradient relative L2 difference:
**0.00565008%**; the forward objectives agree to the test tolerance `1e-12`.
These are small directional validation problems, not a universal error bound
or a production optimization benchmark.

The 2-rank split crosses the averaged design. A separate 4-rank 3D check splits
at x=0 and then y=0 in both children, exercising mixed-stencil chunk corners.
Both pass the same two-step FD criteria. The final 4-rank result is
adjoint `-0.00480203166615`, FD `-0.0048101521612359255` at `h=1e-4`.

Independent C++ algebra checks pass at `1e-14` for pure endpoints, identical
media, isotropic harmonic/arithmetic reduction, and rotation covariance.
Projection normalization, asymmetric-eta continuity, and infinite-beta volume
fractions also pass. Forward-operator checks establish that anisotropic
averaging actually changes the material, preserves homogeneous/constant
limits, and is disabled by the global flag. In the tested stock Meep 1.34.0
interior MaterialGrid fallback, anisotropic `do_averaging` alone changed no
operator entries; TAMA's implementation gives a nonzero change.

### High-beta accuracy correction

Meep's default `subpixel_tol=1e-4` produced quadrature noise: the scalar beta=256
FD derivative changed by about 26% when halving the step, and the tensor
beta=128 adjoint discrepancy was about 1.2%. Tightening the common material
integration tolerance resolved both observed failures without weakening test
thresholds.

For active MaterialGrid averaging, TAMA now sets
`subpixel_tol=min(requested_tolerance, 1e-8)` before initialization.
Forward, adjoint, and local material derivatives use the same tolerance;
`subpixel_maxeval` is unchanged. Stricter user tolerances are preserved, and
inactive averaging leaves the tolerance unchanged. This increases material
setup cost. Already initialized simulations with looser tolerance are rejected
instead of differentiating a different forward operator.

## Test runs and environment

- Final clean paired package build: succeeded; native **API 13**.
- New tensor/averaging tests plus sampler safety and package checks:
  **95 passed**, with the MPI-only case run separately.
- Existing TDA/MultiTDA/sampling unit regressions: **376 passed**.
- Existing native, scalar, 3D, cylindrical, history, and spec checks:
  **183 passed** in the broader regression run. That run also exposed two
  factory-error cleanup failures; both were fixed and are included in the
  final 376 passing tests above. Counts describe test groups, not a claimed
  single all-repository run.
- MPI: **2-rank and 4-rank checks passed**. Launches used MPICH `-launcher fork`.
- C++ tensor algebra self-check: passed.
- Documentation: Sphinx **`-W --keep-going` passed**.

Ubuntu WSL, Python 3.13.12, Meep 1.34.0, NumPy 2.5.2, MPICH 4.3.2;
OMP/OpenBLAS/MKL thread counts were 1 for the final numerical suite.
Runtime: `/home/smrm/tmp/tama-anisotropic-20261007/venv`.
Dependency prefix `/home/smrm/miniconda3/envs/pmp` was used read-only.
The final packages were built from a clean source snapshot to exclude stale
ignored build artifacts. Runtime imports came from the task venv, with
`PYTHONPATH` unset.

Evidence directory: `/home/smrm/tmp/tama-anisotropic-20261007/`:

- `accuracy_source_manifest.json`: source hashes for the tested build;
  `accuracy_source_125334/`: frozen source.
- `accuracy_final_build.log`, `accuracy_final_serial.log`,
  `accuracy_objective_regression.log`, `clean_broad_regression.log`.
- `accuracy_final_mpi2.log`, `accuracy_final_mpi4.log`,
  `accuracy_mpi4_tensor_probe.py`.
- `accuracy_derivative_table.json` / `.md`: complete derivative values;
  `high_beta_investigation.json`: quadrature investigation.
- `material_tensor_selfcheck.log`, `docs_final_build.log`.

Native source SHA256:
`990e17a0c7a637339e53f504ff5570af4201041f30154023cadfb13d345aa868`.
Tensor evaluator SHA256:
`b82494336305b084f10c29b9c21fe0e437fb31a40ee401236f04b2d35c2be729`.

Runnable regression commands, from the repository root in the built environment:

```bash
python -m pytest -q tests/meep/test_native_material_tensor.py \
  tests/meep/test_native_sampler_safety.py tests/common -m 'not mpi2'
python -m pytest -q tests/meep/test_tda_objective.py \
  tests/meep/test_multi_tda_objective.py tests/meep/test_sampling_grid.py
mpirun -launcher fork -np 2 python -m pytest -q \
  tests/meep/test_native_material_tensor.py -m mpi2
```

The C++ self-check has its compile/run command in
[`tests/native/test_material_tensor.cpp`](tests/native/test_material_tensor.cpp).

## Limits and deployment

Each variable is a scalar density between two fixed tensors. Independent tensor
entry optimization, complex/Hermitian tensors, dispersion, conductivity, and
nonlinear design media are not implemented. Endpoint permeability must be equal,
finite, positive, and isotropic. Tensor/averaged cylindrical gradients are not
implemented; the existing scalar cylindrical path is preserved.

The design must use one axis-aligned MaterialGrid Block with `U_DEFAULT` and
`damping=0`, no symmetry reduction or nonzero Bloch vector, and at least half a
grid cell of clearance from PML/absorbers. A fresh, uninitialized simulation
factory is required for anisotropic MaterialGrid averaging. Normal TAMA reuse
is supported and tested.

Gradient evaluation rejects perturbations crossing the undefined zero-normal
averaging branch at mixed densities. Forward evaluation remains available.
Clamped points whose normal stays identically zero are allowed. Infinite-beta
projection at its unsmoothed threshold, and the tested stock Meep asymmetric
projection discontinuity, are rejected for gradients. Use finite beta and a
spatially varying density for averaged optimization.

The general plan stores extra mixed-component displacement samples and computes
local material finite differences, so it can cost more setup time and history
storage than the scalar fast path. Production runtime/memory scaling and a full
WDM optimization with this implementation have not been benchmarked.

Both the Python core and native extension must be reinstalled together for API
13. Only the isolated local test environment was updated. Server7's running
optimization, shared solver environments, existing WDM results, and manuscript
files were not changed. This branch is local and unpublished. ReadTheDocs has
not been rebuilt or deployed; after publication its build must be checked and
triggered manually if the webhook does not start it.
