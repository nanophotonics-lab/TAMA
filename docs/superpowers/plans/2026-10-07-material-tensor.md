---
orphan: true
---

# Tensor material gradients and subpixel averaging

**Goal:** Support Cartesian 2D/3D density optimization between real symmetric,
positive-definite, nondispersive dielectric tensors, including MaterialGrid
projection and averaging and Simulation.eps_averaging.

**Architecture:** Preserve the existing scalar fast path. The general path
records displacement fields and differentiates Meep's discrete inverse
permittivity operator, including its staggered mixed-component stencil. Extend
MaterialGrid averaging to tensor endpoints using the interface-normal tensor
mixing rule, and use the same evaluator for forward initialization and its
local material Jacobian. No external solver or shared environment is modified.

**Constraints:** One axis-aligned MaterialGrid Block, fixed permeability,
no conductivity, dispersion, nonlinear response, nonzero Bloch vector, symmetry
reduction, or PML overlap. Existing cylindrical scalar support stays unchanged;
tensor cylindrical gradients require a separate radial-transpose derivation.

## Implementation and verification

- [x] Establish baseline failures with installed-package provenance.
- [x] Add local inverse-tensor Jacobians and discrete D-field contractions.
- [x] Add anisotropic averaging and test homogeneous/scalar/rotated limits.
- [x] Connect all TDA and MultiTDA forward, adjoint, and FoM-only paths.
- [x] Verify two-step directional finite differences in 2D and 3D.
- [x] Verify dense/sparse history equivalence and MPI boundary cases.
- [x] Run existing scalar regression tests and independent source review.
- [x] Document supported scope, measured errors, and remaining limitations.

## Review focus

Gradient normal dependence at zero interpolation coefficients; singleton design
axes; tensor off-diagonal Yee offsets; forward/adjoint identical material
operators; MPI ghost validity; disabled global averaging; tensor projection
without double scaling by scalar material_factor; rejection of unsupported
complex/dispersive material derivatives.
