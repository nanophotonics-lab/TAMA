# Meep-derived smoothing and anisotropy validation

2026-10-07, `feat/anisotropic-material-averaging`, following implementation
commit `952a9c8`. Tests use the isolated task environment with Meep 1.34.0,
Python 3.13.12, TAMA native API 13, and one thread per MPI rank.
The shared Meep installation and running server optimizations were not changed.

## Upstream sources and adaptations

All sources are pinned to Meep v1.34.0, commit
`3e7b7fee0da4a9b91b07acd9f71d35d674396563`.

- [test_material_grid.py](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/python/tests/test_material_grid.py):
  circle and sphere geometry, source, Bloch vector, resolution, reference, and
  original acceptance criteria retained. The unchanged upstream tests also pass,
  including the circle's default-material variant. TAMA adaptations use a Block.
  Stock, TAMA-prepared scalar, and near-isotropic (`epsilon_xy=1e-6`) paths run
  separately. The latter asserts installation of the custom material operator.
- [test_adjoint_solver.py, test_offdiagonal](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/python/tests/test_adjoint_solver.py):
  unchanged single/multiple-frequency test passes. TAMA retains its SPD tensor,
  geometry, resolution 30, 91x91 density grid, conic filter, seed 9861548,
  perturbation scale `1e-5`, and single-frequency tolerance `0.002`.
  TAMA uses point-field time-integrated energy instead of eigenmode coefficients,
  bandwidth `0.2*fcen` instead of `0.05*fcen`, time 120, and stride 2.
  The ON extension adds beta 8. Both central-FD steps are checked for stability.
- [aniso_disp.cpp](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/tests/aniso_disp.cpp):
  static tensor from the analytic-reference comment, k=(0.813,0,0), resolution
  200, and relative frequency tolerance
  `1e-4` retained. The Lorentz response is removed and NEW nondispersive
  frequencies are calculated from the Maxwell eigenproblem. A one-cell nonzero
  z extent prevents Python's automatic 2D reduction; actual 3D is asserted.
- [meepgeom.cpp](https://github.com/NanoComp/meep/blob/3e7b7fee0da4a9b91b07acd9f71d35d674396563/src/meepgeom.cpp):
  an additional independent interface check uses Meep's geometric Block average
  as the oracle, with two full SPD tensors and normal (0.36,0.48,0.8).
  Each interface bisects the coefficient's actual Yee averaging voxel.
  All three diagonal and six ordered off-diagonal entries are tested. At half
  fill, geometric box and MaterialGrid sphere kernels have the same fill factor;
  off-center fills are deliberately not equated.

These Bloch-wave/resonance tests exercise forward material setup only.
They do not add nonzero-Bloch adjoint support or validate dispersive gradients.

## Results on the corrected TAMA build

| Check | Measured result | Acceptance |
| --- | --- | --- |
| Circle smoothing | Frequency 0.296990792848 at res25, 0.297692634613 at res50; reference 0.298268138732 | Errors 0.428254% to 0.192949%; reduction 2.2195x, exceeding upstream's 2x criterion |
| Sphere smoothing, res25 | TAMA 0.783851214225; geometric Sphere 0.787431038770 | Absolute difference 0.00357982, passing upstream's two-decimal-place criterion; relative difference 0.454621% |
| Homogeneous anisotropic wave | Observed 0.379416830177 / 0.716733104205; analytic 0.379426581861 / 0.716748800876 | Maximum relative error 0.00257011%, below 0.01%; direct Medium and TAMA OFF/ON pass |
| Oblique anisotropic interface | Maximum coefficient difference about 1e-15; stock anisotropic grid is a failing negative control | `rtol=1e-10`, `atol=1e-12`; negative-control difference exceeds 0.01 |
| Upstream-derived TAMA gradient | OFF error 0.0479646%, ON 0.00823544% | Both below 0.2%; both FD step comparisons stable |

The circle reference is a resolution-300 numerical result, not an analytic
exact solution. Its criterion demonstrates better-than-linear convergence in
these two resolutions, not a general proof of second-order convergence.
The sphere test compares different representations in the same solver at one
resolution and has a relatively loose original tolerance.

## Defect found and corrected

Meep 1.34.0's 3D MaterialGrid fallback kernel uses integer `4/3` in its sphere
volume normalization. A separate weak-contrast runtime probe confirmed the
defect: averaged/unaveraged epsilon ratio 1.058823085 in 3D, versus
0.999999719 in 2D for that probe.

Before this correction, TAMA's scalar 3D path delegated to that kernel while its
anisotropic path used a normalized kernel. The sphere frequencies differed by
`0.000430017783` when adding only `epsilon_xy=1e-6`. The original upstream sphere
tolerance missed this discontinuity. The new continuity regression failed.

TAMA now routes scalar 3D MaterialGrid averaging through its existing normalized
tensor evaluator as well. Forward initialization, endpoint validation, material
Jacobian selection, and asymmetric-projection checks use the same condition.
After rebuilding the isolated Python/native pair, the frequency difference is
`2.713633207e-9`, passing the unchanged new threshold `1e-7`.
The 2D scalar path and the averaging-disabled paths remain unchanged.

This intentionally changes scalar 3D averaged results relative to stock Meep
1.34.0. It does not guarantee a smaller error for every coarse geometry:
the sphere's stock numerical difference is 0.400011%, versus corrected TAMA's
0.454621%. Kernel normalization and cross-path continuity are corrected;
spatial representation errors remain and can cancel at individual resolutions.

## Additional time-step accuracy limit

A scalar 3D design with only one z density node exposed a near-cancelling
directional derivative. At Courant 0.25 (dt=0.025), the dense adjoint/FD error is
1.039503%, exceeding the existing 0.5% gradient criterion. Sparse storage is not
the cause: the dense result already fails, and stride 2 gives 0.949063%.

With the same density, direction, spatial mesh, and simulation duration:

| Courant | Dense absolute gradient error | Dense relative error |
| --- | ---: | ---: |
| 0.25 | 1.5350354e-6 | 1.039503% |
| 0.125 | 3.9638560e-7 | 0.224435% |
| 0.0625 | 1.0000040e-7 | 0.054389% |

Absolute error falls by 3.87x and 3.96x when dt halves, consistent with
second-order temporal error. Cancellation amplifies its relative effect:
`sum(abs(g_i*d_i))/abs(g.d)` is 150.1 at Courant 0.25.
The permanent singleton fixture therefore uses Courant 0.125 and retains the
0.5% tolerance. No production/default Courant setting was changed. The failure
and refinement data are preserved rather than presented as a universal 0.5%
accuracy guarantee.

## Reproduction and evidence

Final verification: 16 upstream-derived/independent cases pass, plus 34 material
operator/gradient cases, 376 objective/sampling cases, 127 native scalar/3D/
cylindrical cases, and 64 common/native safety cases. The two selected MPI2
boundary cases pass on both ranks. The unchanged upstream material-grid suite
has 2 selected tests passing, and its off-diagonal adjoint test passes both
single- and multiple-frequency configurations. Earlier coarse singleton failure
is retained in `gradient_coarse_failure.log`; final 34-case results are in
`gradient_final.log`.

The permanent tests are `tests/meep/test_upstream_*.py` and the extended
`tests/meep/test_native_material_tensor.py`. Run with the matching installed
Python/native pair, `PYTHONPATH` unset, and OMP/BLAS/MKL thread counts set to 1:

```bash
python -m pytest -q tests/meep/test_upstream_*.py
python -m pytest -q tests/meep/test_native_material_tensor.py -m 'not mpi2'
mpirun -launcher fork -np 2 python -m pytest -q -p no:cacheprovider \
  tests/meep/test_native_material_tensor.py -m mpi2
```

Raw logs, pinned upstream sources, build-source hashes, and the full time-step
probe are in `/home/smrm/tmp/tama-anisotropic-20261007/upstream_validation`.
The final evidence inventory is `result_manifest.json` in that directory.
Documentation builds locally with Sphinx warnings treated as errors. These local
branch changes have not been deployed to ReadTheDocs; no live rebuild occurred.
