# Meep near-to-far, Mirror, and independent design regions

2026-10-07, `feat/anisotropic-material-averaging`, following commit `46009a8`.
Validation uses built, installed distributions in an isolated WSL environment:
Meep 1.34.0, Python 3.13.12, TAMA native API 14, and one OpenMP/BLAS thread per
MPI rank. No source-tree import overrides are used in the final regression run.

## Implemented behavior

- `Near2FarTarget(near_regions, frequencies, far_points)` accepts ordinary
  `mp.Near2FarRegion` and `mp.Vector3` objects. Meep performs the forward DFT,
  Green-function transform, and spatial pullback. TAMA supplies the finite-time
  Fourier transpose and its existing time-domain material gradient.
- `SimulationSpec(symmetries=[mp.Mirror(...)])` performs actual Meep domain
  reduction. Native sampling, indexed adjoint sources, integration weights, and
  density gradients account for vector parity and mirror planes.
- `designs=[design_a, design_b, ...]` updates independent, nonoverlapping
  `DesignGrid` blocks using one concatenated flat vector. Region sizes may
  differ; each gradient slice uses that region's material derivative. The
  existing `design=...` interface remains supported.
- `TDAObjective` and `MultiTDAObjective` retain one forward and one combined
  adjoint simulation per scalar objective gradient. The spectral MultiTDA mode
  permits separate surfaces, frequencies, far points, and callbacks per target.

The implementation does not call `meep.adjoint.OptimizationProblem` or replace
TAMA's time-domain adjoint with Meep's frequency-domain adjoint solver.

## Numerical checks

The percentages below are directional-gradient discrepancies against central
finite differences of the same finite-time discretized simulation. They are
not errors against an exact Maxwell solution or physical transmission errors.
Two FD steps are used to check step stability.

| Path | Coverage | Largest measured relative gradient discrepancy |
| --- | --- | --- |
| Near-to-far | 2D Ez/Hz, Courant 0.25/0.5, real/complex fields, custom E/H objectives, manual covectors, 3D anisotropic averaging, dense/sparse histories | 0.441524% |
| Mirror temporal targets | Both phases, magnetic components, two mirror planes, 3D normal electric component, tensor averaging, symmetric perturbations | 0.1782% |
| Independent regions | Unequal region sizes, TDA/MultiTDA, 2D/3D/cylindrical m=1, scalar/tensor, averaging OFF/ON, per-region perturbations | 0.161602% |

Near-to-far complex field arrays also match direct Meep forward calculations
within `rtol=2e-12`. The largest two-step FD relative change in the near-to-far
checks is about `1.56e-8`; the independent-region checks reach `5.08e-7`.
For 2D near-to-far gradients, reducing Courant from 0.5 to 0.25 reduces the
observed discrepancy from approximately 0.44% to 0.11%.

The combined two-region + near-to-far + Y-mirror test returns FoM
`0.007563588132874912`, adjoint directional derivative
`0.0001605180128146267`, and FD values
`0.00016088901922100413` / `0.00016088901958963286` at steps `1e-4` / `5e-5`.
The discrepancy is approximately 0.2306%. Full-domain and reduced-domain
values and symmetry-projected gradients are checked separately.

## Supported scope and explicit limits

- Near-to-far gradients support Cartesian 2D/3D at zero Bloch wavevector.
  Cylindrical near-to-far gradients and periodic-image sums are unsupported.
  Near surfaces must lie in a common homogeneous, isotropic, lossless exterior,
  with half a grid cell of clearance from PML/cell boundaries. The exterior
  material condition is a user precondition, not an automatic geometry proof.
- The default near-to-far objective is `0.5*sum(abs(E_far)**2)` over discrete
  points and frequencies. It is not integrated power, normalized transmission,
  or a uniform-wavelength band average. Its callback array is
  `(n_points, n_frequencies, 6)` in `Ex,Ey,Ez,Hx,Hy,Hz` order.
- Near-to-far always restarts the exact forward simulation for the adjoint.
  It requires direct `SimulationSpec` and restart-safe, time-invariant state.
  Adaptive chunk balancing is disabled. Fixed layouts work without mirrors;
  mirrors combined with explicit layouts are rejected as described below.
- Mirror supports Cartesian X/Y in 2D and X/Y/Z in 3D, with phases +/-1.
  Every design block, density, and endpoint tensor must be invariant under
  each reflection. Exchanging independent regions is unsupported. Gradients
  apply to symmetry-preserving perturbations; fixed geometry and sources must
  also respect the symmetry. Temporal Mirror gradients require direct
  `SimulationSpec` and exact indexed target sources.
  Adaptive balancing is disabled and explicit adaptive balancers are rejected;
  its moving-cut bounds still assume the full cell. Temporal targets retain
  fixed-layout support.
- With both `eps_averaging` and `MaterialGrid.do_averaging`, each mirrored
  design-grid axis must have even size or size one. Odd sizes greater than one
  put a one-sided interpolation-normal knot on the mirror plane and are
  rejected. Unaveraged odd grids remain supported.
- Multiple independent regions require distinct MaterialGrids and no overlap.
  Tensor/averaged cylindrical gradients remain outside the existing material
  support. Mirror uses regular target histories instead of the optional
  distributed unique-owner target-history path.
- Sparse near-to-far design histories keep full-rate DFT accumulation. Value
  and gradient evaluations use the same stride-aligned final time, which may
  extend by up to `sampling_interval-1` Meep steps. `last_actual_time` exposes it.
- Spectral MultiTDA targets cannot be mixed with temporal targets or PC-FIR
  wavelength-band settings. Core and native packages must both be rebuilt;
  API 14 rejects older native extensions.

## Defects caught by validation

Mirror source transposes needed component parity, mirror-plane integration
weights, and injection into Meep-owned redundant negative half-cells. The
3D normal-component test failed by approximately 25.4% before the source fix
and now differs from FD by approximately 0.1109%.

Near-to-far source pointers must retain their SWIG wrapper owners. Keeping
the wrappers alive fixes invalid metadata observed during Mirror source
folding. Pre-initialization dimensionality now uses Meep's own inference, so
a zero-z cell works without an explicit `dimensions=2` argument.

A plain-Meep MPI reproduction, without TAMA material preparation or adjoints,
found incorrect near-to-far surface integration with mirrors and explicit
binary chunk layouts. In one case the default-layout full/reduced FoM was
`0.1633500484330274`; the explicit reduced layout returned
`0.4760991164569727` and Meep warned `original_vol 42`, `looped 84`.
X=0, Y=0, and Y=0.3 partitions reproduced the issue. TAMA rejects this
combination rather than returning that result.

Independent review also identified missing sampling metadata in the spectral
MultiTDA constructor. The public `time_step()` path and runtime sampling-setting
propagation now have regression coverage.

MPI4 checks split the world into two independent two-rank Meep groups with
different densities and target locations. Default-layout mirrored near-to-far
and fixed-layout 3D mirrored point objectives match their full-domain
references and finite differences. The native source routing uses the active
Meep group, not `MPI_COMM_WORLD`. Rank-dependent monitor identities are
collectively rejected before per-point routing.

## Reproduction and artifacts

The full common/Meep suite passed **1,055 tests with 70 skips** on checkpoint6
in 338.65 seconds. The final checkpoint7 changes only add the adaptive-Mirror
guards and correct native-error test expectations; its installed distributions
passed **403 affected objective/near-to-far/region tests (3 skips)** and
**16 Mirror tests (5 MPI2 tests deselected)**. The full MPI2-marked suite then
passed **88 tests per rank** in 52.63 seconds. The MPI4 near-to-far and active
group regression selection passed **3 tests per rank**; checkpoint7 has the
same native source as that MPI4-verified build. Counts from overlapping runs
must not be added together.

Final wheel SHA256 values:

```text
tama-0.6.0a1-py3-none-any.whl
2d866a1b9bbdc4a9e519f51b5cecdb4eb23e6b692c7a65a8679741b11c4c0391
tama_meep_native-0.6.0a1-cp313-cp313-linux_x86_64.whl
7ca5e19274b193e3572bc4f725c97325db30577eb1b74119a6f8b4515efa00fa
```

Task environment: `/home/smrm/tmp/tama-meep-extensions-20261007/venv`.
The task directory holds frozen source manifests, build artifacts, serial/MPI
logs, per-feature FD logs, and the plain-Meep layout reproducer. Tests are in
`tests/meep/test_near2far.py`, `test_native_mirror.py`,
`test_multiple_designs.py`, `test_meep_extensions_integration.py`,
`test_mirror_target_safety.py`, and `test_extended_specs.py`.

The native source distribution includes `near2far_sources.hpp` and was built
back into a wheel. Documentation builds with Sphinx warnings treated as
errors. These local changes are not deployed to ReadTheDocs; deployment and
any manual rebuild must follow publication of the branch.

No speedup or peak-memory claim is inferred from these correctness tests.
The shared Meep installation, server optimization jobs, earlier validation
reports, and research-paper results were not modified.
