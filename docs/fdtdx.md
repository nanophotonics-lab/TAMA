# FDTDX backend

The backend calls native FDTDX kernels without patching the engine. Installation
pins retain FDTDX 0.6.2 as the tested environment. Before objective construction,
the cached `verify_fdtdx_compatibility()` check verifies required callable FDTDX
APIs and Equinox internal `while_loop`. Dependency updates require numerical
regression tests.

## Native scene and density

Construct FDTDX geometry, source and boundary objects, call `place_objects` and
`apply_params`, then create the TAMA specification:

```python
import tama
tm = tama.get_backend("fdtdx")
spec = tm.SimulationSpec(arrays, objects, config, key)
design = tm.DesignGrid.from_device(
    objects["design"], epsilon_low=1.0, epsilon_high=4.0,
)
design.validate(spec)
```

`apply_params` must initialize native source caches. The design uses
`epsilon = epsilon_low + (epsilon_high - epsilon_low) * rho`, repeating coarse
voxels onto the native grid. This map replaces native Device material mappings.
Supply periodic conic filtering or tanh projection through a pure differentiable
JAX `transform`. `DesignGrid.from_device()` requires empty native
`param_transforms`. The transform must preserve shape and density bounds.
Avoid hard thresholds and supply correct custom derivatives.

## Observations and objectives

Indices are native Cartesian integer triples. The engine interpolates E/H in
space. The backend averages adjacent H time steps to align with E.
Callbacks receive `(Nt, N, 6)` with component order
`Ex, Ey, Ez, Hx, Hy, Hz`. Do not average H a second time.

| Target | Default scalar objective |
|---|---|
| PointTarget | `dt/2 * sum(abs(component)**2)` |
| FieldRegionTarget | Spatially weighted component energy |
| FluxTarget | `dt * sum(weights * Re(E cross conj(H)) dot normal)` |
| EigenmodeCoefficientTarget | `dt * sum(abs(a)**2)` for fixed profiles |

Supply area/volume weights explicitly. FDTDX uses normalized H, so flux is not
automatically in watts. Flux includes neither a phasor half-factor nor incident
power normalization. Mode overlap uses
`P = 0.5 * sum(w * Re(E_mode cross conj(H_mode)) dot n)` and
`a = sum(w * (E cross conj(H_mode) + conj(E_mode) cross H) dot n) / (4P)`.
P must be positive. The coefficient differs from Meep's power-normalized
coefficient. Supply fixed profiles in native field units and sampling.
The backend does not solve or verify eigenmodes.

`bandpass_kernel(low_hz, high_hz, dt, taps)` uses Hz and seconds. Band applies
the dt convolution factor and uses the full `Nt+K-1` output. `Band.weight`
multiplies the objective, not the filter amplitude. Scalarization sees already
weighted band values. `fom_fn(history, dt)` must return a finite real JAX scalar
with a correct derivative. `validate` compares the dense `sampling_interval=1`
adjoint with full-state AD and centered finite differences for the callback and
design at the supplied density. The check covers neither the chosen sparse
interval nor other densities.

## Broadband, sparse histories and offload

Multiple bands share the forward FDTD calculation and a scalarized adjoint,
independently of sparse sampling. Target histories and FIR processing cost more
memory and work as bands or observations grow.

`sampling_interval=M` saves the design electric-field history every M FDTD time
steps. The FDTD step and target sampling are unchanged. M=1 is the default exact
discrete adjoint path. For M>1, each evaluation runs only the requested sparse
stride, without a dense reference comparison. Its gradient approximation error
depends on the sampling interval and reconstruction.

Both `block_steps` and `SimulationSpec.time_steps` must be divisible by M.
The default `block_steps` is 64. For example, M=3 requires a different block
size and a simulation step count divisible by 3.

**CPU offload defaults to OFF** (`cpu_offload=False`). On CUDA,
`cpu_offload=True` stores design histories in pinned CPU memory. `False` stores
them on the GPU. The example CLI also defaults to `--cpu-offload off`.
FDTD, reconstruction and gradient contraction stay on the GPU.
Target traces and their cotangents remain on device. Transfers do not overlap
copies and computation. Measure peak GPU/host usage separately from history
bytes, using independently initialized processes. Separate compilation time
from warm execution time.

## Supported and excluded configurations

Supported validation lanes include native Cartesian 3D and thin periodic cells,
PML/PEC/PMC/complex Bloch boundaries, uniform/gaussian plane sources and point
dipoles, real32/64 and complex64/128 fields, four target types, FIR bands, fixed
mode profiles, fabrication transforms and single-device CPU-history offload.

The current design is one rectangular scalar nondispersive lossless epsilon
region, separated from sources/PML/clamping. Conductivity/ADE/tensor media,
design-dependent mu/source/boundaries/modes, cylindrical coordinates, arbitrary
unvalidated subclasses, multiple independent design regions, multi-GPU and disk
offload are not implemented.

The runnable example is
[`examples/fdtdx/optimize_broadband.py`](https://github.com/nanophotonics-lab/TAMA/blob/main/examples/fdtdx/optimize_broadband.py).
Its native scene is defined in the adjacent `native_scene.py`. See
[installation](installation.md) for the environment and
[running and testing](running-and-testing.md) for execution commands.
For physical comparisons with Meep, match units, sources, and normalization,
and check mesh/time/domain convergence.
