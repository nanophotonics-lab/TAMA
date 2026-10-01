# FDTDX backend

The backend calls native FDTDX kernels without patching the engine. Installation
pins retain FDTDX 0.6.2 as the tested environment. Before constructing objectives, the cached
`verify_fdtdx_compatibility()` check verifies the required callable FDTDX APIs
and Equinox internal `while_loop`. API availability alone does not validate
changed numerical behavior; dependency updates require numerical regression
validation.

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
voxels onto the native grid. Native Device material mappings and
`param_transforms` are not silently adopted. Supply a pure differentiable JAX
`transform` explicitly when using the periodic conic filter or tanh projection.
The transform must preserve shape and density bounds. Hard thresholds and
incorrect custom derivatives are not validated design gradients.

## Observations and objectives

Indices are native Cartesian integer triples. E/H are spatially interpolated
using the engine's native operation; adjacent H time steps are averaged to
align with E. Callbacks receive `(Nt, N, 6)` with component order
`Ex, Ey, Ez, Hx, Hy, Hz`. Do not average H a second time.

| Target | Default scalar objective |
|---|---|
| PointTarget | `dt/2 * sum(abs(component)**2)` |
| FieldRegionTarget | Spatially weighted component energy |
| FluxTarget | `dt * sum(weights * Re(E cross conj(H)) dot normal)` |
| EigenmodeCoefficientTarget | `dt * sum(abs(a)**2)` for fixed profiles |

Area/volume weights must be supplied explicitly. FDTDX uses normalized H, so
flux is not automatically a quantity in watts. No phasor half-factor or incident
power normalization is silently inserted. Mode overlap uses
`P = 0.5 * sum(w * Re(E_mode cross conj(H_mode)) dot n)` and
`a = sum(w * (E cross conj(H_mode) + conj(E_mode) cross H) dot n) / (4P)`.
P must be positive. This is not Meep's power-normalized coefficient. Profiles
must match native field units/sampling and are not certified as eigenmodes.

`bandpass_kernel(low_hz, high_hz, dt, taps)` uses Hz and seconds. Band applies
the dt convolution factor and uses the full `Nt+K-1` output. `Band.weight`
multiplies the objective, not the filter amplitude. Scalarization sees already
weighted band values. `fom_fn(history, dt)` must return a finite real JAX scalar
with a correct derivative. `validate` checks the actual callback and design at
the supplied density; it does not prove accuracy at every possible density.

## Broadband, sparse histories and offload

Multiple bands share the forward FDTD calculation and a scalarized adjoint.
Target histories and FIR processing still cost memory/work as bands or
observations grow. This structure is independent of sparse-history acceleration.

`sampling_interval=M` saves the design electric-field history every M FDTD time
steps. The FDTD step and target sampling are unchanged. M=1 is the default exact
discrete adjoint path. For M>1, each evaluation runs only the requested sparse
stride, without a dense reference comparison. Its gradient approximation error
depends on the sampling interval and reconstruction.

**CPU offload defaults to OFF** (`cpu_offload=False`). On CUDA,
`cpu_offload=True` stores design histories in pinned CPU memory; `False` stores
them on the GPU. The example CLI also defaults to `--cpu-offload off`.
FDTD, reconstruction and gradient contraction stay on the GPU.
Target traces and their cotangents remain on device. Transfers are not an
asynchronous copy/compute overlap implementation. History bytes are not peak
process VRAM. Fair performance measurements separate compile and warm execution
times and measure GPU/host peaks in independently initialized processes.

## Supported and excluded configurations

Supported validation lanes include native Cartesian 3D and thin periodic cells,
PML/PEC/PMC/complex Bloch boundaries, uniform/gaussian plane sources and point
dipoles, real32/64 and complex64/128 fields, four target types, FIR bands, fixed
mode profiles, fabrication transforms and single-device CPU-history offload.

The current design is one rectangular scalar nondispersive lossless epsilon
region, separated from sources/PML/clamping. Conductivity/ADE/tensor media,
design-dependent mu/source/boundaries/modes, cylindrical coordinates, arbitrary
unvalidated subclasses, multiple independent design regions, multi-GPU and disk
offload are not implemented. Rejection guards are part of the accuracy contract.

The runnable example is
[`examples/fdtdx/optimize_broadband.py`](https://github.com/nanophotonics-lab/TAMA/blob/main/examples/fdtdx/optimize_broadband.py);
its native scene is defined in the adjacent `native_scene.py`. See
[installation](installation.md) for the environment and
[running and testing](running-and-testing.md) for execution commands.
Physical equivalence with Meep
requires matched units, sources, normalization and mesh/time/domain convergence;
native AD agreement alone does not establish that equivalence.
