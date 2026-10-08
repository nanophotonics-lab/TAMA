# Objective Conventions and Targets

This guide describes the Meep backend. See [backend selection](backends.md) and [FDTDX](fdtdx.md) for the other engine.

## Objective Conventions and Weights

For a real scalar objective and a possibly complex callback history `u`, a
manual adjoint signal `q = adjoint_signal_fn(u, dt)` uses the bilinear-covector
convention

```text
delta FoM = dt * Re(sum(q * delta u)).
```

The sum covers time and every spatial/channel axis; there is no implicit
complex conjugation between `q` and `delta u`. `q` is the continuous-time
covector, so do not include another factor of `dt`. A flux-history covector must
be real. The same convention applies to `TDAObjective` and each band of
`MultiTDAObjective`.

`PointTarget.adjoint_source_amplitude` and
`FieldRegionTarget.adjoint_source_amplitude` are logical settings. Gradient
evaluation requires both to remain at `1.0`, because any other public scaling
would scale the returned derivative without changing the FoM. When the
adaptive Cartesian fallback expands a point source, TAMA applies the required
finite-source density normalization to the effective source internally.
Other finite real or complex settings are accepted only for value-only
evaluation, where no adjoint source is created.

The three weight-like inputs have different meanings:

- target `spatial_weights` act over physical sample points;
- `MultiTDAObjective.weights` multiply band-filter amplitudes and therefore
  enter quadratically for the built-in intensity, flux, and mode-power FoMs;
- scalarization `band_coeffs` are derivatives of the final scalar FoM with
  respect to the per-band FoMs, not additional filter or quadrature weights.

The optional `power_complementary` filter bank requires unit filter-amplitude
weights so that `sum_b |H_b(f)|^2` remains approximately one. Apply unequal
optimization priorities through `scalarization_fn`, not through `weights`, in
that mode.

In particular, for `FoM = S(J_1, ..., J_B)`, the coefficient for band `b` is
`c_b = dS/dJ_b`, and the returned gradient is
`sum_b c_b * gradient(J_b)`.

All target `spatial_weights` must be finite and nonnegative and must contain at
least one positive entry. Positions and geometric quadrature weights use the
same Meep length units as the simulation. Regional weights may include any
user-chosen objective measure. Cartesian flux/eigenmode weights represent a
line element in 2D or an area element in 3D; cylindrical surface weights include
the physical area Jacobian described below.

## Regional Field Targets

`FieldRegionTarget` exposes the field distribution at explicitly supplied
physical points. It uses the same native Yee-grid point sampling as
`PointTarget` and injects its exact transpose.

```python
regional_target = tm.FieldRegionTarget(
    positions=sample_positions,
    component=mp.Ez,
    sample_shape=(ny, nx),
    spatial_weights=quadrature_weights,
)
```

When omitted, `sample_shape` defaults to `(len(positions),)`. In the example,
the forward callback history has shape `(n_time, ny, nx)`. A custom `fom_fn`
and its `adjoint_signal_fn` receive and return this same shape. The default
objective is the weighted regional intensity
`0.5 * dt * sum(t,p) spatial_weights[p] * |u[t,p]|^2`; custom callbacks receive
the unweighted raw fields. Reshape the public flat weights as
`np.asarray(target.spatial_weights).reshape(target.sample_shape)` when using
them in a callback. Weights default to one. Supply `dx*dy`, `dV`, or cylindrical
`2*pi*r*dr*dz` weights explicitly when the objective uses those measures.

Regional gradients require the direct `SimulationSpec.make` path. Each spatial
sample receives its own time-domain adjoint waveform through the exact
transpose of the native Yee-grid point sampler. Arbitrary `sim_factory`
objects support value-only regional evaluation. Regional targets cannot
be combined with the legacy `monitor_position(s)`, `component`, or shared
adjoint-source override arguments.

The exact general regional path stores
`O(n_time * n_unique_monitor_channels)` values per MPI rank and creates
point-specific indexed source waveforms. In `MultiTDAObjective`, regional
samples with identical component, position, and adjoint-source settings share
one physical monitor history across bands; each band retains its own shape,
spatial weights, filter, and FoM. Yee support indices sharing one point
waveform, component, and local Meep chunk are packed into one `IndexedSource`,
but distinct regional point waveforms remain separate. Initial capture uses
`history_dtype`, resolved to its real counterpart when Meep uses real fields.
MultiTDA additionally creates full-history filter/adjoint temporaries and
reductions. Large regions can therefore cost substantially more memory and
preprocessing time than a `PointTarget`.

## Flux Targets

`FluxTarget` exposes a signed time-domain Poynting functional through explicitly
supplied surface points:

```python
flux_target = tm.FluxTarget(
    positions=surface_sample_positions,
    normal=mp.Vector3(1, 0, 0),
    spatial_weights=surface_quadrature_weights,
)
```

`normal` must be a signed axis unit vector. Cartesian 2D supports
x- and y-normal targets; Cartesian 3D supports x-, y-, and z-normal targets.
Meep cylindrical simulations support radial and axial normals. Radial samples
must lie at one positive radius; axial samples must lie at one z coordinate.
The configured normal sets the positive direction of power flow. Given `N`
raw samples, the public real power trace has length `N - 1` and is

```text
P[n] = sum(p) w[p] * Re{
    normal dot (conj(E[n,p]) cross 0.5 * (H[n,p] + H[n+1,p]))
}
```

The averaging centers each magnetic-field history onto the electric-field
time grid. `spatial_weights` defaults to one per point. Supply the surface
points and their line or area quadrature weights explicitly. Cylindrical
weights include the
full physical Jacobian exactly once: use `2*pi*r_i*dr_i` on an axial disk and
`2*pi*r0*dz_i` on a radial cylinder. An axial sample at `r=0` must have zero
weight.

For ordinary Cartesian real fields this is the sampled signed power through the
supplied quadrature. For complex fields, and especially a single cylindrical
`exp(i*m*phi)` field, it is a modal Poynting functional rather than an
automatically normalized total physical power. The supplied cylindrical
`2*pi*r` factor performs the azimuthal integral once. TAMA inserts no
additional phasor-style factor of `1/2`; apply any source, one-sided-mode, or
multi-`m` physical normalization externally.

The default `TDAObjective` FoM is `dt * sum(n) P[n]`. Custom `fom_fn` and
`adjoint_signal_fn` callbacks receive the same one-dimensional power trace.
For `MultiTDAObjective`, each band's temporal filter is applied independently
to the constituent E and H histories before the cross product is evaluated;
the default per-band FoM is the integral of that filtered signed-power trace.
Because `weights` are filter-amplitude weights, a flux band's power scales
quadratically with its configured weight, as does the default field-intensity
FoM.

Flux gradients require the direct `SimulationSpec.make` path. The constituent
E/H point histories and their adjoint sources use exact indexed Yee-point
sampler transposes. One gradient evaluation uses one forward run followed by
one combined electric/magnetic adjoint run. The flux pullback first distributes
each centered-H covector to its two raw H samples. Electric source profiles
end at `T+dt/2`, and raw magnetic source profiles end at `T+dt`, where `T` is
the forward recording end time. These offsets match Meep's staggered current
updates. No additional magnetic averaging is applied. MPI, adaptive chunk balancing, sparse Nyquist
design-history reconstruction, and
`reuse_simulation=True` are supported; flux monitor histories remain full
rate. A flux target stores four tangential-field histories per surface point;
their dtype follows Meep's real or complex field storage. Its in-memory monitor
storage is therefore approximately four times that of a one-component regional
target with the same points and duration.
MultiTDA additionally creates the corresponding filtered-history and adjoint
temporaries.

`FluxTarget` does not support azimuthal cylindrical normals,
`source_boundary_mode="finite"`, automatic surface-grid construction, or mode
decomposition. It computes time-domain point quadrature independently of
Meep's DFT `FluxRegion`.

## Eigenmode Coefficient Targets

`EigenmodeCoefficientTarget` isolates the field carried by one selected mode
instead of summing the power in every mode:

```python
mode_target = tm.EigenmodeCoefficientTarget(
    positions=surface_sample_positions,
    normal=mp.Vector3(1, 0, 0),
    mode_region=mp.Volume(
        center=port_center,
        size=mp.Vector3(0, port_width, port_height),
    ),
    frequency=reference_frequency,
    mode=1,
    spatial_weights=surface_quadrature_weights,
    eig_parity=mp.NO_PARITY,
)
```

For cylindrical coordinates, provide the already sampled fixed mode instead:

```python
cylindrical_mode_target = tm.EigenmodeCoefficientTarget(
    positions=radial_samples,
    normal=mp.Vector3(0, 0, 1),
    spatial_weights=2 * np.pi * radii * radial_quadrature_weights,
    reference_mode_fields={
        mp.Er: reference_er,
        mp.Hp: reference_hp,
        mp.Ep: reference_ep,
        mp.Hr: reference_hr,
    },
    reference_m=simulation_spec.m,
)
```

`reference_mode_fields` may also be supplied for a Cartesian target. That path
bypasses MPB and does not require `mode_region` or `frequency`; the mapping must
contain exactly both tangential E/H pairs, with zero arrays for a physically
absent pair. `reference_m` is specific to cylindrical fields. For a Cartesian
target without fixed fields, the explicit positions must lie in the
axis-aligned `mode_region`. On the first evaluation, TAMA uses Meep's
MPB-backed `get_eigenmode()` to solve the requested propagation direction and
stores only the sampled NumPy field profile. For mode fields `(E_m, H_m)`, its power
normalization and public
length-`N-1` coefficient history are

```text
P_m = sum(p) w[p] * Re{normal dot (conj(E_m[p]) cross H_m[p])}

a[n] = 1 / (2 * sqrt(|P_m|)) * sum(p) w[p] * normal dot {
    E[n,p] cross conj(H_m[p])
    + conj(E_m[p]) cross 0.5 * (H[n,p] + H[n+1,p])
}
```

The selected mode must carry positive nonzero power along `normal`. With this
normalization, a self-mode satisfies `|a|^2 = P_m`. The default objective is
`dt * sum(n) |a[n]|^2`; custom callbacks receive the complex coefficient
history.

`MultiTDAObjective` filters each constituent E/H history before applying the
fixed modal overlap, and each band may use its own reference frequency and
mode. The exact overlap transpose, magnetic-field time-centering transpose,
temporal-filter transpose, and indexed Yee-point transpose are all applied.
One gradient evaluation therefore uses one forward run followed by one
combined electric/magnetic adjoint run.

The implementation supports Cartesian 2D lines and 3D planes,
plus cylindrical radial or axial surfaces, with explicit quadrature points.
Meep does not solve cylindrical eigenmodes, so a cylindrical target supplies
`reference_mode_fields` and `reference_m` matching the forward simulation.
Use the same full `2*pi*r` quadrature weights described for `FluxTarget`. A
solver-backed Cartesian mode region must be
design-independent, reciprocal, dispersionless, and nonmagnetic; the cached
reference mode is not differentiated or recomputed after design
updates. Oblique ports, azimuthal cylindrical normals, degenerate-mode
subspaces, automatic port grids, and frequency-dependent broadband mode
profiles are not supported.

## Near-to-Far Targets

`Near2FarTarget` uses Meep's near-to-far transform at explicit frequencies and
far-field points. Supply ordinary Meep near surfaces:

```python
far_target = tm.Near2FarTarget(
    near_regions=near_surfaces,  # sequence of mp.Near2FarRegion objects
    frequencies=[0.6, 0.8],
    far_points=[mp.Vector3(10, 2), mp.Vector3(10, -2)],
)
problem = tm.TDAObjective(
    design=design, simulation=simulation, target=far_target, t_final=t_final,
)
value, gradient = problem.fom_and_grad(x)
far_fields = problem.last_far_fields
```

The callback argument and `last_far_fields` have shape
`(n_far_points, n_frequencies, 6)`, with components
`Ex, Ey, Ez, Hx, Hy, Hz` in Cartesian coordinates or
`Er, Ep, Ez, Hr, Hp, Hz` in cylindrical coordinates, in the supplied
point/frequency order. The default FoM
is `0.5 * sum(abs(far_fields[..., :3])**2)`. This is summed electric-field
intensity at discrete frequencies, not integrated radiated power, a normalized
transmission efficiency, or a PC-FIR wavelength-band energy. Supply an
autograd-compatible `fom_fn(far_fields, dt)` to choose components, weights, or a
different scalar objective.

Manual `adjoint_signal_fn` retains the existing convention
`delta FoM = dt * Re(sum(q * delta far_fields))`. Thus, for this target's
default FoM, a manual covector is zero in the H slots and
`conj(far_fields[..., :3]) / dt` in the E slots. The callback output must have
the same shape as `far_fields`.

Meep supplies the forward DFT, Green-function transform, and spatial source
pullback. TAMA transposes the finite-time Fourier sum into time-domain electric
and magnetic currents, then uses its native design-history gradient.
Near-field DFTs remain full rate
(`decimation_factor=1`); `sampling_interval` only reduces stored design-field
history.

For sparse design histories with stride `K`, both value-only and gradient
evaluations align the final DFT time to the same stored-history step. This may
extend the run by up to `K-1` Meep steps. Use a stride-aligned `t_final` when
comparing dense and sparse results over exactly the same time interval.
`last_actual_time` reports the completed DFT end time for either objective
class.

The near surfaces must be finite axis-aligned lines in 2D or planes in 3D,
with half a grid cell of clearance from PML and cell boundaries. They must
lie in the same homogeneous, isotropic,
lossless exterior medium used by the Green function. The far points must be
in that exterior region. Check this material condition when constructing the
simulation. TAMA does not inspect the exterior geometry.
Cartesian 2D/3D, cylindrical coordinates, and independent design regions are
supported. Cartesian simulations also support compatible mirror symmetries.
Nonzero Bloch wavevectors and periodic-image sums are not supported.

For cylindrical simulations, define radial or axial near-surface lines with
`mp.R` or `mp.Z` normals. Use `mp.Vector3(r, 0, z)` for all coordinates and
nonnegative radii. An axial cap may reach `r=0`; the axis needs no PML clearance.
Far-field gradients support `m=-1, 0, +1`, including mixed temporal and
near-to-far targets. The adjoint uses angular mode `-m`. Native API 16 applies
the radial quadrature and cylindrical Yee-cell volume. Explicit cylindrical
symmetries, tensor MaterialGrid media, and cylindrical subpixel averaging
remain unsupported by the design-gradient backend.

Near-to-far gradients require direct `SimulationSpec` use and restart-safe,
time-invariant simulation state. The adjoint always restarts the forward
simulation to preserve Meep's indexed-source chunk ownership, including when
`reuse_simulation=False`. DFT monitors are removed and sources replaced before
the adjoint run.
The default `chunk_balancer="auto"` is disabled for this target; explicit
adaptive balancers are rejected. A fixed `SimulationSpec.chunk_layout` is
supported without mirrors. Combining mirrors with an explicit chunk layout is
rejected. Mirrored near-to-far objectives use Meep's
default partitioning. All ranks retain the same forward/adjoint ownership.

`MultiTDAObjective(targets=[far_target_a, far_target_b], ...)` combines
per-target spectral objectives through its usual `scalarization_fn`, using one
forward and one combined adjoint simulation. Each target may have its own
surfaces, frequencies, points, and FoM. `last_far_fields` is then a tuple of
target arrays. An all-`Near2FarTarget` list does not accept wavelength bands or
temporal FIR settings.

Near-to-far and temporal targets may also share one objective:

```python
mixed = tm.MultiTDAObjective(
    design=design, simulation=simulation, t_final=t_final,
    targets=[far_target, tm.PointTarget(position=port_point, component=mp.Ez)],
    wavelength_bands=[None, (1.1, 1.8)],
    weights=[1.0, 1.0],
    kernel_length=301,
    scalarization_fn=lambda values: 0.25 * values[0] + 0.75 * values[1],
)
value, gradient = mixed.fom_and_grad(x)
far_fields = mixed.last_far_fields[0]
```

In a mixed list, `wavelength_bands` and `weights` have one entry per target.
Each near-to-far entry requires a `None` band and unit weight. Independent
temporal filters retain their existing amplitude-weight semantics; a
power-complementary bank requires all weights to be one. FIR settings apply
only to temporal entries. Choose objective priorities and any required
normalization in `scalarization_fn`; the spectral intensity and temporal band
energy are different quantities and are not normalized against each other.

Callback sequences, scalarization inputs, and `last_band_*` arrays retain the
complete target order. Near-to-far callbacks receive the spectral array above;
temporal callbacks receive their usual filtered histories. `last_far_fields`
contains only the near-to-far arrays, in their relative target order. Mixed
objectives use the same aligned recording endpoint and near-to-far constraints
described above.
