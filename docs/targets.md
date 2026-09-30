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
`PointTarget` and injects its exact transpose; no automatic volume grid, hidden
alignment, or extra interpolation layer is introduced.

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
them in a callback. Weights default to one; TAMA does not infer `dx*dy`,
`dV`, or a cylindrical `2*pi*r*dr*dz` measure.

Regional gradients require the direct `SimulationSpec.make` path. Each spatial
sample receives its own time-domain adjoint waveform through the exact
transpose of the native Yee-grid point sampler. Arbitrary `sim_factory`
objects remain available for value-only regional evaluation, but approximate
finite-volume regional adjoint sources are not used. Regional targets cannot
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
time grid. `spatial_weights` defaults to one per point. TAMA does not infer
a surface mesh, point spacing, line element, or area element; supply the
appropriate quadrature weights explicitly. Cylindrical weights include the
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
each centered-H covector to its two raw H samples. The resulting magnetic
adjoint waveform then uses the additional causal two-tap source representation
`0.5 * q(t) + 0.5 * q(t - dt)` so its adjoint field aligns with electric-source
design accumulation. MPI, adaptive chunk balancing, sparse Nyquist
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
decomposition. It is a time-domain point-quadrature target, not a
wrapper around Meep's DFT `FluxRegion`; bitwise equivalence to Meep DFT flux is
not claimed.

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

The implementation supports nonsymmetric Cartesian 2D lines and 3D planes,
plus cylindrical radial or axial surfaces, with explicit quadrature points.
Meep does not solve cylindrical eigenmodes, so a cylindrical target supplies
`reference_mode_fields` and `reference_m` matching the forward simulation.
Use the same full `2*pi*r` quadrature weights described for `FluxTarget`. A
solver-backed Cartesian mode region must be
design-independent, reciprocal, dispersionless, and nonmagnetic; the cached
reference mode is intentionally not differentiated or recomputed after design
updates. Oblique ports, azimuthal cylindrical normals, degenerate-mode
subspaces, automatic port grids, and frequency-dependent broadband mode
profiles are not supported. This is a fixed-reference time-domain overlap,
not a wrapper around Meep's DFT `get_eigenmode_coefficients()`.
