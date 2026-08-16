# TDAObjective

## Example

Minimal shape of a TAMA time-domain adjoint setup:

```python
design = tm.DesignGrid(
    material_grid=design_variables,
    center=design_center,
    size=design_region_size,
    shape=(nx, ny),
    background=air,
    design_material=design_material,
)

simulation = tm.SimulationSpec(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=fwd_sources,
    resolution=resolution,
    geometry_center=geometry_center,
    eps_averaging=False,
)

target = tm.PointTarget(
    position=monitor_position,
    component=mp.Ez,
    adjoint_source_size=adjoint_source_size,
    adjoint_source_amplitude=1.0,
)

# A regional objective may be substituted without changing the driver:
# target = tm.FieldRegionTarget(
#     positions=sample_positions,
#     component=mp.Ez,
#     sample_shape=(ny, nx),
#     spatial_weights=quadrature_weights,
# )


tda = tm.TDAObjective(
    design=design,
    simulation=simulation,
    target=target,
    t_final=T_f,
    resolution=resolution,
)

fom, gradient = tda.fom_and_grad(x)
```

`TDAObjective` default objective is:

```text
0.5 * integral |u_monitor(t)|^2 dt
```

Here, `u_monitor` is the selected electric or magnetic field component. A
`FieldRegionTarget` instead sums the configured spatial weights times the
squared magnitude of the regional field samples. A `FluxTarget` instead uses
`integral P(t) dt`, where `P` is the signed power trace defined in the
[`FluxTarget` section](targets.md#flux-targets). An
`EigenmodeCoefficientTarget` uses `integral |a(t)|^2 dt`. For a custom scalar
objective, pass `fom_fn(monitor_history, sample_dt)`. The history is the
selected field history for point and regional targets, the real length-`N-1`
power trace for a flux target, or the complex length-`N-1` modal coefficient
for an eigenmode target. Use autograd-compatible
operations, or pair it with `adjoint_signal_fn` when its derivative must be
supplied manually.

## Inputs

`TDAObjective` follows the same style as Meep objects: users pass ordinary Meep
objects and explicit physical regions rather than hidden configuration.

| Input | Meaning |
| --- | --- |
| `design` | `DesignGrid` describing the optimized MaterialGrid. Required when computing a gradient; optional for value-only evaluation. |
| `simulation` | Optional `SimulationSpec`. Fills `sim_factory` and `resolution`; its native `DesignGrid` use requires `eps_averaging=False`. Its direct `make` path enables exact indexed target transposes and is required for regional, flux, and eigenmode gradients. It is also required for cylindrical gradients and supplies their forward mode `m`. |
| `reuse_simulation` | Reuse the forward `Simulation` for the adjoint run when explicitly set to `True`. Default is `False`; enable only for restart-safe, time-invariant media. A cylindrical reuse changes the mode from `m` to `-m`. |
| `target` | Optional `PointTarget`, `FieldRegionTarget`, `FluxTarget`, or `EigenmodeCoefficientTarget`. Point targets fill the legacy point/source settings. Regional target histories use time as the first axis and `sample_shape` as the remaining axes. Flux and eigenmode targets expose length-`N-1` signed-power or complex modal histories and combine their electric and time-centered magnetic sources in one adjoint run. Their gradients require direct `SimulationSpec` use. Cylindrical surface targets use radial or axial normals and explicit `2*pi*r` physical quadrature; cylindrical eigenmode targets additionally require fixed reference fields and a matching `reference_m`. |
| `update_design` | Function that writes the flat design vector into the Meep design object. Usually calls `MaterialGrid.update_weights(...)`. |
| `t_final` | Forward simulation end time. |
| `sim_factory` | Function returning `mp.Simulation`. Called with no argument for the forward run and with an adjoint source list for the adjoint run. Arbitrary factories use ordinary Cartesian `mp.Source` injection and do not support cylindrical gradient evaluation. |
| `monitor_position` | Physical point where the forward monitor signal is sampled and where the adjoint source is placed. The indexed path injects the exact transpose of the native Yee-grid point interpolation. Cylindrical targets use `(r, 0, z)` with `r >= 0`, may lie on or near `r=0`, and include modal axis parity in the transpose. |
| `component` | Meep monitor/source component. The native path maps 2D TMz to `Ez`, 2D TEz to `Ex`/`Ey`, 3D scalar-isotropic designs to `Ex`/`Ey`/`Ez`, and cylindrical targets according to the [design-gradient component table](design-gradients.md#supported-components). |
| `adjoint_source_size` | Meep source size for the adjoint source. Defaults to `mp.Vector3()`, which selects exact indexed injection for direct `SimulationSpec` use. An explicit nonzero Cartesian size or `source_boundary_mode="finite"` selects ordinary `mp.Source`; eligible axes may then be regularized if layout-first placement fails. Cylindrical gradients require zero size. |
| `adjoint_source_amplitude` | Logical adjoint-source amplitude. Gradient evaluation requires `1.0`; Cartesian adaptive finite-source fallback applies its effective density normalization internally. Other finite values are accepted only for value-only evaluation. |
| `fom_fn` | Optional scalar objective `fom_fn(monitor_history, sample_dt)`. Point history has shape `(time,)`; regional history uses time as the first axis and `sample_shape` as the remaining axes; flux history is a real `(time - 1,)` signed-power trace; and eigenmode history is a complex `(time - 1,)` coefficient. |
| `adjoint_signal_fn` | Optional manual continuous-time bilinear covector described in the [objective convention](targets.md#objective-conventions-and-weights). Requires `fom_fn` and must return the same shape as the callback history. A flux-history covector must be real. |
| `dt` | Legacy expected Meep time step. The initialized simulation's `fields.dt` is authoritative; an explicit value must match it and is otherwise only a fallback for simulation doubles without `fields.dt`. |
| `resolution` | Used with the simulation's Courant factor to infer a fallback time step only when the initialized simulation does not expose `fields.dt` and `dt` is omitted. |
| `sampling_interval` | Dimensionless stride in Meep steps between stored design-grid forward-field samples. Point, regional, and constituent flux/eigenmode field histories remain full rate. Default is `1`. |
| `max_frequency` | Optional maximum relevant frequency in Meep inverse-time units. Computes the design-grid interval from the Nyquist criterion; cannot be combined with `sampling_interval > 1`. |
| `reconstruction_window` | Window applied to the finite sinc reconstruction. Canonical names and accepted aliases are listed in the [sparse reconstruction section](multi-tda-objective.md#sparse-design-history-reconstruction). Default is `kaiser`. |
| `reconstruction_window_params` | Optional reconstruction-window parameters. Only `kaiser` accepts `{"beta": value}`; its default beta is `6`. |
| `reconstruction_half_width` | Number of sparse samples on each side of the finite sinc support. Default is `64`. |
| `history_dtype` | Real or complex floating dtype requested for temporary histories. Default `np.complex128` resolves to `np.float64` when Meep uses real forward fields; complex forward fields require a complex dtype. Native design histories are rank-local disk-backed memory maps, while target histories remain in memory. |
| `chunk_balancer` | Default `"auto"` creates an `AdaptiveAdjointChunkBalancer` for a direct `SimulationSpec` with `chunk_layout=None`. It automatically protects ordinary forward sources, preserves Meep's Cartesian or cylindrical R-Z topology, and calibrates over three to eight gradient evaluations before freezing the best measured safe layout. Pass `None` to opt out or an instance for custom settings. Indexed sources are excluded from geometric constraints. Regional, flux, eigenmode, and cylindrical indexed sources do not support `source_boundary_mode="finite"`. |

`TDAObjective` uses the same sparse Nyquist design-history path described in
the [sparse reconstruction section](multi-tda-objective.md#sparse-design-history-reconstruction); its point, regional, and constituent flux/eigenmode field histories remain full
rate.
