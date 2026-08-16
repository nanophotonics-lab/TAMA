# Design Gradients and Grid Coordinates

Use `centered_grid_coords` when a low-level field-sampling task needs a regular
coordinate grid centered on a Meep region.

```python
coords_x, coords_y = tm.centered_grid_coords(
    center=design_center,
    shape=(nx, ny),
    spacing=(dx, dy),
)
```

These coordinates can be passed directly to `FastFieldGrid` or
`FastGradientGrid`. Objective gradients instead require `DesignGrid`.
The two low-level grid classes require an initialized, nonsymmetric 2D
Cartesian simulation and sample a regular point grid; they are not the native
MaterialGrid-gradient interface used by the objective classes.

When `DesignGrid` is supplied, TAMA instead discovers the native Yee sites
for every electric component required by the design gradient. The user supplies
the physical center and size of the design region without component-specific
shifts or pixel grouping. During the adjoint run, each Yee-site contribution is
scattered into the raw MaterialGrid weights through the discrete transpose of
Meep's MaterialGrid coordinate map.

The native path currently requires:

- a 2D or 3D Cartesian simulation, or a 2D Meep cylindrical simulation;
- exactly one MaterialGrid geometry object, used by one matching,
  axis-aligned `mp.Block`;
- matching `DesignGrid` and Block center, size, shape, and dimensionality;
- `eps_averaging=False`, no Meep symmetries, and either Meep's default
  `k_point=False` metallic boundaries or normal-incidence Gamma-point periodic
  boundaries selected explicitly with `k_point=mp.Vector3()`;
- a design Block that does not overlap PML or absorber layers;
- MaterialGrid `do_averaging=False`, `beta=0`, `damping=0`, and no overlapping
  grid combination;
- two linear, scalar-isotropic, nondispersive endpoint media without electric
  or magnetic susceptibilities, nonlinearities, or conductivity;
- equal, fixed permeability in the two endpoint media;
- `material_factor` equal to the endpoint permittivity contrast; and
- identical rank-local Yee layouts in the forward and adjoint simulations.

## Supported Components

Supplying `DesignGrid` selects the following electric components from the
simulation dimensionality and monitored polarization:

| Simulation / monitor polarization | Electric components used for the design gradient |
| --- | --- |
| 2D TMz (`Ez`, `Hx`, or `Hy`) | `Ez` |
| 2D TEz (`Hz`, `Ex`, or `Ey`) | `Ex` and `Ey` |
| 2D Cartesian `FluxTarget` or `EigenmodeCoefficientTarget` | `Ex`, `Ey`, and `Ez` because both tangential E/H pairs are sampled |
| 3D scalar-isotropic permittivity (any single `Ex`/`Ey`/`Ez`/`Hx`/`Hy`/`Hz` target) | `Ex`, `Ey`, and `Ez` |
| Cylindrical `m=0` with an `Ep`/`Hr`/`Hz` target | `Ep` |
| Cylindrical `m=0` with an `Er`/`Ez`/`Hp` target | `Er` and `Ez` |
| Cylindrical `m=0` `FluxTarget` or `EigenmodeCoefficientTarget` | `Er`, `Ep`, and `Ez` because both tangential E/H pairs are sampled |
| Cylindrical `m=±1` with any `Er`/`Ep`/`Ez`/`Hr`/`Hp`/`Hz` target | `Er`, `Ep`, and `Ez` |

## Cylindrical Coordinates and Angular Modes

For a Meep cylindrical simulation, use
`DesignGrid(coordinate_system="cylindrical", shape=(nr, nz))` with a
`MaterialGrid` whose grid size is `mp.Vector3(nr, 1, nz)`. The matching
`SimulationSpec` must use `dimensions=mp.CYLINDRICAL` and an integer `m`.
Cylindrical gradient evaluation currently supports `m=-1`, `0`, and `+1` and
requires `simulation=SimulationSpec(...)`; an arbitrary `sim_factory` is not
sufficient because TAMA must construct the adjoint simulation with mode
`-m`.

Both objective classes support `PointTarget`, `FieldRegionTarget`,
`FluxTarget`, and `EigenmodeCoefficientTarget` in this coordinate system.
The two surface targets are restricted to radial or axial surfaces and use
explicit physical quadrature, as detailed in the [target guide](targets.md).

Each `TDAObjective` or `MultiTDAObjective` evaluates one angular mode. The
`Multi` in `MultiTDAObjective` refers to wavelength bands, not angular modes.
Meep represents that mode by complex amplitudes multiplying `exp(i*m*phi)`.
The adjoint simulation uses `-m` because the angular part of the bilinear
forward/adjoint pairing must cancel; it is not a second forward-mode
evaluation. Returned field, flux, and mode-overlap values are therefore
single-`m` modal quantities under the supplied source and quadrature
normalization. Run separate forward objectives and apply the required physical
normalization and combination externally when a real three-dimensional result
needs contributions from multiple values of `m`.

Cylindrical gradients use the same linear, scalar-isotropic, nondispersive
permittivity restrictions listed above, including fixed and equal endpoint
permeability. The design region may touch `r=0` but must not extend below it.
Point and regional targets use zero-size indexed adjoint sources. Their samples
may lie on or near `r=0`; the native transpose folds each Yee interpolation
entry through Meep's cylindrical modal parity. Exactly on the axis, regularity
allows only these components:

| Forward mode | Nonzero point-target components at `r=0` |
| --- | --- |
| `m=0` | `Ez`, `Hz` |
| `m=±1` | `Er`, `Ep`, `Hr`, `Hp` |

For `m=±1`, the axis relations are `Ep = i m Er` and `Hp = i m Hr`.
Components constrained to zero are rejected when a gradient is requested.
Meep symmetries are not supported by TDA or MultiTDA objectives in any
coordinate system. Custom `sim_factory` results must therefore use
`symmetries=[]`.
The default `AdaptiveAdjointChunkBalancer` supports Meep's native cylindrical
R-Z topology with zero-size indexed adjoint sources.

Both objective classes support the sparse Nyquist reconstruction path for
these cylindrical modes. `reuse_simulation=True` is also supported for
restart-safe, time-invariant media; the reused simulation changes from `m` to
`-m` before the adjoint run. See
[`examples/tda_cylindrical_mode.py`](../examples/tda_cylindrical_mode.py) for a
runnable `m=+1` objective with an axis-touching design region. Its `--target`
option selects `point`, `regional`, `flux`, or `eigenmode`:

```bash
python examples/tda_cylindrical_mode.py --target point
mpirun -np 2 python examples/tda_cylindrical_mode.py --target eigenmode
```

## Cartesian 3D Gradients

In 3D, each electric component is sampled at its own staggered Yee sites and
scattered through the trilinear MaterialGrid transpose into one shared scalar
permittivity gradient. This is a full-vector field contraction, not support for
anisotropic or tensor MaterialGrid weights. The spatial scatter, including
Meep's component-specific Yee-cell integration weights, is the discrete
transpose of Meep's coordinate map. The complete gradient still includes the
time-discretization and adjoint-source approximations of the time-domain
method.

## Gradient Output, Storage, and Inputs

The returned native gradient is with respect to raw MaterialGrid weights and is
flattened in NumPy C order, matching `x.reshape(design.shape)`. For a 2D
`(nx, ny)` design, y is the fastest-varying index; for a 3D `(nx, ny, nz)`
design, z is fastest; for a cylindrical `(nr, nz)` design, z is fastest. If
`update_design` applies an external filter, projection, or other nonlinear
mapping, apply that mapping's transpose or vector-Jacobian product outside the
objective. MaterialGrid's internal projection and averaging options remain
unsupported by this native path. The [periodic fabrication filter](fabrication.md)
documents TAMA's external conic filter, tanh projection, and their VJPs for
full-cell periodic designs.

Forward design-field histories are stored as temporary, rank-local memory maps.
When Meep initializes real fields, TAMA automatically uses the real
counterpart of `history_dtype` (`complex128` to `float64`, for example) for
the raw design and target-field histories. Complex forward Meep fields require
a complex `history_dtype`; explicitly requesting a real dtype then raises an
error instead of discarding the imaginary field. A derived eigenmode
coefficient remains complex even when its constituent Meep field histories are
real. MPI history reductions retain `float32` or `complex64` storage when
requested and split buffers into bounded-count collectives, so neither
single-precision histories nor arrays exceeding one MPI call's count limit
require a full double-precision staging array. The per-rank design-history
storage is approximately
`resolved_dtype.itemsize * stored_rows *
sum(local_Yee_sites_per_component)`. A 3D gradient records `Ex`, `Ey`, and `Ez`
histories and can therefore use substantially more temporary disk than a 2D
TMz gradient. Set `TMPDIR` before launching Python or `mpirun` to select a
scratch location. Temporary history files are removed during normal cleanup;
an externally killed process can leave `tama_history_*.dat` files behind.

Minimal polarization-specific inputs are:

```python
# 2D TEz: the monitored/source component is Hz; the design gradient uses Ex/Ey.
tez_target = tm.PointTarget(
    position=monitor_position,
    component=mp.Hz,
)

# 3D: use a 3D MaterialGrid and one matching axis-aligned Block.
material_grid_3d = mp.MaterialGrid(
    mp.Vector3(nx, ny, nz),
    air,
    design_material,
)
design_size_3d = mp.Vector3(sx, sy, sz)
design_3d = tm.DesignGrid(
    material_grid=material_grid_3d,
    center=design_center_3d,
    size=design_size_3d,
    shape=(nx, ny, nz),
    background=air,
    design_material=design_material,
)
geometry_3d = [
    mp.Block(
        center=design_center_3d,
        size=design_size_3d,
        material=material_grid_3d,
    )
]
simulation_3d = tm.SimulationSpec(
    cell_size=cell_size_3d,
    boundary_layers=pml_layers,
    geometry=geometry_3d,
    sources=fwd_sources,
    resolution=resolution,
    dimensions=3,
    eps_averaging=False,
)
target_3d = tm.PointTarget(
    position=monitor_position_3d,
    component=mp.Ez,
)
```

The forward sources and geometry must produce the corresponding polarization;
changing only the target component does not convert an existing TMz simulation
to TEz.

These `DesignGrid`, `SimulationSpec`, `PointTarget`, `FieldRegionTarget`,
`FluxTarget`, and `EigenmodeCoefficientTarget` objects can be passed to either
`TDAObjective` or `MultiTDAObjective`.

For less repetitive objective setup, use the bundled input objects:

| Object | Bundles |
| --- | --- |
| `DesignGrid` | `MaterialGrid`, native design-region center/size/shape, `d epsilon / d rho`, and geometric coordinate/area metadata. |
| `SimulationSpec` | Common `mp.Simulation` constructor inputs such as cell size, PML layers, geometry, default sources, resolution, Courant factor, `geometry_center`, `chunk_layout`, dimensions, Gamma-point periodic boundaries, and cylindrical `m`. Leaving `chunk_layout` unset enables automatic mixed-topology balancing in the objective classes. |
| `PointTarget` | Point-monitor position, field component, and matching adjoint-source size/amplitude. |
| `FieldRegionTarget` | Ordered exact field-sample positions, their spatial output shape and quadrature weights, one field component, and a shared adjoint amplitude. |
| `FluxTarget` | Ordered surface-sample positions, a signed axis-aligned normal, and explicit physical surface-quadrature weights. |
| `EigenmodeCoefficientTarget` | A fixed reference mode and explicit physical surface quadrature for its complex time-domain overlap coefficient. Cartesian targets may solve the mode with MPB; cylindrical targets take fixed sampled mode fields. |

`DesignGrid.coords_x`, `coords_y`, `coords_z`, `coords_r`, `cell_area`, and
`cell_volume` describe its regular design-variable grid. For cylindrical
grids, `cell_area` is the coordinate-plane area `dr * dz`, not the physical
cylindrical volume measure. Native design gradients do not sample at these
nominal coordinates; they use the discovered Yee sites and Meep integration
weights instead.

`SimulationSpec.make()` behaves like the examples' usual `make_sim(sources=None)`
factory: no argument creates the forward simulation, and an explicit source list
creates the adjoint simulation. Its `courant` input defaults to Meep's default
`0.5`; set it explicitly only when the simulation requires a different Courant
factor. TAMA reads the resulting time step from the initialized Meep fields.
The default `k_point=False` preserves Meep's perfect-metal boundaries; set
`k_point=mp.Vector3()` explicitly for normal-incidence Gamma-point periodic
boundaries. For an x-y-periodic 3D unit cell, combine this with z-directed PML
layers such as `mp.PML(pml_thickness, direction=mp.Z)`. Forward and adjoint
simulations then use the same zero Bloch wavevector, including with
`reuse_simulation=True`. Nonzero Bloch wavevectors and oblique-incidence
gradients are not yet supported by `SimulationSpec`.
