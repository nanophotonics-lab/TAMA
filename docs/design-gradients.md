# Design Gradients and Grid Coordinates

This guide describes the Meep backend. See [backend selection](backends.md) and [FDTDX](fdtdx.md) for the other engine.

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
Cartesian simulation and sample a regular point grid.

When `DesignGrid` is supplied, TAMA instead discovers the native Yee sites
for every electric component required by the design gradient. The user supplies
the physical center and size of the design region without component-specific
shifts or pixel grouping. During the adjoint run, each Yee-site contribution is
scattered into the raw MaterialGrid weights through the discrete transpose of
Meep's MaterialGrid coordinate map or constitutive-operator Jacobian.

The native path requires:

- a 2D or 3D Cartesian simulation, or a 2D Meep cylindrical simulation;
- one distinct MaterialGrid per independent design region, each used by one
  matching, axis-aligned `mp.Block`;
- matching `DesignGrid` and Block center, size, shape, and dimensionality;
- Cartesian mirror symmetry as described below, or no symmetry; either Meep's default
  `k_point=False` metallic boundaries or normal-incidence Gamma-point periodic
  boundaries selected explicitly with `k_point=mp.Vector3()`;
- a design Block that does not overlap PML or absorber layers, with an
  additional half-grid-cell clearance for tensor/averaging stencils;
- MaterialGrid `damping=0` and no overlapping
  grid combination;
- two linear, real symmetric positive-definite, nondispersive dielectric
  endpoint tensors without electric
  or magnetic susceptibilities, nonlinearities, or conductivity;
- equal, fixed isotropic permeability in the two endpoint media; and
- identical rank-local Yee layouts in the forward and adjoint simulations.

Cartesian 2D/3D supports `Simulation.eps_averaging`,
`MaterialGrid.do_averaging`, and internal `beta`/`eta` projection. The scalar
path with `eps_averaging=False`, `do_averaging=False`, and `beta=0` retains its
direct interpolation transpose and requires `material_factor` to equal the
endpoint permittivity contrast. Other Cartesian cases derive the material
Jacobian from the MaterialGrid endpoints and do not multiply by
`material_factor`. Cylindrical gradients retain the scalar restrictions below.

## Independent Design Regions

Both objective classes accept `designs=[design_a, design_b, ...]` instead of
`design=design_a`. Supply only one of these arguments. Each region
has its own MaterialGrid, shape, endpoints, and projection/averaging settings.
Regions must not overlap and must use the same dimensionality and coordinate
system.

The optimizer vector and returned gradient concatenate the flattened region
arrays in list order:

```python
x = np.concatenate([rho_a.ravel(), rho_b.ravel()])
problem = tm.TDAObjective(
    designs=[design_a, design_b], simulation=simulation,
    target=target, t_final=t_final,
)
value, gradient = problem.fom_and_grad(x)
gradient_a = gradient[:rho_a.size].reshape(rho_a.shape)
gradient_b = gradient[rho_a.size:].reshape(rho_b.shape)
```

All regions share one forward simulation and one combined adjoint simulation
for a scalar objective. Histories remain rank-local and separate for each
region. Overlapping MaterialGrid combinations and tied variables across
different regions are not supported.

## Cartesian Mirror Symmetry

Pass ordinary Meep objects through
`SimulationSpec(symmetries=[mp.Mirror(mp.Y, phase=1)])`. Cartesian gradients
support distinct mirror axes and phases `+1` or `-1`; rotations and cylindrical
symmetries are not supported. Meep reduces the simulation domain.
In 2D, the mirror axis must be `mp.X` or `mp.Y`; 3D also permits `mp.Z`.
Mirror gradients require a direct `SimulationSpec` and exact indexed adjoint
sources. Custom simulation factories and finite-size `PointTarget` sources
remain available for forward-only evaluation, but gradients reject them.
`MultiTDAObjective` uses its regular target-history path with mirrors, even
when `target_history_block_size` requests distributed target histories.
Automatic adaptive chunk balancing is disabled for mirrors, and explicit
adaptive balancers are rejected because their moving-cut bounds assume the
full cell. Fixed layouts remain supported for temporal targets; mirrored
near-to-far targets require Meep's default partitioning.

Each independent design Block must map onto itself under every mirror. Its
density array must be reflection symmetric, and its fixed endpoint tensors
must be invariant under the same reflection. TAMA checks these design
conditions at evaluation. Mirrors that exchange two independent regions are
rejected. Fixed geometry and forward sources must also obey the requested
symmetry and phase.

When both `eps_averaging` and `MaterialGrid.do_averaging` are enabled, the
design grid must have an even number of samples along each mirror axis
(a singleton axis is also allowed). Odd sample counts greater than one place
an interpolation knot on the mirror plane. Meep's one-sided interface normal
there can produce an averaged tensor that breaks reflection symmetry, so
TAMA rejects this combination. Without MaterialGrid averaging, odd counts
remain supported.

The returned full-shaped density gradient is distributed over reflection
orbits. Its dot product gives the directional derivative only for
symmetry-preserving perturbations. Filters, initialization, and optimizer
updates must preserve that symmetry.

## Supported Components

Supplying `DesignGrid` selects the following electric components from the
simulation dimensionality and monitored polarization:

| Simulation / monitor polarization | Electric components used for the design gradient |
| --- | --- |
| Cartesian tensor, projection, or averaging path | All three constitutive rows, using `Dx`, `Dy`, and `Dz` histories |
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
Cylindrical gradients support `m=-1`, `0`, and `+1` and require a direct
`simulation=SimulationSpec(...)` to construct the adjoint simulation with mode
`-m`.

Both objective classes support `PointTarget`, `FieldRegionTarget`,
`FluxTarget`, `EigenmodeCoefficientTarget`, and `Near2FarTarget` in this
coordinate system.
Flux and eigenmode targets are restricted to radial or axial surfaces and use
explicit physical quadrature, as detailed in the [target guide](targets.md).
Near-to-far targets use fixed chunks and always restart the forward simulation,
as described in the [near-to-far target guide](targets.md#near-to-far-targets).

Each `TDAObjective` or `MultiTDAObjective` evaluates one angular mode. The
`Multi` in `MultiTDAObjective` refers to wavelength bands, not angular modes.
Meep represents that mode by complex amplitudes multiplying `exp(i*m*phi)`.
The adjoint simulation uses `-m` to cancel the angular dependence in the
bilinear forward/adjoint pairing. Returned field, flux, and mode-overlap values are
single-`m` modal quantities under the supplied source and quadrature
normalization. Run separate forward objectives and apply the required physical
normalization and combination externally when a real three-dimensional result
needs contributions from multiple values of `m`.

Cylindrical gradients require linear, scalar-isotropic, nondispersive
permittivity, fixed and equal endpoint permeability, `eps_averaging=False`,
`MaterialGrid(do_averaging=False, beta=0)`, and `material_factor` equal to the
endpoint permittivity contrast. The design region may touch `r=0` but must not extend below it.
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
Meep symmetries are not supported for cylindrical gradients. Cylindrical
custom `sim_factory` results must therefore use `symmetries=[]`.
For temporal targets, the default `AdaptiveAdjointChunkBalancer` supports
Meep's native cylindrical R-Z topology with zero-size indexed adjoint sources.

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

## Cartesian Tensor Gradients and Averaging

Each design variable remains a scalar density mixing two fixed dielectric
tensors. The endpoint tensors may have unequal diagonal and nonzero real
off-diagonal entries.

The general path records displacement fields and contracts
`-D_adjoint * (d M / d rho) * d D_forward / dt`, where `M` is Meep's discrete
inverse-permittivity operator. It uses the staggered mixed-component stencil,
including neighboring displacement samples. The material Jacobian uses local
finite differences of the same material evaluator, including interpolation,
projection, and the density-dependent interface normal. This adds material
setup cost, but does not run additional FDTD solves for each design variable.
The complete gradient still includes the existing time-discretization and
adjoint-source approximations of the time-domain method.

`eps_averaging` controls Meep's global subpixel averaging; `do_averaging`
requests MaterialGrid interface averaging when global averaging is enabled.
Internal `beta` projection also applies when averaging is disabled. For
anisotropic MaterialGrid endpoints, TAMA supplies interface-normal tensor
averaging. It reduces to harmonic normal and arithmetic tangential averaging
for isotropic endpoints. Forward,
adjoint, and FoM-only evaluations use the same material operator.
The tensor mixing rule follows
[Kottke, Farjadpour, and Johnson](https://arxiv.org/abs/0708.1031);
[Meep's subpixel smoothing guide](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/)
describes the isotropic reduction and averaging controls.

TAMA initializes this tensor operator before creating Meep fields. A custom
`sim_factory` must return an uninitialized simulation when anisotropic
MaterialGrid averaging is enabled. Rebuild the Meep native extension together
with the Python package; this path requires native API 16.

In 3D, scalar-isotropic MaterialGrid averaging also uses TAMA's normalized
volume-averaging kernel and requires an uninitialized simulation from a custom
`sim_factory`. This corrects the averaging-kernel normalization in stock Meep
1.34.0, so existing 3D averaged MaterialGrid results can change. Geometric
objects continue to use Meep's own averaging, and zero-normal MaterialGrid
points retain their pointwise material evaluation.

For active MaterialGrid averaging, TAMA limits `subpixel_tol` to `1e-8`,
preserving stricter factory settings. Forward initialization and the local
material Jacobian use that same tolerance. This tighter tolerance increases
material setup cost. A factory must return an uninitialized simulation if this tolerance
change is needed.

Gradient evaluation rejects material-derivative stencils that cross Meep's
zero-normal averaging branch at mixed densities. Forward evaluation remains
available, as do clamped points whose normal remains identically zero under
the perturbation. Unsmooth infinite-beta projection at its threshold is also rejected
for gradients. Use finite beta and a spatially varying design for averaged
optimization. Supported media are listed in the native-path requirements above.

## Gradient Output, Storage, and Inputs

The returned native gradient is with respect to raw MaterialGrid weights and is
flattened in NumPy C order, matching `x.reshape(design.shape)`. For a 2D
`(nx, ny)` design, y is the fastest-varying index; for a 3D `(nx, ny, nz)`
design, z is fastest; for a cylindrical `(nr, nz)` design, z is fastest. If
`update_design` applies an external filter, projection, or other nonlinear
mapping, apply that mapping's transpose or vector-Jacobian product outside the
objective. MaterialGrid's internal projection and averaging derivatives are
already included, so do not apply their VJP a second time. The [periodic fabrication filter](fabrication.md)
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
sum(local_history_entries_per_component)`. Tensor plans have additional
entries for the mixed-component displacement stencils; their storage can
exceed three scalar component histories. A scalar 3D gradient records `Ex`, `Ey`, and `Ez`
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
| `SimulationSpec` | Common `mp.Simulation` constructor inputs such as cell size, PML layers, geometry, default sources, resolution, Courant factor, `geometry_center`, `chunk_layout`, dimensions, Gamma-point periodic boundaries, and cylindrical `m`. See [MPI execution](mpi.md) for chunk-balancing conditions. |
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
