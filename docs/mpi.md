# MPI Chunk Topology and Source-Boundary Safety

This guide describes the Meep backend. See [backend selection](backends.md) and [FDTDX](fdtdx.md) for the other engine.

For temporal targets without mirror symmetry, leave
`SimulationSpec.chunk_layout` unset to use the default automatic chunk
balancer. In MPI, before the first value or gradient evaluation, this path performs
one source-free Meep initialization, captures Meep's native
`BinaryPartition`, and projects only its split positions onto safe Meep grid
planes. Serial execution skips this probe. TAMA preserves Cartesian mixed-axis
trees and cylindrical R-Z trees.

Mirror symmetry and near-to-far targets disable `chunk_balancer="auto"` and
reject explicit adaptive balancers. They keep fixed forward/adjoint ownership.
Fixed layouts are supported for temporal targets and non-mirrored near-to-far
targets. Near-to-far targets combined with mirrors require Meep's default
partitioning; an explicit `chunk_layout` is rejected. See the
[target guide](targets.md#near-to-far-targets).

```python
simulation = tm.SimulationSpec(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    geometry_center=geometry_center,
    resolution=resolution,
)

# Temporal targets without mirrors use chunk_balancer="auto".
# Pass chunk_balancer=None only to opt out explicitly.
```

The automatic path resolves `SimulationSpec.sources` once per objective
evaluation and treats every ordinary `mp.Source`, including subclasses such as
`mp.EigenModeSource`, as a permanent source-boundary constraint.
`mp.IndexedSource` is excluded from geometric constraints. Use `source_volumes`
on a custom `AdaptiveAdjointChunkBalancer` only for additional source supports
that are not present in `SimulationSpec.sources`. A callable may return fresh source objects
on each evaluation, but their centers and sizes must remain unchanged after the
layout is prepared.

With the direct `SimulationSpec.make` path, default
zero-size Cartesian and cylindrical `PointTarget` entries use the exact
rank-local transpose of their native point-sampling stencil through
`mp.IndexedSource`. Every `FieldRegionTarget` sample uses the same exact
indexed path, as does every constituent E/H sample of a `FluxTarget` or
`EigenmodeCoefficientTarget`. Indexed targets remain zero size. Explicit nonzero
Cartesian point-target sizes,
`source_boundary_mode="finite"`, and arbitrary `sim_factory` inputs use
ordinary `mp.Source` objects instead. Do not also list indexed target positions
in `protected_points` or `source_volumes`; TAMA retains these as permanent
constraints.

During three to eight successful gradient evaluations, TAMA combines Meep's
per-rank timing with TAMA's rank-local sampling, reconstruction, and
gradient-accumulation time. A split-position candidate selected after one
evaluation is measured by the next evaluation within the fixed Meep topology.
TAMA freezes the best measured layout after two consecutive observations are
within the imbalance threshold or fail to improve the best critical-rank work
by at least 1%. A trial that is 2% or more slower than the best is rolled back without
ending calibration; the best measured layout is frozen after eight
observations at the latest. Calibration uses ordinary optimizer evaluations.
Value-only evaluations prepare source-safe boundaries
but do not advance the workload calibration. Once frozen, the objective stops
collecting calibration timings and retains that layout for its lifetime.

One `AdaptiveAdjointChunkBalancer` instance is bound on first use to one
`SimulationSpec` object and one effective adjoint-source configuration. Create
a separate balancer for another simulation or target configuration; reusing the
instance across them raises an error.

For non-indexed Cartesian target sources, the default source-boundary mode is
automatic and layout-first:

1. Move the mixed topology's split positions away from the source support.
2. If source support is the only reason no safe split remains, expand only the
   conflicting zero-size source axes to one Meep grid cell and renormalize the
   source amplitude to preserve its integrated weight.
3. Retry the same partition tree with the expanded source.

For grid spacing `Delta = 1 / resolution`, each changed source axis gains width
`Delta` and contributes a factor `1 / Delta` to the source amplitude. This
policy applies at every resolution in Cartesian 2D and 3D.

For indexed Cartesian `PointTarget` entries,
the `auto` and `layout` modes leave the source and native topology
unconstrained at the target; `finite` deliberately selects the
ordinary finite-source path. `FieldRegionTarget`, `FluxTarget`, and
`EigenmodeCoefficientTarget` always use exact indexed sources, so an adaptive
balancer configured with `finite` is rejected when any is present. For
cylindrical simulations, `auto` and `layout` keep exact indexed adjoint
sources unchanged, while `finite` is rejected. Adaptive updates use the
measured per-rank work, preserve Meep's R-Z partition tree, and count
computational cells as `nr * nz`. No physical `2*pi*r` volume factor is used
for workload balancing.

Custom constraints or thresholds may be supplied explicitly:

```python
chunk_balancer = tm.AdaptiveAdjointChunkBalancer(
    protected_points=permanent_protected_points,
    source_volumes=additional_source_volumes,
    source_boundary_mode="auto",
    finite_source_width_cells=1.0,
)
```

Finite-source regularization cannot fix a structurally infeasible partition,
such as a tree that leaves too few grid cells for `min_chunk_cells`. TAMA
raises an error in that case. During later workload rebalancing, an unsafe
proposal is rejected and the last safe mixed layout is retained. After a
gradient evaluation, `objective.last_chunk_balance` contains the latest
`AdaptiveChunkBalanceReport`, while
`objective.last_source_boundary_decision` contains the initialization
decision. Its `method`, `reason`, `chunk_layout`, `source_sizes`,
`source_amplitudes`, and `changed_axes` fields report whether the layout or
finite-source path was selected and the settings actually used by the adjoint
sources. A value-only evaluation prepares this decision but does not rebalance
or replace `last_chunk_balance`.

For callers constructing low-level Meep sources directly,
`regularize_source_size_and_amplitude` remains available for explicit finite
source normalization. Objective users should instead select
`AdaptiveAdjointChunkBalancer(source_boundary_mode="finite")` so TAMA applies
the normalization internally.
