from typing import Any, Optional, Sequence, overload

import numpy as np
from numpy.typing import NDArray

API_VERSION: int

def create_native_design_plan(
    fields_addr: int,
    geps_addr: int,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    nx: int,
    ny: int,
    nz: int,
    component: int,
    material_jacobian: bool = False,
    eps_averaging: bool = False,
    subpixel_tol: float = 1e-4,
    subpixel_maxeval: int = 100000,
    material_derivative_step: float = 1e-5,
) -> Any:
    """Create a rank-local Yee/material-Jacobian plan.

    Cylindrical plans use `center_x/size_x` for r, `center_z/size_z`
    for z, and logical counts `(nx, ny) == (nr, nz)` with `nz == 1`.
    """
    ...

def configure_native_material_operator(
    structure_addr: int,
    geps_addr: int,
    eps_averaging: bool,
    subpixel_tol: float,
    subpixel_maxeval: int,
) -> None:
    """Initialize anisotropic or 3D scalar MaterialGrid averaging before field creation."""
    ...

def native_design_plan_local_size(plan: Any) -> int:
    """Return the rank-local history entry count, including tensor stencils."""
    ...

def native_design_plan_signature(plan: Any) -> NDArray[np.int64]:
    """Return Yee coordinates; tensor plans append component and stencil node."""
    ...

def sample_native_design_plan_into(
    plan: Any,
    destination: NDArray[np.complex128],
) -> None:
    """Sample rank-local native Yee values into `destination`."""
    ...

def sample_native_design_plan_real_into(
    plan: Any,
    destination: NDArray[np.float64],
) -> None:
    """Sample real rank-local native Yee values into a writable C array.

    The plan's Meep fields must use real storage. `destination` must be a
    native-endian C-contiguous writable 1D `float64` array.
    """
    ...

def accumulate_native_design_product_local_inplace(
    plan: Any,
    forward_values: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Apply the native MaterialGrid transpose to one field-product row."""
    ...

def accumulate_native_design_real_product_local_inplace(
    plan: Any,
    forward_values: NDArray[np.float64],
    accumulator: NDArray[np.float64],
) -> None:
    """Apply the native MaterialGrid transpose to one real field-product row."""
    ...

def accumulate_native_design_midpoint_product_local_inplace(
    plan: Any,
    forward_values: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> bool:
    """Accumulate using the midpoint of consecutive adjoint samples.

    The first call caches the current adjoint field and returns `False`.
    Later calls update `accumulator` and return `True`.
    """
    ...

def accumulate_native_design_real_midpoint_product_local_inplace(
    plan: Any,
    forward_values: NDArray[np.float64],
    accumulator: NDArray[np.float64],
) -> bool:
    """Accumulate real fields using consecutive-adjoint-sample midpoints."""
    ...

def run_native_design_adjoint_segment(
    plans: Sequence[Any],
    histories: Sequence[NDArray],
    reconstruction_weights: NDArray[np.float64],
    accumulator: NDArray,
    sampling_interval: int,
    reconstruction_first_offset: int,
    fine_step_count: int,
    start_fine_index: int,
    sample_count: int,
    dt: float,
    midpoint: bool,
    gradient_offsets: Optional[Sequence[int]] = None,
) -> float:
    """Run one native adjoint segment and return non-FDTD work seconds.

    The segment accumulates decreasing fine-grid indices beginning at
    `start_fine_index` and performs one Meep step between consecutive rows.
    """
    ...

def native_forward_step_count(fields_addr: int, run_until: float) -> int:
    """Return the number of Meep steps for a rounded-time run duration."""
    ...

def run_native_forward_segment(
    monitor_plans: Sequence[Any],
    monitor_histories: Sequence[NDArray],
    design_plans: Sequence[Any],
    design_histories: Sequence[NDArray],
    sampling_interval: int,
    fine_step_count: int,
    start_fine_index: int,
    sample_count: int,
    monitor_times: NDArray[np.float64],
    fields_addr: int = 0,
) -> float:
    """Sample rank-local histories and advance one native forward segment.

    Monitor plans are point-monitor or eigenmode-overlap plans bound to the
    same Meep fields. Point histories have one column per configured history
    index, or one per point when no subset is configured. Overlap histories
    have two columns. Design histories contain
    `fine_step_count // sampling_interval + 1` rows. The half-open sample
    range begins at `start_fine_index`, and the call advances Meep by one
    fewer step than `sample_count`. The return value is sampling time only;
    it excludes FDTD stepping.
    """
    ...

def create_component_grid_plan(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int = ...,
) -> Any:
    """Precompute native sampling metadata for one field component grid.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer, e.g. `int(mp.Ez)`.
            Defaults to Ez in the native module.

    Returns:
        Opaque native plan reused by plan-based sampling functions.
    """
    ...

def sample_component_grid_plan_allreduced(
    plan: Any,
) -> NDArray[np.complex128]:
    """Sample a component grid using a precomputed native plan.

    Args:
        plan: Opaque object returned by `create_component_grid_plan`.

    Returns:
        Complex sampled field grid after MPI all-reduction.
    """
    ...

@overload
def create_component_point_plan(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
) -> Any:
    """Precompute metadata for paired 2D `(x, y)` or cylindrical `(r, z)` points."""
    ...

@overload
def create_component_point_plan(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    coords_z: Sequence[float],
    component: int,
) -> Any:
    """Precompute metadata for paired Cartesian 3D `(x, y, z)` points."""
    ...

def sample_component_point_plan_allreduced(
    plan: Any,
) -> NDArray[np.complex128]:
    """Sample all paired monitor coordinates with one packed MPI reduction."""
    ...

def component_point_plan_indexed_stencil(
    plan: Any,
) -> tuple[
    NDArray[np.intp],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.intp],
    NDArray[np.complex128],
]:
    """Return local Cartesian/cylindrical transpose stencil metadata.

    Returns:
        `(offsets, components, chunk_indices, local_indices, amplitudes)`.
        `offsets` has one CSR-style interval per monitored point. The four
        remaining one-dimensional arrays describe each local Yee support
        entry and have length `offsets[-1]`. `amplitudes` contains the
        exact transpose interpolation weights used by `mp.IndexedSource`.
    """
    ...

def configure_component_point_plan_history(
    plan: Any,
    indices: NDArray[np.int64],
) -> None:
    """Restrict local/native-forward history columns to selected points."""
    ...

def populate_sourcedata(
    sourcedata_addr: int,
    component: int,
    chunk_index: int,
    local_index: int,
) -> None:
    """Populate one Meep sourcedata object with an exact rank-local field index."""
    ...

def merge_sourcedata(
    destination_addr: int,
    source_addresses: Sequence[int],
) -> None:
    """Merge singleton sourcedata indices sharing one component and chunk."""
    ...

def sample_component_point_plan_local_into(
    plan: Any,
    destination: NDArray[np.complex128],
) -> None:
    """Sample rank-local paired monitor contributions into `destination`."""
    ...

def sample_component_point_plan_local_real_into(
    plan: Any,
    destination: NDArray[np.float64],
) -> None:
    """Sample real local monitor contributions into a writable C array.

    The plan's Meep fields must use real storage. `destination` must be a
    native-endian C-contiguous writable 1D `float64` array.
    """
    ...

def create_eigenmode_overlap_plan(
    component_plans: Sequence[Any],
    weights: Sequence[NDArray[np.complex128]],
    output_channels: NDArray[np.intp],
) -> Any:
    """Bind component point plans to fixed electric/magnetic weights."""
    ...

def sample_eigenmode_overlap_plan_local_into(
    plan: Any,
    destination: NDArray[np.complex128],
) -> None:
    """Write rank-local electric and magnetic overlap contributions."""
    ...

def component_grid_plan_local_complete_mask(
    plan: Any,
) -> NDArray[np.bool_]:
    """Return points whose interpolation support is fully local to this rank.

    Args:
        plan: Opaque object returned by `create_component_grid_plan`.

    Returns:
        Boolean grid with shape `(len(coords_x), len(coords_y))`.
    """
    ...

def component_grid_plan_local_boundary_mask(
    plan: Any,
) -> NDArray[np.bool_]:
    """Return boundary points whose interpolation support includes this rank.

    Args:
        plan: Opaque object returned by `create_component_grid_plan`.

    Returns:
        Boolean grid with shape `(len(coords_x), len(coords_y))`.
    """
    ...

def configure_component_grid_plan_history(
    plan: Any,
    local_indices: NDArray[np.int64],
    boundary_indices: NDArray[np.int64],
) -> None:
    """Cache fixed history indices and boundary communication groups.

    In MPI, every active rank must call this function in the same order.
    Within each boundary support group, ranks must configure the same point
    sequence; use `FastFieldGrid.history_sampling_indices()` rather than
    caller-selected subsets.
    """
    ...

def sample_component_grid_plan_history_into(
    plan: Any,
    destination: NDArray[np.complex128],
) -> None:
    """Sample a configured history row directly into `destination`.

    This is collective over the configured boundary support communicators;
    every active rank must call it in the same order.
    """
    ...

def configure_component_grid_plan_accumulation(
    plan: Any,
    indices: NDArray[np.int64],
) -> None:
    """Cache fixed indices used by repeated gradient accumulation."""
    ...

def sample_component_grid_plan_points_local(
    plan: Any,
    indices: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Sample selected flat point indices from rank-local plan entries."""
    ...

def sample_component_grid_plan_points_allreduced(
    plan: Any,
    indices: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Sample selected flat point indices and MPI-sum rank contributions."""
    ...

def sample_component_grid_plan_points_support_reduced(
    plan: Any,
    indices: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Sample selected boundary indices and reduce only support ranks.

    Supporting ranks must call with identical index order.
    """
    ...

def sample_component_grid(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int = ...,
) -> NDArray[np.complex128]:
    """Sample a Meep field component on a 2D coordinate grid.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer. Defaults to Ez in the native
            module.

    Returns:
        Complex sampled field grid.
    """
    ...

def sample_component_grid_allreduced(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int = ...,
) -> NDArray[np.complex128]:
    """Sample a component grid and MPI-all-reduce rank-local contributions.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer. Defaults to Ez in the native
            module.

    Returns:
        Complex sampled field grid with MPI rank contributions combined.
    """
    ...

def accumulate_component_product_allreduced(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
    multiplier: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return MPI-all-reduced `sampled_field * multiplier`.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.
        multiplier: Complex grid multiplied pointwise with the sampled field.

    Returns:
        Complex product grid after MPI all-reduction.
    """
    ...

def accumulate_component_product_local_inplace(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
    multiplier: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Accumulate `sampled_field * multiplier` into `accumulator` in place.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.
        multiplier: Complex grid multiplied pointwise with the sampled field.
        accumulator: Complex grid updated in place on the local rank.
    """
    ...

def accumulate_component_product_plan_local_inplace(
    plan: Any,
    multiplier: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Plan-based in-place accumulation of `sampled_field * multiplier`.

    Args:
        plan: Opaque object returned by `create_component_grid_plan`.
        multiplier: Complex grid multiplied pointwise with the sampled field.
        accumulator: Complex grid updated in place on the local rank.
    """
    ...

def accumulate_component_product_plan_points_local_inplace(
    plan: Any,
    indices: NDArray[np.int64],
    values: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Plan-based in-place accumulation for selected flat point indices.

    Args:
        plan: Opaque object returned by `create_component_grid_plan`.
        indices: Flat point indices with nonzero multiplier values.
        values: Complex multiplier values matching `indices`.
        accumulator: Complex grid updated in place on the local rank.
    """
    ...

def accumulate_component_product_plan_configured_local_inplace(
    plan: Any,
    values: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Accumulate products using indices cached in `plan`."""
    ...

def accumulate_component_difference_product_plan_configured_local_inplace(
    plan: Any,
    values: NDArray[np.complex128],
    dt: float,
    accumulator: NDArray[np.complex128],
) -> bool:
    """Accumulate values times the current-minus-previous field derivative.

    The first call caches the current field and returns `False`. Later calls
    update `accumulator` and return `True`.
    """
    ...

def reduce_complex_grid_sum(
    local_grid: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """MPI-sum a complex grid over the active Meep process group.

    Args:
        local_grid: Rank-local complex grid.

    Returns:
        Complex grid after MPI summation.
    """
    ...

def reduce_complex_grid_sum_inplace(
    local_grid: NDArray,
) -> NDArray:
    """MPI-sum a complex grid in place over the active Meep process group.

    `local_grid` must be a writable C-contiguous native-endian 2D
    `complex64` or `complex128` array. The same array object and dtype are
    returned after summation.
    """
    ...

def reduce_real_grid_sum_inplace(
    local_grid: NDArray,
) -> NDArray:
    """MPI-sum a real grid in place over the active Meep process group.

    `local_grid` must be a writable C-contiguous native-endian 2D
    `float32` or `float64` array. The same array object and dtype are returned
    after summation.
    """
    ...

class TabulatedCubic:
    """Native complex piecewise-cubic callable created by its factory."""

    def __call__(self, time: float) -> complex: ...

def create_tabulated_cubic(
    breaks: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
) -> TabulatedCubic:
    """Create a native callable from cubic polynomial coefficients.

    `breaks` has shape `(n + 1,)` and `coefficients` has shape
    `(4, n)`.
    """
    ...

class TabulatedRealCubic:
    """Native real piecewise-cubic callable created by its bank factory."""

    def __call__(self, time: float) -> float: ...

def create_tabulated_real_cubic_bank(
    breaks: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> tuple[TabulatedRealCubic, ...]:
    """Create one real cubic callable per shared coefficient-bank channel.

    `breaks` has shape `(n + 1,)` and `coefficients` has shape
    `(4, n, channels)`. All returned callables share both arrays.
    """
    ...

class TabulatedBSpline:
    """Native complex cubic B-spline callable created by its bank factory."""

    def __call__(self, time: float) -> complex: ...

def create_tabulated_bspline_bank(
    knots: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
) -> tuple[TabulatedBSpline, ...]:
    """Create complex cubic B-splines sharing knots and coefficients.

    `knots` has shape `(n + 4,)` and `coefficients` has shape
    `(n, channels)` for cubic not-a-knot B-splines.
    """
    ...

class TabulatedRealBSpline:
    """Native real cubic B-spline callable created by its bank factory."""

    def __call__(self, time: float) -> float: ...

def create_tabulated_real_bspline_bank(
    knots: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> tuple[TabulatedRealBSpline, ...]:
    """Create real cubic B-splines sharing knots and coefficients.

    `knots` has shape `(n + 4,)` and `coefficients` has shape
    `(n, channels)` for cubic not-a-knot B-splines.
    """
    ...

@overload
def shift_tabulated_bspline(
    source: TabulatedBSpline,
    time_shift: float,
) -> TabulatedBSpline: ...
@overload
def shift_tabulated_bspline(
    source: TabulatedRealBSpline,
    time_shift: float,
) -> TabulatedRealBSpline:
    """Create a shifted view that evaluates `source(time - time_shift)`."""
    ...

def fold_near2far_sources(
    fields_addr: int,
    near2far_addr: int,
    source_addresses: Sequence[int],
) -> list[tuple[int, int, NDArray[np.intp], NDArray[np.complex128]]]:
    """Return chunk-grouped sources for the exact near-to-far transpose."""
    ...

def sample_ez_grid(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int = ...,
) -> NDArray[np.complex128]:
    """Compatibility alias for `sample_component_grid`.

    Args:
        fields_addr: Integer address of Meep's low-level `fields` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer. Defaults to Ez in the native
            module.

    Returns:
        Complex sampled component grid.
    """
    ...
