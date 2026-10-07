import gc
import sys
import weakref
from types import SimpleNamespace

import meep as mp
import numpy as np
import pytest
import scipy.interpolate as spi

from tama import native_sampler
from tama.backends.meep.sampling_grid import (
    _prepare_native_material,
    _validate_native_material_grid,
)


@pytest.mark.parametrize("endpoint", ["medium1", "medium2"])
@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("epsilon_diag", mp.Vector3(2, np.nan, 4), "finite real permittivity"),
        ("epsilon_diag", mp.Vector3(2, 3 + 1j, 4), "finite real permittivity"),
        ("epsilon_offdiag", mp.Vector3(0.1j, 0, 0), "finite real permittivity"),
        ("epsilon_offdiag", mp.Vector3(10, 0, 0), "positive-definite"),
        ("E_susceptibilities", [mp.LorentzianSusceptibility()], "nondispersive"),
        ("H_susceptibilities", [mp.LorentzianSusceptibility()], "nondispersive"),
        ("E_chi2_diag", mp.Vector3(1, 0, 0), "linear lossless"),
        ("H_chi3_diag", mp.Vector3(0, 0, 1), "linear lossless"),
        ("D_conductivity_diag", mp.Vector3(0, 1, 0), "linear lossless"),
        ("B_conductivity_diag", mp.Vector3(0, 0, 1), "linear lossless"),
        ("mu_diag", mp.Vector3(1, 2, 1), "isotropic permeability"),
        ("mu_diag", mp.Vector3(-1, -1, -1), "positive.*permeability"),
        ("mu_diag", mp.Vector3(0, 0, 0), "positive.*permeability"),
        ("mu_diag", mp.Vector3(np.nan, np.nan, np.nan), "permeability"),
        ("mu_diag", mp.Vector3(1j, 1j, 1j), "permeability"),
        ("mu_offdiag", mp.Vector3(0.1, 0, 0), "isotropic permeability"),
        ("mu_diag", mp.Vector3(2, 2, 2), "fixed permeability"),
    ],
)
def test_material_endpoints_rejected_before_meep_initialization(
    monkeypatch, endpoint, attribute, value, message,
):
    grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        mp.Medium(epsilon_diag=mp.Vector3(2, 3, 4)),
        mp.Medium(epsilon_diag=mp.Vector3(5, 6, 7)),
        weights=np.array([0.2, 0.4, 0.6, 0.8]),
        do_averaging=True,
    )
    setattr(getattr(grid, endpoint), attribute, value)
    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2),
        resolution=4,
        geometry=[mp.Block(size=mp.Vector3(1, 1, mp.inf), material=grid)],
        eps_averaging=True,
    )

    def unexpected_initialization(*args):
        pytest.fail("unsupported material reached Meep initialization")

    monkeypatch.setattr(simulation, "_init_structure", unexpected_initialization)
    with pytest.raises(ValueError, match=message):
        _prepare_native_material(simulation, SimpleNamespace(material_grid=grid))
    assert simulation.structure is None
    assert simulation.fields is None


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"grid_type": 1}, "U_DEFAULT"),
        ({"damping": 0.1}, "damping=0"),
        ({"beta": -1}, "beta>=0"),
        ({"beta": np.nan}, "beta>=0"),
        ({"eta": -0.1}, "eta in"),
        ({"eta": 1.1}, "eta in"),
        ({"eta": np.nan}, "eta in"),
        ({"beta": np.inf, "eta": 0}, "strictly interior"),
        ({"beta": np.inf, "eta": 1}, "strictly interior"),
    ],
)
def test_native_material_controls_are_validated(changes, message):
    grid = mp.MaterialGrid(mp.Vector3(2, 2), mp.Medium(), mp.Medium(epsilon=4))
    for attribute, value in changes.items():
        setattr(grid, attribute, value)
    with pytest.raises(ValueError, match=message):
        _validate_native_material_grid(grid)


def test_native_material_validation_accepts_spd_and_fixed_positive_mu():
    grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        mp.Medium(epsilon_diag=mp.Vector3(2, 3, 4), mu=2),
        mp.Medium(
            epsilon_diag=mp.Vector3(5, 6, 7),
            epsilon_offdiag=mp.Vector3(0.2, 0.1, 0.3),
            mu=2,
        ),
        do_averaging=True,
        beta=np.inf,
        eta=0.3,
    )
    _validate_native_material_grid(grid)


def test_native_method_boundary_translates_cpp_allocation_failure():
    with pytest.raises(MemoryError):
        native_sampler._raise_bad_alloc_for_testing()


def test_native_design_flat_index_uses_checked_size_t_arithmetic():
    assert native_sampler._native_design_flat_index_for_testing(
        50_000,
        50_000,
        2,
        49_999,
        49_999,
        1,
    ) == 4_999_999_999

    with pytest.raises(OverflowError, match="out of range or overflows"):
        native_sampler._native_design_flat_index_for_testing(
            2**63,
            3,
            1,
            2**63 - 1,
            2,
            0,
        )


def test_complex_mpi_double_count_checks_the_expanded_count():
    int_max = np.iinfo(np.int32).max
    assert native_sampler._complex_mpi_double_count_for_testing(
        int_max // 2
    ) == 2 * (int_max // 2)
    with pytest.raises(OverflowError, match="MPI_DOUBLE"):
        native_sampler._complex_mpi_double_count_for_testing(
            int_max // 2 + 1
        )


def test_native_plan_capsules_use_tama_qualified_names():
    simulation = mp.Simulation(
        cell_size=mp.Vector3(1.0, 1.0),
        resolution=4,
    )
    simulation.init_sim()
    try:
        fields_addr = int(simulation.fields.this)
        grid_plan = native_sampler.create_component_grid_plan(
            fields_addr,
            [0.0],
            [0.0],
            int(mp.Ez),
        )
        point_plan = native_sampler.create_component_point_plan(
            fields_addr,
            [0.0],
            [0.0],
            int(mp.Ez),
        )
        assert "tama.native_sampler.ComponentGridPlan" in repr(grid_plan)
        assert "tama.native_sampler.ComponentPointPlan" in repr(point_plan)
    finally:
        simulation.reset_meep()


@pytest.mark.parametrize(
    "simulation_kwargs",
    (
        {"cell_size": mp.Vector3(1.0, 1.0, 1.0)},
        {
            "cell_size": mp.Vector3(1.0, 0.0, 1.0),
            "dimensions": mp.CYLINDRICAL,
        },
    ),
    ids=("cartesian-3d", "cylindrical"),
)
def test_raw_component_grid_operations_reject_non_2d_fields(
    simulation_kwargs,
):
    simulation = mp.Simulation(resolution=4, **simulation_kwargs)
    simulation.init_sim()
    try:
        fields_addr = int(simulation.fields.this)
        coords_x = [0.0]
        coords_y = [0.0]
        multiplier = np.ones((1, 1), dtype=np.complex128)
        accumulator = np.zeros((1, 1), dtype=np.complex128)
        operations = (
            lambda: native_sampler.sample_component_grid(
                fields_addr, coords_x, coords_y, int(mp.Ex)
            ),
            lambda: native_sampler.create_component_grid_plan(
                fields_addr, coords_x, coords_y, int(mp.Ex)
            ),
            lambda: native_sampler.sample_component_grid_allreduced(
                fields_addr, coords_x, coords_y, int(mp.Ex)
            ),
            lambda: native_sampler.accumulate_component_product_allreduced(
                fields_addr,
                coords_x,
                coords_y,
                int(mp.Ex),
                multiplier,
            ),
            lambda: native_sampler.accumulate_component_product_local_inplace(
                fields_addr,
                coords_x,
                coords_y,
                int(mp.Ex),
                multiplier,
                accumulator,
            ),
        )

        for operation in operations:
            with pytest.raises(
                ValueError,
                match="requires a Cartesian 2D Meep simulation",
            ):
                operation()
    finally:
        simulation.reset_meep()


def test_tabulated_cubic_types_require_factory_construction():
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        native_sampler.TabulatedCubic()
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        native_sampler.TabulatedRealCubic()
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        native_sampler.TabulatedBSpline()
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        native_sampler.TabulatedRealBSpline()

    breaks = np.array([0.0, 1.0], dtype=np.float64)
    complex_coefficients = np.zeros((4, 1), dtype=np.complex128)
    complex_coefficients[3, 0] = 2.0 + 3.0j
    complex_source = native_sampler.create_tabulated_cubic(
        breaks,
        complex_coefficients,
    )
    assert complex_source(0.5) == pytest.approx(2.0 + 3.0j)

    real_coefficients = np.zeros((4, 1, 2), dtype=np.float64)
    real_coefficients[3, 0] = (1.5, -2.0)
    real_sources = native_sampler.create_tabulated_real_cubic_bank(
        breaks,
        real_coefficients,
    )
    assert tuple(source(0.5) for source in real_sources) == pytest.approx(
        (1.5, -2.0)
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_tabulated_bspline_bank_matches_scipy_and_shift(complex_values):
    rng = np.random.default_rng(314159)
    times = np.cumsum(rng.uniform(0.01, 0.08, 17))
    values = rng.normal(size=(times.size, 5))
    if complex_values:
        values = values + 1j * rng.normal(size=values.shape)
    spline = spi.make_interp_spline(times, values, k=3, axis=0)
    factory = (
        native_sampler.create_tabulated_bspline_bank
        if complex_values
        else native_sampler.create_tabulated_real_bspline_bank
    )
    sources = factory(spline.t, spline.c)
    query = np.unique(np.concatenate((
        times,
        np.linspace(times[0], times[-1], 1001),
    )))
    actual = np.column_stack(
        [[source(time) for time in query] for source in sources]
    )
    reference = spi.CubicSpline(
        times,
        values,
        axis=0,
        extrapolate=False,
    )(query)

    assert np.allclose(actual, reference, rtol=2.0e-13, atol=2.0e-13)
    assert all(source(times[0] - 0.01) == 0.0 for source in sources)
    assert all(source(times[-1] + 0.01) == 0.0 for source in sources)
    assert all(
        source(np.nextafter(times[0], -np.inf)) == 0.0
        for source in sources
    )
    assert all(
        source(np.nextafter(times[-1], np.inf)) == 0.0
        for source in sources
    )

    time_shift = 0.037
    shifted = native_sampler.shift_tabulated_bspline(sources[2], time_shift)
    shifted_query = query + time_shift
    shifted_actual = np.asarray([shifted(time) for time in shifted_query])
    assert np.allclose(
        shifted_actual,
        reference[:, 2],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert shifted(times[0] + time_shift - 0.01) == 0.0
    assert shifted(times[-1] + time_shift + 0.01) == 0.0


def test_tabulated_bspline_shift_retains_shared_arrays():
    times = np.linspace(0.1, 0.9, 12)
    spline = spi.make_interp_spline(
        times,
        np.column_stack((np.sin(times), np.cos(times))),
        k=3,
        axis=0,
    )
    knot_ref = weakref.ref(spline.t)
    coefficient_ref = weakref.ref(spline.c)
    sources = native_sampler.create_tabulated_real_bspline_bank(
        spline.t,
        spline.c,
    )
    shifted = native_sampler.shift_tabulated_bspline(sources[1], 0.25)
    del spline, sources
    gc.collect()

    assert knot_ref() is not None
    assert coefficient_ref() is not None
    assert shifted(times[5] + 0.25) == pytest.approx(np.cos(times[5]))

    del shifted
    gc.collect()
    assert knot_ref() is None
    assert coefficient_ref() is None


@pytest.mark.parametrize("complex_values", [False, True])
def test_tabulated_bspline_composed_shift_preserves_exact_endpoints(
    complex_values,
):
    times = np.array([
        0.13372910821753037,
        0.17461611,
        0.24417265,
        0.24853221264216468,
    ])
    values = np.array([2.0, 3.0, 4.0, 5.0])[:, np.newaxis]
    if complex_values:
        values = values.astype(np.complex128) * (1.0 + 0.25j)
    spline = spi.make_interp_spline(times, values, k=3, axis=0)
    factory = (
        native_sampler.create_tabulated_bspline_bank
        if complex_values
        else native_sampler.create_tabulated_real_bspline_bank
    )
    source = factory(spline.t, spline.c)[0]
    shifted = native_sampler.shift_tabulated_bspline(source, 0.031)
    shifted = native_sampler.shift_tabulated_bspline(shifted, -0.013)

    assert shifted(times[0] + 0.018) == pytest.approx(values[0, 0])
    assert shifted(times[-1] + 0.018) == pytest.approx(values[-1, 0])


def test_tabulated_cubic_factories_release_heap_type_references():
    breaks = np.array([0.0, 1.0], dtype=np.float64)
    complex_coefficients = np.zeros((4, 1), dtype=np.complex128)
    real_coefficients = np.zeros((4, 1, 2), dtype=np.float64)
    complex_type = native_sampler.TabulatedCubic
    real_type = native_sampler.TabulatedRealCubic
    bspline_type = native_sampler.TabulatedBSpline
    real_bspline_type = native_sampler.TabulatedRealBSpline
    bspline_knots = np.array(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    complex_bspline_coefficients = np.zeros((4, 2), dtype=np.complex128)
    real_bspline_coefficients = np.zeros((4, 2), dtype=np.float64)

    complex_source = native_sampler.create_tabulated_cubic(
        breaks,
        complex_coefficients,
    )
    real_sources = native_sampler.create_tabulated_real_cubic_bank(
        breaks,
        real_coefficients,
    )
    bspline_sources = native_sampler.create_tabulated_bspline_bank(
        bspline_knots,
        complex_bspline_coefficients,
    )
    real_bspline_sources = native_sampler.create_tabulated_real_bspline_bank(
        bspline_knots,
        real_bspline_coefficients,
    )
    del complex_source, real_sources, bspline_sources, real_bspline_sources
    gc.collect()

    complex_refcount = sys.getrefcount(complex_type)
    real_refcount = sys.getrefcount(real_type)
    bspline_refcount = sys.getrefcount(bspline_type)
    real_bspline_refcount = sys.getrefcount(real_bspline_type)
    for _ in range(100):
        complex_source = native_sampler.create_tabulated_cubic(
            breaks,
            complex_coefficients,
        )
        real_sources = native_sampler.create_tabulated_real_cubic_bank(
            breaks,
            real_coefficients,
        )
        bspline_sources = native_sampler.create_tabulated_bspline_bank(
            bspline_knots,
            complex_bspline_coefficients,
        )
        real_bspline_sources = (
            native_sampler.create_tabulated_real_bspline_bank(
                bspline_knots,
                real_bspline_coefficients,
            )
        )
        del complex_source, real_sources, bspline_sources, real_bspline_sources
    gc.collect()

    assert sys.getrefcount(complex_type) == complex_refcount
    assert sys.getrefcount(real_type) == real_refcount
    assert sys.getrefcount(bspline_type) == bspline_refcount
    assert sys.getrefcount(real_bspline_type) == real_bspline_refcount
