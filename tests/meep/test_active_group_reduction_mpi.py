import meep as mp
import numpy as np
import pytest

from tama import native_sampler


def _local_values(offset: float) -> np.ndarray:
    return np.asarray(
        [
            [offset + 1.0j * (offset + 0.5), offset + 1.0 + 2.0j],
            [offset + 2.0 - 3.0j, offset + 3.0 + 4.0j],
        ],
        dtype=np.complex128,
    )


@pytest.mark.mpi2
def test_inplace_reduction_reuses_buffer_and_matches_full_group_sum():
    if mp.comm.Get_size() != 2:
        pytest.skip("requires exactly two MPI ranks")

    local = _local_values(10.0 * mp.my_rank())
    pointer = local.__array_interface__["data"][0]
    expected = _local_values(0.0) + _local_values(10.0)

    reduced = native_sampler.reduce_complex_grid_sum_inplace(local)

    assert reduced is local
    assert local.__array_interface__["data"][0] == pointer
    assert np.array_equal(local, expected)

    local_real = _local_values(10.0 * mp.my_rank()).real.copy()
    real_pointer = local_real.__array_interface__["data"][0]
    expected_real = _local_values(0.0).real + _local_values(10.0).real

    reduced_real = native_sampler.reduce_real_grid_sum_inplace(local_real)

    assert reduced_real is local_real
    assert local_real.__array_interface__["data"][0] == real_pointer
    assert np.array_equal(local_real, expected_real)

    out_of_place_input = _local_values(10.0 * mp.my_rank())
    unchanged = out_of_place_input.copy()
    out_of_place = native_sampler.reduce_complex_grid_sum(
        out_of_place_input
    )

    assert out_of_place is not out_of_place_input
    assert np.array_equal(out_of_place_input, unchanged)
    assert np.array_equal(out_of_place, expected)


@pytest.mark.mpi2
@pytest.mark.parametrize(
    "dtype",
    (np.float32, np.float64, np.complex64, np.complex128),
)
def test_forced_chunked_reduction_preserves_dtype_buffer_and_sum(dtype):
    if mp.comm.Get_size() != 2:
        pytest.skip("requires exactly two MPI ranks")

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    if np.issubdtype(dtype, np.complexfloating):
        base = base + 1.0j * (base + 0.5)
    local = np.asarray(base + 16.0 * mp.my_rank(), dtype=dtype)
    rank_zero = np.asarray(base, dtype=dtype)
    rank_one = np.asarray(base + 16.0, dtype=dtype)
    expected = rank_zero + rank_one
    pointer = local.__array_interface__["data"][0]

    reduced = native_sampler._reduce_grid_sum_inplace_for_testing(local, 3)

    assert reduced is local
    assert reduced.dtype == dtype
    assert local.__array_interface__["data"][0] == pointer
    assert np.array_equal(local, expected)

    empty = np.empty((0, 3), dtype=dtype)
    reduced_empty = native_sampler._reduce_grid_sum_inplace_for_testing(
        empty,
        3,
    )

    assert reduced_empty is empty
    assert reduced_empty.dtype == dtype
    assert reduced_empty.shape == (0, 3)


@pytest.mark.mpi2
def test_native_adjoint_exception_reaches_every_active_rank():
    if mp.comm.Get_size() != 2:
        pytest.skip("requires exactly two MPI ranks")

    with pytest.raises(KeyboardInterrupt):
        native_sampler._synchronize_native_adjoint_exception_for_testing(
            mp.my_rank() == 0,
        )


@pytest.mark.mpi4
def test_native_adjoint_exception_stays_within_active_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    try:
        if group_index == 0:
            with pytest.raises(KeyboardInterrupt):
                native_sampler._synchronize_native_adjoint_exception_for_testing(
                    mp.my_rank() == 0,
                )
        else:
            result = (
                native_sampler._synchronize_native_adjoint_exception_for_testing(
                    False,
                )
            )
            assert result is None
    finally:
        mp.end_divide_parallel()


@pytest.mark.mpi4
def test_inplace_reduction_stays_within_active_meep_process_group():
    if mp.comm.Get_size() != 4:
        pytest.skip("requires exactly four MPI ranks")

    group_index = mp.divide_parallel_processes(2)
    try:
        group_offset = 100.0 * group_index
        local = _local_values(group_offset + 10.0 * mp.my_rank())
        pointer = local.__array_interface__["data"][0]
        expected = (
            _local_values(group_offset) +
            _local_values(group_offset + 10.0)
        )

        reduced = native_sampler.reduce_complex_grid_sum_inplace(local)

        assert reduced is local
        assert local.__array_interface__["data"][0] == pointer
        assert np.array_equal(local, expected)

        local_real = _local_values(
            group_offset + 10.0 * mp.my_rank()
        ).real.copy()
        real_pointer = local_real.__array_interface__["data"][0]
        expected_real = (
            _local_values(group_offset).real
            + _local_values(group_offset + 10.0).real
        )

        reduced_real = native_sampler.reduce_real_grid_sum_inplace(local_real)

        assert reduced_real is local_real
        assert local_real.__array_interface__["data"][0] == real_pointer
        assert np.array_equal(local_real, expected_real)
    finally:
        mp.end_divide_parallel()
