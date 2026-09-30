import numpy as np
import pytest

from tama import native_sampler


def test_native_sampler_supports_more_than_one_rank_mask_word_without_65_ranks():
    rank_word_bits = np.dtype(np.uintp).itemsize * 8
    nproc = 2 * rank_word_bits + 3
    supported_ranks = [
        0,
        rank_word_bits - 1,
        rank_word_bits,
        rank_word_bits + 1,
        2 * rank_word_bits,
        nproc - 1,
        rank_word_bits,
    ]

    word_bits, word_count, popcount, membership = (
        native_sampler._support_mask_summary_for_testing(
            nproc,
            supported_ranks,
        )
    )

    assert word_bits == rank_word_bits
    assert word_count == 3
    assert popcount == len(set(supported_ranks))
    assert np.flatnonzero(membership).tolist() == sorted(set(supported_ranks))


def test_native_adjoint_signal_checkpoint_processes_pending_sigint():
    with pytest.raises(KeyboardInterrupt):
        native_sampler._check_native_adjoint_signal_for_testing()


if __name__ == "__main__":
    test_native_sampler_supports_more_than_one_rank_mask_word_without_65_ranks()
