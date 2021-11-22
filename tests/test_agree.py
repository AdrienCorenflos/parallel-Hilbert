import jax
import numpy as np
import pytest

import hilbert_sort.jax as jax_backend
import hilbert_sort.numba as np_backend


@pytest.fixture(scope="module", autouse=True)
def config_pytest():
    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dim_x", [2, 3, 4])
@pytest.mark.parametrize("N", [150, 250])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_random_agree(dim_x, N, seed):
    np.random.seed(seed)
    x = np.random.randn(N, dim_x)
    np_res = np_backend.hilbert_sort(x)
    jax_res = jax_backend.hilbert_sort(x)
    np.testing.assert_allclose(np_res, jax_res)


@pytest.mark.parametrize("nDests", [2, 3, 4, 5])
@pytest.mark.parametrize("N", [150, 250])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_transpose_bits(nDests, N, seed):
    np.random.seed(seed)
    x = np.random.randint(0, 150021651, (5,))
    np_res = np_backend.transpose_bits(x, nDests)
    jax_res = jax_backend.transpose_bits(x, nDests)
    np.testing.assert_allclose(np_res, jax_res)


@pytest.mark.parametrize("nDests", [5, 7, 12])
@pytest.mark.parametrize("N", [150, 250])
@pytest.mark.parametrize("seed", [0, 42, 666])
def test_unpack_coords(nDests, N, seed):
    np.random.seed(seed)
    x = np.random.randint(0, 150021651, (nDests,))
    max_int = 150021651
    np_res = np_backend.unpack_coords(x)
    jax_res = jax_backend.unpack_coords(x, max_int)
    np.testing.assert_allclose(np_res, jax_res)


def test_gray_decode():
    for n in range(5, 1_000):
        np_res = np_backend.gray_decode(n)
        jax_res = jax_backend.gray_decode(n)
        np.testing.assert_allclose(np_res, jax_res)
