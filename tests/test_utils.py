import anndata
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

from ison_spatial import utils


def test_get_device():
    assert isinstance(utils.get_device(), torch.device)


def test_pearson_correlation_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.random((20, 4))
    b = rng.random((20, 4))
    got = utils.pearson_correlation(torch.tensor(a), torch.tensor(b), 0).numpy()
    expected = [pearsonr(a[:, j], b[:, j])[0] for j in range(4)]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_convert_to_tensor_dense_and_sparse_are_float32():
    x = np.random.default_rng(0).random((5, 4))
    dense = anndata.AnnData(x)
    sparse = anndata.AnnData(csr_matrix(x))
    for ad in (dense, sparse):
        t = utils.convert_to_tensor(ad)
        assert t.shape == (5, 4) and t.dtype == torch.float32
        np.testing.assert_allclose(t.numpy(), x, rtol=1e-6)


def test_convert_to_tensor_transpose():
    ad = anndata.AnnData(np.random.default_rng(0).random((5, 4)))
    t = utils.convert_to_tensor(ad, transpose=True)
    assert t.shape == (4, 5) and t.dtype == torch.float32
