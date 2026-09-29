import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ison import preprocess


def _adata(x, obs, var):
    return anndata.AnnData(x, obs=pd.DataFrame(index=obs), var=pd.DataFrame(index=var))


def test_remove_mito():
    ad = _adata(np.ones((3, 4)), ["c1", "c2", "c3"], ["MT-CO1", "mt-nd1", "ACTB", "GAPDH"])
    out = preprocess.remove_mito(ad)
    assert list(out.var_names) == ["ACTB", "GAPDH"]


def test_filter_mat_dense_keeps_features_above_80th_percentile():
    x = np.zeros((10, 10))
    for j in range(10):
        x[: j + 1, j] = 1  # feature j is detected in j+1 cells
    ad = _adata(x, [f"c{i}" for i in range(10)], [f"g{j}" for j in range(10)])
    out = preprocess.filter_mat(ad)
    assert list(out.var_names) == ["g8", "g9"]


def test_filter_mat_sparse_matches_dense():
    x = np.zeros((10, 10))
    for j in range(10):
        x[: j + 1, j] = 1
    ad = _adata(csr_matrix(x), [f"c{i}" for i in range(10)], [f"g{j}" for j in range(10)])
    out = preprocess.filter_mat(ad)
    assert list(out.var_names) == ["g8", "g9"]


def test_prep_inner_join():
    rng = np.random.default_rng(0)
    rna = _adata(rng.random((4, 5)), ["a", "b", "c", "d"], ["g1", "g2", "g3", "g4", "g5"])
    atac = _adata(rng.random((3, 6)), ["b", "c", "d"], [f"p{i}" for i in range(6)])
    st = _adata(rng.random((7, 4)), [f"s{i}" for i in range(7)], ["g2", "g3", "g5", "gX"])
    x, y, sx = preprocess.prep(rna, atac, st)
    assert list(x.obs_names) == list(y.obs_names) == ["b", "c", "d"]
    assert list(x.var_names) == list(sx.var_names) == ["g2", "g3", "g5"]
