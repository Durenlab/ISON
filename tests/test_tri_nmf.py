import numpy as np
import torch

from ison import tri_nmf


def test_adjacency_is_symmetric_binary():
    coords = np.array([[i, j] for i in range(6) for j in range(6)], dtype=float)
    adj = tri_nmf.compute_adjacency_with_limits(coords, lattice="square")
    assert adj.shape == (36, 36)
    assert (adj == adj.T).all()
    assert set(np.unique(adj)) <= {0, 1}
    assert adj.diagonal().sum() == 0


def test_kl_nmf_runs_and_returns_finite_nonnegative_factors():
    torch.manual_seed(0)
    peaks, genes, cells, spots, k = 30, 20, 16, 16, 3
    peak_o = torch.rand(peaks, cells)
    x1 = torch.rand(genes, cells)
    x2 = torch.rand(genes, spots)
    coords = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    w1, w2, h1, h2 = tri_nmf.KL_NMF(
        peak_o, x1, x2, k, maxiter=3, lambda1=1.0, lambda2=0.001,
        batch_size=4, coords=coords, device=torch.device("cpu"),
    )
    # two platform-effect components are appended to K
    assert w1.shape == (peaks, k + 2) and w2.shape == (genes, k + 2)
    assert h1.shape == (k + 2, cells) and h2.shape == (k + 2, spots)
    for m in (w1, w2, h1, h2):
        assert torch.isfinite(m).all() and (m >= 0).all()
