import ison
from ison import preprocess, tri_nmf, utils


def test_public_api_exists():
    for name in ("read_data", "prep", "filter_mat", "remove_mito"):
        assert callable(getattr(preprocess, name))
    for name in ("KL_NMF", "compute_adjacency_with_limits"):
        assert callable(getattr(tri_nmf, name))
    for name in ("convert_to_tensor", "pearson_correlation", "get_device"):
        assert callable(getattr(utils, name))


def test_tri_nmf_uses_package_utils():
    assert tri_nmf.utils is utils
    assert ison.__file__ is not None
