import scanpy as sc
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse


def convert_to_tensor(adata, transpose=False, device=None):
    # Cast to float32 before densifying: adata.X is often float64, and
    # densifying a large sparse ATAC matrix (e.g. peaks x cells) at float64
    # doubles peak host RAM versus the float32 the GPU training loop uses,
    # which OOM'd on samples with ~600k peaks x ~10k cells.
    if transpose== True:
        adata = torch.from_numpy(adata.X.T.astype(np.float32, copy=False))
    else:
        if issparse(adata.X):
            adata = torch.from_numpy(adata.X.astype(np.float32).toarray())
        else:
            adata = torch.from_numpy(np.asarray(adata.X, dtype=np.float32))
    if device:
        adata=adata.to(device)

    return adata

def pearson_correlation(A, B, d):
    # Calculate mean along the columns (axis=0)
    mean_A = torch.mean(A, dim=0)
    mean_B = torch.mean(B, dim=0)

    # Calculate covariance
    cov_AB = torch.sum((A - mean_A) * (B - mean_B), dim=d)

    # Calculate standard deviation
    std_A = torch.sqrt(torch.sum((A - mean_A) ** 2, dim=d))
    std_B = torch.sqrt(torch.sum((B - mean_B) ** 2, dim=d))

    # Calculate Pearson correlation coefficient
    correlation = cov_AB / (std_A * std_B)

    return correlation

def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        return torch.device("cuda")
    else:
        print("GPU is not available")
        return torch.device('cpu')