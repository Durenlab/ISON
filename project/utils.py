import scanpy as sc
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse


def convert_to_tensor(adata, transpose=False, device=None):
    if transpose== True:
        adata = torch.from_numpy(adata.X.T)
    else:
        if issparse(adata.X):
            adata = torch.from_numpy(adata.X.A)
        else:
            adata = torch.from_numpy(adata.X)
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