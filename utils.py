import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import issparse

class CustomDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        assert len(self.data_tensor) == len(self.label_tensor)
    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        data = self.data_tensor[idx]
        label = self.label_tensor[idx]
        return data, label

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
        return torch.device("cuda:0")
    else:
        print("GPU is not available")
        return torch.device('cpu')