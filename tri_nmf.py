import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import random
from scipy.sparse import csr_matrix, diags, lil_matrix
from sklearn.model_selection import KFold
import tracemalloc
import time
from sklearn.neighbors import kneighbors_graph
import sys
sys.path.insert(0, '/data2/duren_lab/idebnat/palmetto/Modest/Chrom_spatial/model1/')
import utils
import gc
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.spatial import distance_matrix

# torch.cuda.memory_summary(device=None, abbreviated=False)
        
def KL_NMF(PeakO, X1, X2, K, maxiter, iterLoss=0, lambda1=1, lambda2=1, batch_size=1024, W10=None, W20=None, H10=None, H20=None, coords=None, device=None, dtype=torch.float32):
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if iterLoss == 0:
        iterLoss = 0
    if lambda1 is None or lambda2 is None:
        lambda1 = 1
        lambda2 = 1
        
    if W10 is None or W20 is None or H10 is None or H20 is None:
        generator = torch.Generator(device=device).manual_seed(600)
        W10 = torch.rand(PeakO.shape[0], K, dtype=dtype, device=device, generator=generator)
        W20 = torch.rand(X1.shape[0], K, dtype=dtype, device=device, generator=generator)
        H10 = torch.rand(K, PeakO.shape[1], dtype=dtype, device=device, generator=generator)
        H20 = torch.rand(K, X2.shape[1], dtype=dtype, device=device, generator=generator)    
        
    eps=1e-9

	#platform effects
    V1=torch.mean(X1, dim=1).unsqueeze(1)
    V2=torch.mean(X2, dim=1).unsqueeze(1)
    V=torch.cat((V1,V2), dim=1).to(device=device, dtype=dtype) #merge gene expression

    Vpeak=torch.mean(PeakO, dim=1).unsqueeze(1)
    Vp=torch.cat((Vpeak, Vpeak), dim=1).to(device=device, dtype=dtype) #merge peaks

    F1=torch.zeros((2, X1.shape[1]), device=device, dtype=dtype)
    F1[0,:] =F1[0,:] + 1
    F2=torch.zeros((2, X2.shape[1]), device=device, dtype=dtype)
    F2[1,:] =F2[1,:] + 1
    
    W10=torch.cat((W10, Vp), 1)
    W20=torch.cat((W20, V), 1)
    H10=torch.cat((H10, F1), 0)
    H20=torch.cat((H20, F2), 0)

    eps=1e-9
    eps1=1e-4
    l2= torch.sum(X1 * torch.log((X1 + eps1)/ (eps1)) - X1 )
    l3= torch.sum(X2 * torch.log((X2 + eps1)/ (eps1)) - X2 )
    l1= torch.sum(PeakO * torch.log((PeakO + eps1)/ (eps1)) - PeakO)
   
        
    mb_size = batch_size 
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    n_batch = min(n1, n2) // mb_size
    
    onesW10 = torch.ones(PeakO.shape[0], 1, dtype=dtype, device=device)
    onesW20 = torch.ones(X1.shape[0], 1, dtype=dtype, device=device)
    onesH10 = torch.ones(1, PeakO.shape[1], dtype=dtype, device=device)
    onesH20 = torch.ones(1, X2.shape[1], dtype=dtype, device=device)
    
    kf = KFold(n_batch)
    H1 = H10.clone()
    H2 = H20.clone()  
    
    lm_splits = []
    peakosplits = []
    x1splits = []
    x2splits = []
    
    mu1 = (l1 * lambda1)/ l2
    mu2 = (l1 * lambda1)/ l3

    adjacency_matrix=compute_adjacency_with_limits(coords, radius=100)
    degree_values = np.array(adjacency_matrix.sum(axis=1))  # Get degree sum as 1D array
    degree_matrix = diags(degree_values)   # Convert sum result to 1D array
    lapm = degree_matrix - adjacency_matrix
    lapm = np.asarray(lapm)
	
	#storing data into batches
    
    for i, (s1, s2) in enumerate(zip(kf.split(X1.t()), kf.split(X2.t()))):
        mb1 =s1[1]
        mb2 =s2[1]

        lm_splits.append(torch.tensor(lapm[mb2, :][:, mb2]).to(device=device, dtype=dtype))
        peakosplits.append(PeakO[:, mb1]) #.to(device=device, dtype=dtype)
        x1splits.append(X1[:, mb1])
        x2splits.append(X2[:, mb2])
        
    for iters in range(maxiter):

        for i, (s1, s2) in enumerate(zip(kf.split(X1.t()), kf.split(X2.t()))):
        
            mb1 = s1[1]
            mb2 = s2[1]

            peakoi = peakosplits[i].to(device=device, dtype=dtype)
            x1i = x1splits[i].to(device=device, dtype=dtype)
            x2i = x2splits[i].to(device=device, dtype=dtype)

            W10_H10_mb1 = torch.mm(W10, H10[:, mb1]) + eps
            W20_H10_mb1 = torch.mm(W20, H10[:, mb1]) + eps
            W20_H20_mb2 = torch.mm(W20, H20[:, mb2]) + eps

            # Update H1
            numer = W10.T @ (peakoi / W10_H10_mb1) + mu1 * W20.T @ (x1i / W20_H10_mb1)
            denom = W10.T @ onesW10 + mu1 * W20.T @ onesW20 + eps
            H1[:, mb1] = F.relu(H10[:, mb1] * (numer / denom))

            # Update H2
            numer = mu2 * W20.T @ (x2i / W20_H20_mb2) + 2 * lambda2 * H20[:, mb2] @ lm_splits[i]
            denom = mu2 * W20.T @ onesW20 + eps  
            H2[:, mb2] = F.relu(H20[:, mb2] * (numer / denom))

            # Update W1
            denom = onesH10[:, mb1] @ H1[:, mb1].T + eps
            numer = (peakoi / (W10 @ H1[:, mb1] + eps)) @ H1[:, mb1].T
            W1 = F.relu(W10 * (numer / denom))

            # Update W2
            denom = mu1 * (onesH10[:, mb1] @ H1[:, mb1].T) + mu2 * (onesH20[:, mb2] @ H2[:, mb2].T) + eps
            numer = mu1 * (x1i / (W20 @ H1[:, mb1] + eps)) @ H1[:, mb1].T + mu2 * (x2i / (W20 @ H2[:, mb2] + eps)) @ H2[:, mb2].T
            W2 = F.relu(W20 * (numer / denom))

            # Assign updated matrices
            W10, W20 = W1, W2
            H10, H20 = H1, H2

            # Free up memory
            del peakoi, x1i, x2i
            peakosplits[i] = peakosplits[i].cpu()
            x1splits[i] = x1splits[i].cpu()
            x2splits[i] = x2splits[i].cpu()
            torch.cuda.empty_cache()
            
        W20_H10 = torch.matmul(W20, H10)
        W20_H20 = torch.matmul(W20, H20)
        W10_H10 = torch.matmul(W10, H10)
        

        dnorm1 = (torch.sum(X1 * torch.log((X1 + eps) / (W20_H10 + eps)) - X1 + W20_H10)) / l2
        dnorm2 = (torch.sum(X2 * torch.log((X2 + eps) / (W20_H20 + eps)) - X2 + W20_H20)) / l3
        dnorm3 = (torch.sum(PeakO * torch.log((PeakO + eps) / (W10_H10 + eps)) - PeakO + W10_H10)) / l1      
        
        print(f"{iters}:{dnorm1}, {dnorm2}, {dnorm3}")
            
    return W1, W2, H1, H2

# for incorporating spatial information
def compute_adjacency_with_limits(coords, radius):
    n = len(coords)
    dist_matrix = distance_matrix(coords, coords)
    A = np.zeros((n, n), dtype=int)

    for i in range(n):
        dists = dist_matrix[i]
        neighbor_indices = np.argsort(dists)
        neighbor_indices = neighbor_indices[neighbor_indices != i]
        close_neighbors = neighbor_indices[dists[neighbor_indices] <= radius]

        num_close = len(close_neighbors)

        if num_close <= 3:
            max_neighbors = 3  # corner
        elif num_close <= 5:
            max_neighbors = 5  # edge
        else:
            max_neighbors = 8  # interior

        # Take closest N neighbors
        for j in close_neighbors[:max_neighbors]:
            A[i, j] = 1
            A[j, i] = 1  # make symmetric

    return A