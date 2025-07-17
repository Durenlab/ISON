import scanpy as sc
import pandas as pd
import torch
import argparse
import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import sys


utils_path = os.path.join(os.path.dirname(__file__), 'project')
sys.path.append(utils_path)

import utils
import preprocess
import tri_nmf

dev= utils.get_device()

if __name__ == '__main__':
    #Parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scRNA", help="Path to scRNA-seq dataset (cells x genes)", type=str, default=None, required=True)
    parser.add_argument("--scATAC", help="Path to scATAC-seq dataset (cells x peaks)", type=str, default=None, required=True)
    parser.add_argument("--ST", help="Path to spatial transcriptomics dataset (spots x genes)", type=str, default=None, required=True)
    parser.add_argument("--coords", help="Path to spatial transcriptomics coordinates file (spots x 2)", type=str, default=None, required=True)
    parser.add_argument('--output-dir', help='Output directory', default='.')
    parser.add_argument('--lambda1', help='Hyperparameter tuning for gene expression matrices', default=15.0, type=float)
    parser.add_argument('--lambda2', help='Hyperparameter tuning for spatial matrix', default=0.001, type=float)
    parser.add_argument('--K', help='Number of components', default=8, type=int)
    parser.add_argument('--batch_size', help='Size of mini batches', default=512, type=int)
    parser.add_argument('--filename', '-ofp', help="Output file name", default="spatial", type=str)

    args = parser.parse_args()

    #Load datasets
    rna=preprocess.read_data(args.scRNA)
    atac=preprocess.read_data(args.scATAC)
    sprna=preprocess.read_data(args.ST)

    #coords
    coords=pd.read_csv(args.coords, index_col=0)
    print(coords.columns)
    coords=coords[["x", "y"]]
    
    #preprocessing
    rna=preprocess.remove_mito(rna) 
    sprna=preprocess.remove_mito(sprna) 
   
    
    trainmu, testmu, trainsp=preprocess.prep(rna, atac, sprna, raw=False, merge=True)

    train_mu=utils.convert_to_tensor(trainmu,transpose=False)
    test_mu=utils.convert_to_tensor(testmu, transpose=False)
    train_sp=utils.convert_to_tensor(trainsp,transpose=False)
    
    train_sp=train_sp.to(dev, dtype=torch.float32)
    train_mu=train_mu.to(dev, dtype=torch.float32)
    test_mu=test_mu.to(dev, dtype=torch.float32)
    
    coords=coords[coords.index.isin(trainsp.obs.index)]

    lambda1= args.lambda1
    lambda2= args.lambda2
    k=args.K

    #Model train 
    
    ### inputs are anndata.X
    print("Training...")
    W1,W2,H1,H2=tri_nmf.KL_NMF(test_mu.t(), train_mu.t(), train_sp.t(), K=k, batch_size=args.batch_size, maxiter=50, iterLoss=1, lambda1=lambda1, lambda2=lambda2, coords=coords,  device=dev)
    print("Training complete.")
    W1=W1[:,:k]
    H2=H2[:k,:]
    O2_hat=torch.mm(W1,H2)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'weights.pt')
    torch.save({'W1': W1, 'H1': H1, 'W2': W2, 'H2': H2}, output_filename)
    print(f"Weights file saved to {output_filename}")

    #Save spatial-Atac
    pred_atac=anndata.AnnData(X=O2_hat.t().to('cpu').detach().numpy(), 
    obs=pd.DataFrame(index=trainsp.obs.index),  
    var=pd.DataFrame(index=testmu.var.index))

    output_filename = os.path.join(output_dir, args.filename + '-ATAC' +'.h5ad')
    pred_atac.write_h5ad(output_filename)

    print(f"spatial-ATAC saved to {output_filename}")
    
    #Save ST
    W2=W2[:,:k]
    X2_hat=torch.mm(W2,H2)
    
    pred_atac=anndata.AnnData(X=X2_hat.t().to('cpu').detach().numpy(), 
    obs=pd.DataFrame(index=trainsp.obs.index),  
    var=pd.DataFrame(index=trainsp.var.index))

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, args.filename + '-RNA' +'.h5ad')
    pred_atac.write_h5ad(output_filename)

    print(f"Denoised RNA saved to {output_filename}")
