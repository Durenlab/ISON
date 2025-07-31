import scanpy as sc
import pandas as pd
import torch
import argparse
import numpy as np
import random
import os
from scipy.stats import pearsonr
from scipy.sparse import issparse
import magic

import sys
utils_path = os.path.join(os.path.dirname(__file__), 'project')
sys.path.append(utils_path)
import utils
import preprocess
import tri_nmf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--st_ATAC", help="Path to spatial transcriptomics ATAC dataset (spots x peaks)", type=str, default=None, required=True)
    parser.add_argument("-o",'--output-dir', help='Output directory', default='.')
    parser.add_argument('--filename', '-ofp', help="Output file name", default="spatial", type=str)
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    dataset_path = os.path.join(output_dir, args.filename + '-ATAC' +'.h5ad')
    
    actual_atac=preprocess.read_data(args.st_ATAC)
    pred_atac=preprocess.read_data(dataset_path)
    
    sc.pp.log1p(actual_atac)
    #Magic imputation for ATAC
    magic_operator = magic.MAGIC()
    X_magic = magic_operator.fit_transform(actual_atac.X)
    actual_atac.X=X_magic

    idx=actual_atac.var.index.intersection(pred_atac.var.index)
    pred_atac=pred_atac[:,pred_atac.var.index.isin(idx)]
    actual_atac=actual_atac[:,actual_atac.var.index.isin(idx)]
    
    idx=actual_atac.obs.index.intersection(pred_atac.obs_names)
    pred_atac=pred_atac[pred_atac.obs.index.isin(idx),:]
    actual_atac=actual_atac[actual_atac.obs.index.isin(idx),:]
    
    #preprocess
    test_sp=utils.convert_to_tensor(actual_atac, transpose=False)
    pred_sp=utils.convert_to_tensor(pred_atac, transpose=False)

    pcc1=utils.pearson_correlation(pred_sp, test_sp, d=0)
    #pcc1[pcc1.isnan()] = 0
    #pcc1=pcc1[~torch.isinf(pcc1)]

    pcc2=utils.pearson_correlation(pred_sp, test_sp, d=1)
    #pcc2[pcc2.isnan()] = 0
    #pcc2=pcc2[~torch.isinf(pcc2)]
     
    print("PCC-gene:", pcc1.mean().data.item(), "PCC-cell:", pcc2.mean().data.item())
    
    pcc1_num=pcc1.cpu().detach().numpy()
    
    if issparse(actual_atac.X):
        actual_data = actual_atac.X.toarray()
    else:
        actual_data = actual_atac.X

    if issparse(pred_atac.X):
        pred_data = pred_atac.X.toarray()
    else:
        pred_data = pred_atac.X

    # Flatten the arrays
    actual_data_flattened = actual_data.flatten()
    pred_data_flattened = pred_data.flatten()

    # Compute Pearson correlation coefficient
    corr_coefficient, p_value = pearsonr(actual_data_flattened, pred_data_flattened)

    # Print the result
    print("Pearson correlation coefficient:", corr_coefficient)
    print("P-value:", p_value)
        
    