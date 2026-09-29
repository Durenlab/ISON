import scanpy as sc
import os
import numpy as np
from scipy.sparse import issparse

def read_data(filepath, dlim=None):
    if "Visium" in filepath:
        if filepath.endswith(".h5"):
            adata=sc.read_10x_h5(filepath)
        elif os.path.isdir(filepath):
            adata=sc.read_10x_mtx(filepath)
            
    else:
        if filepath.endswith(".h5ad"):
            adata=sc.read_h5ad(filepath)
        elif filepath.endswith(".h5"):
            adata=sc.read_10x_h5(filepath)
        elif filepath.endswith((".csv", ".tsv")):
            if filepath.endswith(".tsv"):
                adata=sc.read_text(filepath)
            else:
                adata=sc.read_csv(filepath)
    return adata

#inner join
def prep(geneexp, peaks, spgene, merge=True, raw=False):
    if raw:
        geneexp=filter_mat(geneexp)
        peaks=filter_mat(peaks)
        spgene=filter_mat(spgene)
      
    if merge:
        
        #barcodes

        idx=geneexp.obs.index.intersection(peaks.obs_names)
        geneexp=geneexp[geneexp.obs.index.isin(idx)]
        Y=peaks[peaks.obs.index.isin(idx)]

        #features
        
        geneexp=geneexp[:,~geneexp.var.index.duplicated()]
        spgene=spgene[:,~spgene.var.index.duplicated()]

        idx=geneexp.var.index.intersection(spgene.var_names)
        X=geneexp[:,geneexp.var.index.isin(idx)]
        SX=spgene[:,spgene.var.index.isin(idx)]
    
        return X, Y, SX
    
    return geneexp, peaks, spgene
    

def filter_mat(X):
    
    if issparse(X.X)==True:
        a1 = np.sum(X.X.A > 0, axis=0)
    else:
        a1 = np.sum(X.X > 0, axis=0)

    # Calculate the 80th percentile
    cut1 = np.percentile(a1, 80)

    # Filter cells based on the calculated percentiles
    X = X[:, a1 > cut1]
     
    return X


def remove_mito(adata):
    mt=adata.var_names.str.lower().str.startswith('mt')
    keep=np.invert(mt)
    adata=adata[:, keep].copy()

    return adata