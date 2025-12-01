import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.sparse import issparse
from scipy.stats import pearsonr
import anndata as ad
 

import utils
import preprocess
import tri_nmf

cuda0 =  utils.get_device()
torch.cuda.empty_cache()

rna=preprocess.read_data("./rnap21.h5ad")
atac=preprocess.read_data("./atacp21.h5ad")
sprna=preprocess.read_data("./sprnap21.h5ad")

labels=pd.read_csv("./P21.tsv", index_col=0, delimiter="\t")
print(labels.index)
rna.obs=rna.obs.merge(labels, left_index=True, right_index=True, how="inner")
sprna.obs=sprna.obs.merge(labels, left_index=True, right_index=True, how="left")
atac.obs=atac.obs.merge(labels, left_index=True, right_index=True, how="inner")

sc.pp.log1p(rna)
sc.pp.log1p(atac)
sc.pp.log1p(sprna)

print(rna.shape, atac.shape, sprna.shape)

#preprocessing
trainmu, testmu, trainsp=preprocess.prep(rna, atac, sprna, raw=False, merge=True)
print(trainmu.shape, testmu.shape, trainsp.shape)

import tangram as tg
tg.pp_adatas(trainmu, trainsp, gene_to_lowercase = False)
tag=tg.map_cells_to_space(trainmu, trainsp, device=cuda0, mode='clusters', cluster_label='RNA_clusters')
pred_atac=tg.project_genes(tag, testmu, cluster_label='RNA_clusters')

pred_atac.write_h5ad("/data2/duren_lab/idebnat/palmetto/Modest/Chrom_spatial/model1/Benchmark/P21/tangram21_pred_new_clusters.h5ad")
