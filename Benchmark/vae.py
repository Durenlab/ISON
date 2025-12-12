import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import pandas as pd
import torch.nn.functional as F
import random
import seaborn as sns
import magic

import sys
sys.path.insert(0, './modules_alz/')
import utils
import preprocess
import tri_nmf


bemodel=torch.load("./bemodel_p21.pt", map_location=torch.device('cpu'))
bemodel=bemodel.to(dev)

model2=torch.load("./model2_p21.pt", map_location=torch.device('cpu'))
model2=model2.to(dev)

rna=preprocess.read_data("./rnap21.h5ad")
atac=preprocess.read_data("./atacp21.h5ad")
sprna=preprocess.read_data("./sprnap21.h5ad")
spatac=preprocess.read_data("./spatacp21.h5ad")

#inner join 
rna, atac, sprna, spatac=preprocess.prep(rna, atac, sprna, spatac)

#converting to tensors

train_mu=utils.convert_to_tensor(utils.get_zscore(rna),transpose=False)
test_mu=utils.convert_to_tensor(utils.get_zscore(atac), transpose=False)

train_sp=utils.convert_to_tensor(utils.get_zscore(sprna),transpose=False)
test_sp=utils.convert_to_tensor(utils.get_zscore(spatac), transpose=False)

lat=10
spat_new=bemodel.forward(train_sp.to(dtype=torch.float32), lat)[1]
mult_new=bemodel.forward(train_mu.to(dtype=torch.float32), lat)[1]

zmu = F.normalize(spat_new, dim=1)
zsp = F.normalize(mult_new, dim=1)
preddec=model2.decoder(spat_new)
