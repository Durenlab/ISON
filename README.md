# Integrated Spatial Omics Network from spatial transcriptomics and single cell multiome data (ISON)

ISON is a computational tool developed to infer spatial chromatin accessibility by integrating spatial transcriptomics with single-cell multiomics data. It performs dimension reduction and learns the embeddings in a joint latent space and uses these embeddings to predict chromatin accessibility data in a spatial context.

## Input

- Single-cell multiome (scRNA + scATAC seq)
- Spatial transcriptomics data
- Coordinates file with only x and y coordinates

## Installation

- Python >= 3.8 
The `environment.yml` file lists all the libraries required to run ISON, with `PyTorch` configured for CPU-only usage.
If you have a GPU and wish to enable CUDA support, please install the appropriate version of `PyTorch` by following the instructions at https://pytorch.org/get-started/locally.

## Running the model

To run ISON, run the `run-ison.sh` script.

Make sure the `.sh` file is executable, then run it from your terminal:

```
chmod +x run-ison.sh
./run-ison.sh
```

You can change the hyperparameters $\lambda_1$, $\lambda_2$, $K$ and `batch_size`.
