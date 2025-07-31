# Integrated Spatial Omics Network from spatial transcriptomics and single cell multiome data (ISON)

ISON is a computational tool developed to infer spatial chromatin accessibility by integrating spatial transcriptomics with single-cell multiomics data. It performs dimension reduction and learns the embeddings in a joint latent space and uses these embeddings to predict chromatin accessibility data in a spatial context.

## Input

- Single-cell multiome 
	- scRNA (cells x genes)
	- scATAC (cells x peaks)
- Spatial transcriptomics (spots x genes)
- Coordinates file with only x and y coordinates (spots x 2)

## Installation

- Python >= 3.8 
The `environment.yml` file lists all the libraries required to run ISON, with `PyTorch` configured for CPU-only usage.
If you have a GPU and wish to enable CUDA support, please install the appropriate version of `PyTorch` by following the instructions at <https://pytorch.org/get-started/locally>.

```
conda env create -f environment.yml
conda activate ison_env
```

## Running the model

To run ISON, run the `run-ison.py` .

```
python run-ison.py --scRNA "$SCRNA" \
                        --scATAC "$SCATAC" \
                        --ST "$ST" \
                        --coords "$COORDS" \
                        --output-dir "$OUTPUT_DIR" \
                        --lambda1 "$LAMBDA1" \
                        --lambda2 "$LAMBDA2" \
                        --K "$K" \
                        --batch_size "$BATCH_SIZE"\
```

The default parameters are set as: $\lambda_1$ = 15, $\lambda_2$ = 0.001, $K$ = 8 and `batch_size` = 512.
To evaluate ISON, there are toy datasets provided in the `data` folder in .h5ad format.  evaluation.py in `project` folder is provided to compute the Pearson correlation coefficient (PCC) between the true and predicted ATAC values.

## Tutorial

To run ISON using processed P21 data, run the `tutorial.sh` script. Processed P21 and P22 data is available at <https://drive.google.com/drive/folders/15LMxbizMdrBuPALeMwLvaeg5yc63zljh?usp=share_link>. The `evaluation.py` script computes the PCC between ground truth and predicted chromatin accessibility values for ISON's performance evaluation.

Make sure the .sh file is executable, then run it from your terminal:

```
chmod +x tutorial.sh
./tutorial.sh
```
