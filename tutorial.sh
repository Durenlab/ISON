#!/bin/bash

SCRNA="/Users/ishitadebnath/ISON-copy/data/rnap21.h5ad"
SCATAC="/Users/ishitadebnath/ISON-copy/data/atacp21.h5ad"
ST="/Users/ishitadebnath/ISON-copy/data/sprnap21.h5ad"
COORDS="/Users/ishitadebnath/ISON-copy/data/coordsp21.csv"
OUTPUT_DIR="./output_p21"


LAMBDA1=15
LAMBDA2=0.001
K=8
BATCH_SIZE=512

echo
echo "PHASE 1: TRAIN NMF MODEL"
echo

python train-ison.py --scRNA "$SCRNA" \
                        --scATAC "$SCATAC" \
                        --ST "$ST" \
                        --coords "$COORDS" \
                        --output-dir "$OUTPUT_DIR" \
                        --lambda1 "$LAMBDA1" \
                        --lambda2 "$LAMBDA2" \
                        --K "$K" \
                        --batch_size "$BATCH_SIZE"\

echo
echo "PHASE 2: EVALUATION"
echo

SP_ATAC="/Users/ishitadebnath/ISON-copy/data/spatacp21.h5ad"

python evaluation.py --st_ATAC "$SP_ATAC" \
                         -o "$OUTPUT_DIR" \
                        


