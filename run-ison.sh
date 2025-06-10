#!/bin/bash

SCRNA="/Users/idebnat/Desktop/ISON/rnap21.h5ad"
SCATAC="/Users/idebnat/Desktop/ISON/atacp21.h5ad"
ST="/Users/idebnat/Desktop/ISON/sprnap21.h5ad"
COORDS="/Users/idebnat/Desktop/ISON/coordsp21.csv"
OUTPUT_DIR="./output_p21"


LAMBDA1=15
LAMBDA2=0.001
K=8
BATCH_SIZE=512

echo
echo "PHASE 1: TRAIN NMF MODEL"
echo

python3 train-ison.py --scRNA "$SCRNA" \
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

SP_ATAC="/Users/idebnat/Desktop/ISON/spatacp21.h5ad"

python3 evaluation.py --st_ATAC "$SP_ATAC" \
                         -o "$OUTPUT_DIR" \
                        


