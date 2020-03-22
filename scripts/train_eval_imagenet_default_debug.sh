#!/bin/bash

data_dir="./data/imagenet/images"
output_dir="./output/imagenet/K65536_default_exp1"
python -m torch.distributed.launch --nproc_per_node=4 \
    train_debug.py \
    --data-dir ${data_dir} \
    --dataset imagenet \
    --nce-k 65536 \
    --output-dir ${output_dir} \
    --data-format zip

