#!/bin/bash

data_dir="./data/imagenet/images"
output_dir="./output/imagenet/K65536_default_exp1"
python -m torch.distributed.launch --nproc_per_node=4 \
    train.py \
    --data-dir ${data_dir} \
    --dataset imagenet \
    --nce-k 65536 \
    --output-dir ${output_dir} \
    --data-format image \
    --amp-opt-level O2


python -m torch.distributed.launch --nproc_per_node=4 \
    eval.py \
    --dataset imagenet \
    --data-dir ${data_dir} \
    --pretrained-model ${output_dir}/current.pth \
    --output-dir ${output_dir}/eval \
    --data-format image

