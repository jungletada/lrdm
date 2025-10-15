#!/usr/bin/env bash
set -e
set -x


version=${1:-"full"}
batchsize=${2:-4}

python script/depth/train.py \
    --config config/train_weather_warmup.yaml \
    --training_version ${version} \
    --max_train_batch_size ${batchsize}

python script/depth/train.py \
    --config config/train_weather_finetune.yaml \
    --training_version ${version} \
    --max_train_batch_size ${batchsize}
