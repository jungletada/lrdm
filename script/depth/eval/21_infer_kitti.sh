#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-depth-v1-1"}
subfolder=${2:-"eval"}
n_ensemble=${3:-1}


python script/depth/infer.py \
    --base_checkpoint $ckpt \
    --seed 2025 \
    --base_data_dir data/kitti \
    --denoise_steps 2 \
    --ensemble_size ${n_ensemble} \
    --processing_res 0 \
    --dataset_config config/dataset_depth/data_kitti_eigen_test.yaml \
    --output_dir output/${subfolder}/kitti_eigen_test/prediction
