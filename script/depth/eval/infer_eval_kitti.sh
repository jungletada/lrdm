#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
ckpt=${1:-"checkpoint/marigold-depth-v1-1"}
finetune_ckpt=${2:-"checkpoint/marigold-depth-v1-1"}
# finetune_ckpt=${2:-"output/train_weather_depth/checkpoint/iter_025000"}
subfolder=${3:-"eval"}
n_ensemble=${4:-1}
output_child_dir=${5:-"latent_restore"}

python script/depth/infer.py \
    --base_checkpoint $ckpt \
    --seed 2026 \
    --base_data_dir data/kitti \
    --denoise_steps 2 \
    --ensemble_size ${n_ensemble} \
    --processing_res 0 \
    --dataset_config config/dataset_depth/data_kitti_eigen_test.yaml \
    --output_dir output/${subfolder}/kitti_eigen_test/${output_child_dir} \
    --version restore \
    --finetune_checkpoint ${finetune_ckpt}


python script/depth/eval.py \
    --base_data_dir data/kitti \
    --dataset_config config/dataset_depth/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir output/${subfolder}/kitti_eigen_test/${output_child_dir} \
    --output_dir output/${subfolder}/kitti_eigen_test/eval_metric
