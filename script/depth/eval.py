# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from omegaconf import OmegaConf
from tabulate import tabulate
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import (
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from src.util import metric
from src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from src.util.metric import MetricTracker


eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]


def get_args():
     # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Metrics Evaluation"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory with predictions obtained from inference.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the config file of the evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base path to the datasets.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda.")
    return parser.parse_args()


def extract_group(filename):
    """
        Extract the group based on the filename
    """
    parts = filename.split('/')
    if parts[0] == 'fog':
        return parts[0] + parts[-2]  # eg., fog + 75m = fog75m
    elif len(parts) >= 1:
        return parts[0]
    else:
        return 'unknown'


def save_filled_depth(args, valid_mask, depth_raw, pred_name):
    """
        Save filled depth
    """
    valid_mask_1 = valid_mask.squeeze()
    depth_raw_1 = depth_raw.squeeze()
    # Create aligned_depth subdirectory and save aligned depth maps there
    aligned_depth_dir = os.path.join(args.output_dir, "align_depth_rgb")
    aligned_pred_path = os.path.join(aligned_depth_dir, pred_name)
    # Ensure the full directory structure exists
    os.makedirs(os.path.dirname(aligned_pred_path), exist_ok=True)
    aligned_depth = valid_mask_1 * depth_raw_1 + (1 - valid_mask_1) * depth_pred
    aligned_depth_png_path = aligned_pred_path.replace('.npy', '.png').replace('pred_', '')
    os.makedirs(os.path.dirname(aligned_depth_png_path), exist_ok=True)
    Image.fromarray((aligned_depth * 256.0).astype(np.uint16)).save(aligned_depth_png_path)
    

def save_prediction_heatmap(args, depth_pred, depth_raw, pred_name):
    """
        Save predicted and ground truth depth as heatmaps
    """
    def save_heatmap(tensor, filename):
        """
            Save depth_pred_ts and depth_raw_ts as heatmap PNGs
        """
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
        else:
            arr = tensor
        plt.figure()
        plt.axis('off')
        plt.imshow(arr, cmap="Spectral")
        plt.tight_layout(pad=0)
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    pred_heatmap_path = os.path.join(args.output_dir, pred_name.replace('.npy', '_pred.png'))
    gt_heatmap_path = os.path.join(args.output_dir, pred_name.replace('.npy', '_gt.png'))
    save_heatmap(depth_pred, 
                 pred_heatmap_path)
    save_heatmap(torch.from_numpy(depth_raw).to(device).squeeze(), 
                 gt_heatmap_path)
    
    
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # -------------------- Device --------------------
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")
    # -------------------- Data -----------------------
    cfg_data = OmegaConf.load(args.dataset_config)
    dataset = get_dataset(
        cfg_data, 
        base_data_dir=args.base_data_dir, 
        mode=DatasetMode.EVAL,
        join_split=False,
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    # -------------------- Eval metrics --------------------
    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    # -------------------- Per-sample metrics file --------------------
    per_sample_filename = os.path.join(args.output_dir, "per_sample_metrics.csv")
    # write title
    # with open(per_sample_filename, "w+") as f:
    #     f.write("filename,")
    #     f.write(",".join([m.__name__ for m in metric_funcs]))
    #     f.write("\n")

    # # -------------------- Evaluate --------------------
    # for data in tqdm(dataloader, desc="Evaluating"):
    #     # GT data
    #     depth_raw_ts = data["depth_raw_linear"].squeeze()
    #     valid_mask_ts = data["valid_mask_raw"].squeeze()
    #     rgb_name = data["rgb_relative_path"][0]

    #     depth_raw = depth_raw_ts.numpy()
    #     valid_mask = valid_mask_ts.numpy()

    #     depth_raw_ts = depth_raw_ts.to(device)
    #     valid_mask_ts = valid_mask_ts.to(device)

    #     # Load predictions
    #     rgb_basename = os.path.basename(rgb_name)
    #     pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".npy")
    #     pred_name = os.path.join(os.path.dirname(rgb_name), pred_basename)
    #     pred_path = os.path.join(args.prediction_dir, pred_name.replace("pred_", ""))
        
    #     if not os.path.exists(pred_path):
    #         logging.warning(f"Can't find prediction: {pred_path}")
    #         continue

    #     depth_pred = np.load(pred_path)
    #     # depth_pred = depth_pred.astype(np.float32)  # Convert to float32 to avoid float16 linalg issues
    #     # print(depth_pred.shape, type(depth_pred))
    #     # Align with GT using least square
    #     if "least_square" == args.alignment:
    #         depth_pred, scale, shift = align_depth_least_square(
    #             gt_arr=depth_raw,
    #             pred_arr=depth_pred,
    #             valid_mask_arr=valid_mask,
    #             return_scale_shift=True,
    #             max_resolution=args.alignment_max_res,
    #         )

    #     elif "least_square_disparity" == args.alignment:
    #         # convert GT depth -> GT disparity
    #         gt_disparity, gt_non_neg_mask = depth2disparity(
    #             depth=depth_raw, return_mask=True
    #         )
    #         # LS alignment in disparity space
    #         pred_non_neg_mask = depth_pred > 0
    #         valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

    #         disparity_pred, scale, shift = align_depth_least_square(
    #             gt_arr=gt_disparity,
    #             pred_arr=depth_pred,
    #             valid_mask_arr=valid_nonnegative_mask,
    #             return_scale_shift=True,
    #             max_resolution=args.alignment_max_res,
    #         )
    #         # convert to depth
    #         disparity_pred = np.clip(
    #             disparity_pred, a_min=1e-3, a_max=None
    #         )  # avoid 0 disparity
    #         depth_pred = disparity2depth(disparity_pred)

    #     # Clip to dataset min max
    #     depth_pred = np.clip(
    #         depth_pred, a_min=dataset.min_depth, a_max=dataset.max_depth
    #     )
    #     # clip to d > 0 for evaluation
    #     depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

    #     # Evaluate (using CUDA if available)
    #     sample_metric = []
    #     depth_pred_ts = torch.from_numpy(depth_pred).to(device)
        
    #     # save_filled_depth(args, valid_mask, depth_raw, pred_name)
        
    #     # save_prediction_heatmap(args, depth_pred, depth_raw, pred_name)

    #     for met_func in metric_funcs:
    #         _metric_name = met_func.__name__
    #         _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
    #         sample_metric.append(str(_metric))
    #         metric_tracker.update(_metric_name, _metric)

    #     # Save per-sample metric
    #     with open(per_sample_filename, "a+") as f:
    #         f.write(pred_name + ",")
    #         f.write(",".join(sample_metric))
    #         f.write("\n")

    # # -------------------- Save metrics to file --------------------
    # eval_text = f"Evaluation metrics:\n\
    # of predictions: {args.prediction_dir}\n\
    # on dataset: {dataset.disp_name}\n\
    # with samples in: {dataset.filename_ls_path}\n"

    # eval_text += f"min_depth = {dataset.min_depth}\n"
    # eval_text += f"max_depth = {dataset.max_depth}\n"
    # eval_text += tabulate(
    #     [metric_tracker.result().keys(), metric_tracker.result().values()]
    # )

    # metrics_filename = "eval_metrics"
    # if args.alignment:
    #     metrics_filename += f"-{args.alignment}"
    # metrics_filename += ".txt"

    # _save_to = os.path.join(args.output_dir, metrics_filename)
    # with open(_save_to, "w+") as f:
    #     f.write(eval_text)
    #     logging.info(f"Evaluation metrics saved to {_save_to}")
        
    result_df = pd.read_csv(per_sample_filename)
    # Apply grouping function: extract_group
    result_df['group'] = result_df['filename'].apply(extract_group)
    # Get average
    grouped_avg = result_df.groupby('group').mean(numeric_only=True)
    per_group_filename = os.path.join(args.output_dir, "per_group_metrics.csv")
    grouped_avg.to_csv(per_group_filename, index='group')
