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
import os
import sys
import argparse
import logging
from typing import Optional
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from marigold.ramit_model.ramit import RAMiTCond
from marigold import MarigoldDepthPipeline
from marigold import LDRMDepthPipeline

from src.dataset import (
    WeatherKITTILatentGroupedDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from src.util.seeding import seed_all
from safetensors.torch import load_file as safe_load_file


def get_args():
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Marigold : Monocular Depth Estimation : Dataset Inference"
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="prs-eth/marigold-depth-v1-1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--finetune_checkpoint",
        type=str,
        default="output/train_weather_depth/checkpoint/latest/unet/diffusion_pytorch_model.safetensors",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="restore",
        help="[concat|original|restore]",
    )
    parser.add_argument(
        "--ramit_checkpoint",
        type=str,
        default="output/train_rasmit_latent/checkpoint/iter_080000/ramit.pth",
        help="Checkpoint path or hub name.",
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
        "--output_dir", 
        type=str, 
        required=True, 
        help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        required=True,
        help="Diffusion denoising steps.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        required=True,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        required=True,
        help="Number of predictions to be ensembled.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
    )
    return parser.parse_args()


def get_pipeline(args):
    """
        Get the pipeline for specific models
    """
    if args.version == 'concat':
        unet = UNet2DConditionModel.from_config(args.base_checkpoint, subfolder="unet")
        unet.conv_in = RAMiTCond()
        
        # Load the trained checkpoint weights
        if args.finetune_checkpoint.endswith(".safetensors"):
            state_dict = safe_load_file(args.finetune_checkpoint)
        else:
            state_dict = torch.load(args.finetune_checkpoint, map_location='cpu')
            
        unet.load_state_dict(state_dict)
        
        vae = AutoencoderKL.from_pretrained(args.base_checkpoint, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(args.base_checkpoint, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(args.base_checkpoint, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_checkpoint, subfolder="tokenizer")
        
        pipeline = LDRMDepthPipeline(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            variant=args.variant, 
            torch_dtype=args.dtype,
        )
        
    elif args.version == 'restore':
        vae = AutoencoderKL.from_pretrained(args.base_checkpoint, subfolder="vae")
        scheduler = DDIMScheduler.from_pretrained(args.base_checkpoint, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(args.base_checkpoint, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(args.base_checkpoint, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(args.finetune_checkpoint, subfolder="unet")
        adapter = RAMiTCond.from_pretrained(args.finetune_checkpoint, subfolder="adapter")
        pipeline = LDRMDepthPipeline(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            adapter=adapter,
        )
    
    elif args.version == 'original':
        # Use Original MarigoldDepthPipeline
        pipeline: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
            args.base_checkpoint, 
            variant=args.variant, 
            torch_dtype=args.dtype
        )
        
    else:
        raise NotImplementedError
    
    return pipeline
      

def _psnr_per_channel(x: torch.Tensor, y: torch.Tensor, data_range: Optional[float] = None, eps: float = 1e-12):
    """
    x, y: shape [N, C, H, W] (本例 N=1, C=4)
    返回: (psnr_per_channel[C], psnr_mean[1])
    """
    assert x.shape == y.shape and x.dim() == 4, f"Expect [N,C,H,W], got {x.shape} vs {y.shape}"
    # 自动估计动态范围（可改成固定 1.0 或 255.0）
    if data_range is None:
        x_min = torch.amin(x)
        y_min = torch.amin(y)
        x_max = torch.amax(x)
        y_max = torch.amax(y)
        data_range = (torch.maximum(x_max, y_max) - torch.minimum(x_min, y_min)).item()
        if data_range == 0:
            data_range = 1.0  # 退化情况避免除零

    # 每通道 MSE：对 N, H, W 取均值，保留 C
    mse = torch.mean((x - y) ** 2, dim=(0, 2, 3))  # shape [C]
    psnr_c = 10.0 * torch.log10((data_range ** 2) / (mse + eps))  # shape [C]
    return psnr_c, psnr_c.mean()

    
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed

    logging.debug(f"Arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{args.base_checkpoint}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {args.seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if args.seed is None:
        import time
        args.seed = int(time.time())

    seed_all(args.seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(
                    f"The directory '{directory}' already exists. Are you sure to continue? (y/n): "
                )
                .strip()
                .lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    # check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset = WeatherKITTILatentGroupedDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        args.dtype = torch.float16
        args.variant = "fp16"
        logging.warning(
            f"Running with half precision ({args.dtype}), might lead to suboptimal result."
        )
    else:
        args.dtype = torch.float32
        args.variant = None
    
    pipeline = get_pipeline(args)
    
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("Proceeding without xformers")

    pipeline = pipeline.to(device)
    
    # Move RAMiT module to the same device if it exists
    if hasattr(pipeline, 'ramit_module') and pipeline.ramit_module is not None:
        pipeline.ramit_module = pipeline.ramit_module.to(device)
    
    logging.info(
        f"Loaded depth pipeline: scale_invariant={pipeline.scale_invariant}, shift_invariant={pipeline.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    domains = ['raingan', 'snowgan', 'fog1', 'rain', 'snow', 'fog2']
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Depth Inference on {dataset.disp_name}", leave=True):
            sunny_image = batch["sunny_image"].to(device)
            rgb_filename = batch["rgb_relative_path"][0]

            sunny_latent, res_latent_sunny = pipeline.reconstruct_rgb(sunny_image)

            # 1) sunny_latent vs res_latent_sunny
            psnr_c, psnr_avg = _psnr_per_channel(sunny_latent, res_latent_sunny)
            print(f"[1] sunny vs res_sunny: PSNR per-channel = {psnr_c.tolist()}, mean = {psnr_avg.item():.4f} dB")

            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))

            for i, weather_image in enumerate(batch["weather_image"]):
                weather_image = weather_image.to(device)
                rgb_latent, res_latent_weather = pipeline.reconstruct_rgb(weather_image)
                # 2) sunny_latent vs rgb_latent（列表）
                if isinstance(rgb_latent, (list, tuple)):
                    for i, t in enumerate(rgb_latent):
                        psnr_c, psnr_avg = _psnr_per_channel(sunny_latent, t)
                        print(f"[2] {domains[i]} sunny vs rgb_latent[{i}]: PSNR per-channel = {psnr_c.tolist()}, mean = {psnr_avg.item():.4f} dB")
                else:
                    # 如果 pipeline 返回的不是列表，兼容单张
                    psnr_c, psnr_avg = _psnr_per_channel(sunny_latent, rgb_latent)
                    print(f"[2] {domains[i]} sunny vs rgb_latent: PSNR per-channel = {psnr_c.tolist()}, mean = {psnr_avg.item():.4f} dB")

                # 3) sunny_latent vs res_latent_weather（列表）
                if isinstance(res_latent_weather, (list, tuple)):
                    for i, t in enumerate(res_latent_weather):
                        psnr_c, psnr_avg = _psnr_per_channel(sunny_latent, t)
                        print(f"[3] {domains[i]} sunny vs res_latent_weather[{i}]: PSNR per-channel = {psnr_c.tolist()}, mean = {psnr_avg.item():.4f} dB")
                else:
                    psnr_c, psnr_avg = _psnr_per_channel(sunny_latent, res_latent_weather)
                    print(f"[3] {domains[i]} sunny vs res_latent_weather: PSNR per-channel = {psnr_c.tolist()}, mean = {psnr_avg.item():.4f} dB")
                
            break  
            # if not os.path.exists(scene_dir):
            #     os.makedirs(scene_dir)
            # save_to = os.path.join(scene_dir, rgb_basename)
            # if os.path.exists(save_to):
            #     logging.warning(f"Existing file: '{save_to}' will be overwritten")