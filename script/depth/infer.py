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
from tqdm.auto import tqdm
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
from marigold.ramit_model.ramit import RAMiT
from src.dataset import (
    BaseDepthDataset,
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
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time
        seed = int(time.time())

    seed_all(seed)

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

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, 
        base_data_dir=base_data_dir, 
        mode=DatasetMode.EVAL,
        join_split=False,
    )
    assert isinstance(dataset, BaseDepthDataset)
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    unet = UNet2DConditionModel.from_config(args.base_checkpoint, subfolder="unet")
    unet.conv_in = RAMiT()
    
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
    
    pipeline = MarigoldDepthPipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        # variant=variant, 
        # torch_dtype=dtype,
    )

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("Proceeding without xformers")

    pipeline = pipeline.to(device)
    logging.info(
        f"Loaded depth pipeline: scale_invariant={pipeline.scale_invariant}, shift_invariant={pipeline.shift_invariant}"
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Depth Inference on {dataset.disp_name}", leave=True
        ):
            # rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)  # [3, H, W]
            # rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            # input_image = Image.fromarray(rgb_int)
            
            input_image = batch["rgb_int"]
            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
            
            # Perform inference
            pipe_out: MarigoldDepthOutput = pipeline(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=0,
                color_map=None,
                show_progress_bar=False,
                resample_method=resample_method,
                generator=generator,
            )
            depth_pred: np.ndarray = pipe_out.depth_np
            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
            if not os.path.exists(scene_dir):
                os.makedirs(scene_dir)
            # pred_basename = get_pred_name(
            #     rgb_basename, dataset.name_mode, suffix=".npy"
            # )
            pred_basename = rgb_basename.replace(".png", ".npy")
            
            save_to = os.path.join(scene_dir, pred_basename)
            if os.path.exists(save_to):
                logging.warning(f"Existing file: '{save_to}' will be overwritten")
            
            np.save(save_to, depth_pred)
