# Latent Restoration Diffusion Model for Robust Monocular Depth Estimation under Adverse Weather

<!-- This project implements Marigold, a Computer Vision method for estimating image characteristics. Initially proposed for
extracting high-resolution depth maps in our CVPR 2024 paper **"Repurposing Diffusion-Based Image Generators for Monocular 
Depth Estimation"**, we extended the method to other modalities as described in our follow-up paper **"Marigold: Affordable 
Adaptation of Diffusion-Based Image Generators for Image Analysis"**.  -->

Team:
[Dingjie PENG](),
[Junpei XUE](),


## 🛠️ Setup

The inference code was tested on:

- Ubuntu 24.04 LTS, Python 3.10.12,  CUDA 11.7, GeForce RTX A6000 (pip)

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/jungletada/Marigold-Weather.git
cd Marigold-Weather
```


### Dataset
WeatherKITTI dataset
Please put the dataset folder as `data/kitti`.
<pre>
data/kitti
├── align_depth
│   ├── 2011_09_26_drive_0086_sync
│   ├── 2011_09_26_drive_0087_sync
│   ├── ...
├── depth
├── fog
├── latent_train
├── mix_rain
├── mix_snow
├── raingan
├── rgb
└── snowgan
 </pre>


### 💻 Dependencies

Install the dependencies:

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
```

```bash
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

<!-- ### ⚙️ Inference settings

The default settings are optimized for the best results. However, the behavior of the code can be customized:

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.

- `--ensemble_size`: Number of inference passes in the ensemble. Larger values tend to give better results in evaluations at the cost of slower inference; for most cases 1 is enough. Default: 1.

- `--denoise_steps`: Number of denoising diffusion steps. Default settings are defined in the model checkpoints and are sufficient for most cases.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  
  
  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: `None`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.


<!-- ### 🎮 Run inference (for academic comparisons)
These settings correspond to our paper. For academic comparison, please run with the settings below (if you only want to do fast inference on your own images, you can set `--ensemble_size 1`).
 -->

You can find all results in the `output` directory. Enjoy!


## 🦿 Evaluation on test datasets <a name="evaluation"></a>
Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```
Then, run the inference and evaluation code, you need to specify the folder for `finetune_ckpt` and `output_child_dir`:
```bash
bash script/depth/infer_eval_kitti.sh \
  output/train_weather_finetune_${version}/checkpoint/latest \
  ${version}
```
Note: although the seed has been set, the results might still be slightly different on different hardware.
 
## 🏋️ Training
Based on the previously created environment, install extended requirements:
```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```
Set environment parameters for the data directory:
```bash
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```
<!-- Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}` -->

<!-- ### Prepare for training data
**Depth**
Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/depth/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing. -->


### Run training script for finetuning. 
${version}=`full`, `encoder_only` or `adapter_only`  

${batchsize}=`4` (default)  

For example, training with full UNet with a batch size of 4:
```bash


```
Resume from a checkpoint, e.g.:

```bash
python script/depth/train.py --resume_run output/train_weather_depth_${version}/checkpoint/latest
```

------

<!-- ### Run training script for latent adapter
```bash
python script/depth/train_latent.py --config config/train_latent_adapter.yaml
``` -->


<!-- ### Compose checkpoint:
Only the U-Net and scheduler config are updated during training. They are saved in the training directory. To use the inference pipeline with your training result:
- replace `unet` folder in Marigold checkpoints with that in the `checkpoint` output folder.
- replace the `scheduler/scheduler_config.json` file in Marigold checkpoints with `checkpoint/scheduler_config.json` generated during training.
Then refer to [this section](#evaluation) for evaluation.

**Note**: Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption. -->

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🎓 Citation

Please cite our papers:

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{ke2025marigold,
  title={Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis},
  author={Bingxin Ke and Kevin Qu and Tianfu Wang and Nando Metzger and Shengyu Huang and Bo Li and Anton Obukhov and Konrad Schindler},
  year={2025},
  eprint={2505.09358},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
