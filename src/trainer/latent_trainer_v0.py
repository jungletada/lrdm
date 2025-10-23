import os
import shutil
import logging
from tqdm import tqdm
from typing import List, Union, Optional
from PIL import Image
import numpy as np
from datetime import datetime
from omegaconf import OmegaConf

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class LatentTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: nn.Module,
        vae: AutoencoderKL,
        train_dataloader: DataLoader,
        visualize_dataset,
        device,
        out_dir_ckpt,
        accumulation_steps: int,
    ):
        self.cfg: OmegaConf = cfg
        self.model: nn.Module = model
        self.vae = vae
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.save_name = 'ramit_model.safetensors'
        self.out_dir_ckpt = out_dir_ckpt
        self.train_loader: DataLoader = train_dataloader
        self.visualize_dataset: DataLoader = visualize_dataset
        self.accumulation_steps: int = accumulation_steps
        self.latent_scale_factor = 0.18215
        self.vae.requires_grad_(False)
        
        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(
            optimizer=self.optimizer, 
            lr_lambda=lr_func)

        # Building Loss
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.train_metrics = MetricTracker(*["loss"])
    
        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.smooth_l1_loss = nn.SmoothL1Loss()

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def train(self, t_end=None):
        logging.info("Start training")
        device = self.device
        self.model.to(device)
        self.vae.to(device)
        
        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in self.train_loader:
                self.model.train()
                self.vae.eval()
                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>>>>> With gradient accumulation >>>>>>
                img_w = batch['rgb_norm'].to(self.device)
                sunny = batch['sunny_norm'].to(self.device)
                
                with torch.no_grad():
                    lnt_w, logvar_w = self.encode_rgb(img_w)    # use vae encoder to get latent
                    # target, logvar_t = self.encode_rgb(sunny)   # use vae encoder to get latent
                    
                # Forward sunny and weather
                pred_lnt = self.model(img_w, lnt_w)
                pred_img = self.decode_rgb(pred_lnt)
                loss = self.smooth_l1_loss(pred_img, sunny)

                # 2) backward
                self.train_metrics.update("loss", loss.item())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1
                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0
            
                # # 3) SupCon（多正样本；同一 scene_id 视为正样本集合）
                # #   SupCon 参考：Khosla et al., NeurIPS 2020:contentReference[oaicite:3]{index=3}
                # assert scene_ids.dtype == torch.long and scene_ids.dim() == 1 and scene_ids.size(0) == batch_size
                # Z_list = [gap(Tss)] + [gap(Tsw) for Tsw in Tsw_list]   # (m+1) 个 [B, D]
                # Z = torch.cat(Z_list, dim=0)                           # [(m+1)*B, D]
                # G = scene_ids.repeat(num_domains + 1)                            # [(m+1)*B]
                # L_supcon = supcon_loss(Z, G, temperature=0.1)

        #         # 4) 组内一致（方差，鼓励多风格输出坍缩到同一语义）
        #         stacked = torch.stack([gap(Tsw) for Tsw in pred_weathers], dim=0)  # [m, B, D]
        #         L_var = stacked.var(dim=0, unbiased=False).mean()             # 标量

        #         # 5) 多源 CORAL（坏天气→晴天，二阶统计对齐；Deep CORAL:contentReference[oaicite:4]{index=4}）
        #         L_coral = coral_multi_to_sunny(pred_sunny, pred_weathers)

        #         # 6) PatchGAN 多类域对抗（K+1 域；PatchGAN 思想来自 pix2pix:contentReference[oaicite:5]{index=5}）
        #         #    若 D 内部接了 GRL，则生成端的对抗项就是下面的 CE；若未接 GRL，可用 L_adv = -L_D。
        #         logits_s = self.discriminator(pred_sunny)   # 晴域 logits: [B, C_dom, h', w']
        #         logits_w = [self.discriminator(pred_wtr) for pred_wtr in pred_weathers]       # 各坏天气域 logits
                
        #         # 约定 domain 索引：晴=0；每种坏天气依次 1..m
        #         sunny_id = 0
        #         domain_ids = list(range(1, num_domains + 1))
        #         L_discrinative = ce_patch(logits_s, sunny_id) + torch.stack([
        #             ce_patch(lw, domain_ids[j]) for j, lw in enumerate(logits_w)
        #         ]).mean()

        #         use_grl = True  # 如果 D 或连接到 D 的桥里实现了 GRL，就置 True
        #         L_adv = L_discrinative if use_grl else (-L_discrinative)

        #         # 7) 总损失
        #         L = lam1 * L_idendity + lam2 * L_reconstruct + lam3 * L_supcon + lam4 * L_var + lam5 * L_coral + lam6 * L_adv

        # # 你也可以返回一个 dict 便于 log
        # logs = {
        #     "L": L, "L_id": L_idendity, "L_rec": L_reconstruct, "L_supcon": L_supcon,
        #     "L_var": L_var, "L_coral": L_coral, "L_adv": L_adv, "L_D": L_discrinative
        # }
        # return L, logs

    @torch.no_grad
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor    # scale latent

        return rgb_latent, logvar
    
    def decode_rgb(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into rgb.
        Args:
            latent (`torch.Tensor`):
               latent to be decoded.
        Returns:
            `torch.Tensor`: Decoded rgb.
        """
        latent = latent / self.latent_scale_factor
        z = self.vae.post_quant_conv(latent)
        rgb = self.vae.decoder(z) # decode
        return rgb
    
    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
       
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")
        
        # Save safetensors
        adapter_path = os.path.join(ckpt_dir, self.save_name)
        self.model.save_pretrained(adapter_path, safe_serialization=True)
        logging.info(f"Model is saved to: {adapter_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        _model_path = os.path.join(ckpt_path, self.save_name)
        
        if not os.path.exists(_model_path):
            raise FileNotFoundError(f"Model file not found at {_model_path}")
        
        # Load model weights based on file type
        if _model_path.endswith('.safetensors'):
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors library is required to load .safetensors files")
            
            # Load using safetensors
            state_dict = {}
            with safe_open(_model_path, framework="pt", device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            # Load using torch.load for .bin files
            state_dict = torch.load(_model_path, map_location='cpu')
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        logging.info(f"Adapter parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        
        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

    def train_base(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Load data with "input_latent" and "target_latent"
                rgb_latent = batch["input_latent"].to(device)
                rgb_image = batch["rgb_norm"].to(device)
                target = batch["target_latent"].to(device)
                
                model_pred = self.model(rgb_image, rgb_latent)  # [B, 4, h, w]
                
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")
                    exit(0)

                loss = self.smooth_l1_loss(model_pred, target)
                # charbonnier_loss = self.charbonnier_loss(model_pred, target)
                # ssim_loss = self.ssim_loss(model_pred, target)
                # loss = 2 * smooth_l1_loss + charbonnier_loss + ssim_loss
                
                # self.train_metrics.update("smooth_l1_loss", smooth_l1_loss.item())
                # self.train_metrics.update("charbonnier_loss", charbonnier_loss.item())
                # self.train_metrics.update("ssim_loss", ssim_loss.item())
                self.train_metrics.update("loss", loss.item())
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                accumulated_step += 1
                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dict(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0