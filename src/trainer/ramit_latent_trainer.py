import logging
import numpy as np
import os
import shutil
import torch
import torch.nn as nn

from datetime import datetime
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Union

from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence


class RAMiTLatentTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: nn.Module,
        train_dataloader: DataLoader,
        device,
        out_dir_ckpt,
        accumulation_steps: int,
    ):
        self.cfg: OmegaConf = cfg
        self.model: nn.Module = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.train_loader: DataLoader = train_dataloader
        self.accumulation_steps: int = accumulation_steps
        
        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Building Loss
        self.smooth_l1_loss = nn.SmoothL1Loss()
        # self.charbonnier_loss = CharbonnierLoss()
        # self.ssim_loss = SSIMLoss()
        self.train_metrics = MetricTracker(*["loss"])
    
        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period

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
                target = batch["target_latent"].to(device)
                
                model_pred = self.model(rgb_latent)  # [B, 4, h, w]
                
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
        # temp_ckpt_dir = None
        # if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
        #     temp_ckpt_dir = os.path.join(
        #         os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
        #     )
        #     if os.path.exists(temp_ckpt_dir):
        #         shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
        #     os.rename(ckpt_dir, temp_ckpt_dir)
        #     logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save Model
        # model_path = os.path.join(ckpt_dir)
        
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "ramit.pth"))
        logging.info(f"Model is saved to: {ckpt_dir}")

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

        # # Remove temp ckpt
        # if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
        #     shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
        #     logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.safetensors")
        
        # Check if safetensors file exists, otherwise try .bin file
        if not os.path.exists(_model_path):
            _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        
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
        
        self.model.unet.load_state_dict(state_dict)
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

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
