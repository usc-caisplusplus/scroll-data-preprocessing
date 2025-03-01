import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import segmentation_models_pytorch as smp
from timesformer_pytorch import TimeSformer
import wandb

from torch.optim import AdamW
from omegaconf import DictConfig
from typing import Tuple

from data_preprocessing.scheduler import get_scheduler


class TimeSformerModel(pl.LightningModule):
    """
    PyTorch Lightning Module for the TimeSformer segmentation model.

    Args:
        pred_shape (Tuple[int, int]): Shape of the prediction mask (height, width).
        cfg (DictConfig): Configuration object.
        size (int): Size of the input patches.
        with_norm (bool): Whether to apply Batch Normalization to the input.
    """
    def __init__(self, pred_shape: Tuple[int, int], cfg: DictConfig, size: int = 256, with_norm: bool = False):
        super(TimeSformerModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        # Define loss functions: Dice and Soft BCE with Logits
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.dice_loss(x, y) + 0.5 * self.bce_loss(x, y)

        # Initialize TimeSformer backbone
        self.backbone = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=16,
            num_frames=cfg.in_chans,
            num_classes=16,  # Output dimension
            channels=1,      # Input channels
            depth=8,
            heads=6,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1
        )
        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor reshaped to (B, 1, 4, 4).
        """
        if x.ndim == 4:
            x = x[:, None]  # Add channel dimension if missing
        if self.hparams.with_norm:
            x = self.normalization(x)
        # Permute to (B, num_frames, channels, H, W) as expected by TimeSformer
        x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
        return x.view(-1, 1, 4, 4)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch containing (x, y).
            batch_idx: Index of the batch.

        Returns:
            dict: Dictionary containing the training loss.
        """
        x, y = batch
        outputs = self(x)
        loss = self.loss_func(outputs, y)
        if torch.isnan(loss):
            print("Loss is NaN!")
        self.log("train/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch containing (x, y, xyxys).
            batch_idx: Index of the batch.

        Returns:
            dict: Dictionary containing the validation loss.
        """
        x, y, xyxys = batch
        print(f'x.shape: {x.shape}, y.shape: {y.shape}')
        outputs = self(x)
        loss = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).cpu()
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            pred_patch = F.interpolate(
                y_preds[i].unsqueeze(0), scale_factor=16, mode='bilinear'
            ).squeeze(0).squeeze(0).numpy()
            self.mask_pred[y1:y2, x1:x2] += pred_patch
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))
        self.log("val/total_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """
        At the end of the validation epoch, log the accumulated mask predictions.
        """
        # Avoid division by zero when normalizing
        mask = np.divide(
            self.mask_pred,
            self.mask_count,
            out=np.zeros_like(self.mask_pred),
            where=self.mask_count != 0
        )
        # Log the predicted mask to wandb
        wandb.log({"masks": [np.clip(mask, 0, 1)]}, commit=False)
        # Reset mask accumulators
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[_LRScheduler]]: The optimizer and scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams.cfg.lr)
        scheduler = get_scheduler(self.hparams.cfg, optimizer)
        return [optimizer], [scheduler]
