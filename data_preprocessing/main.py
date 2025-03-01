import os
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_preprocessing.utils import set_random_seed, create_directories
from data_preprocessing.preprocessing import prepare_datasets
from data_preprocessing.dataset import ScrollDataset
from data_preprocessing.model import TimeSformerModel
import wandb

# Increase PIL limit to avoid decompression errors
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

@hydra.main(config_path="./config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run the training pipeline.

    Args:
        cfg (DictConfig): Configuration object containing all settings.
    """
    # Setup
    set_random_seed(cfg.seed)
    create_directories([cfg.model_dir])
    torch.set_float32_matmul_precision('medium')

    # Build run identifier
    run_slug = (
        f"training_scrolls_valid={cfg.validation_fragment_id}_{cfg.size}x{cfg.size}_submissionlabels_timesformer_finetune"
    )

    # Prepare datasets
    fragment_ids = cfg.get('fragment_ids', [])
    results_dict = prepare_datasets(fragment_ids, cfg)

    for scroll_type in ['fragments', 'segments']:
        if results_dict[scroll_type]['valid_xyxys']:
            results_dict[scroll_type]['valid_xyxys'] = np.stack(results_dict[scroll_type]['valid_xyxys'])

    # Print the length of each item in results_dict
    for scroll_type, data in results_dict.items():
        for key, value in data.items():
            if isinstance(value, list):
                print(f"{scroll_type} - {key}: {len(value)}")


    # Create dataset and dataloaders (using fragments)

    train_dataset = ScrollDataset(
        images=results_dict['fragments']['train_images'],
        masks=results_dict['fragments']['train_masks'],
        cfg=cfg,
    )
    valid_dataset = ScrollDataset(
        images=results_dict['fragments']['valid_images'],
        masks=results_dict['fragments']['valid_masks'],
        cfg=cfg,
        xyxys=results_dict['fragments']['valid_xyxys'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get prediction shape
    pred_shape = results_dict['fragments']['valid_images'][0].shape[:2]

    # Initialize Wandb logger and model
    wandb_logger = WandbLogger(project="vesuvius", name=run_slug)
    model = TimeSformerModel(pred_shape=pred_shape, cfg=cfg, size=cfg.size)
    wandb_logger.watch(model, log="all", log_freq=100)

    # Setup PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=-1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy="ddp_find_unused_parameters_true",
        callbacks=[
            ModelCheckpoint(
                filename=f"timesformer_{cfg.validation_fragment_id}_epoch{{epoch}}",
                dirpath=cfg.model_dir,
                monitor="train/total_loss",
                mode="min",
                save_top_k=cfg.epochs,
            ),
        ],
    )

    # Start training
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    wandb.finish()

if __name__ == "__main__":
    main()