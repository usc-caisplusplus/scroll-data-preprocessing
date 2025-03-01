import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omegaconf import DictConfig


class ScrollDataset(Dataset):
    """
    Custom Dataset for scroll images and corresponding masks.

    Args:
        images (List[np.ndarray]): List of image patches.
        masks (List[np.ndarray]): List of mask patches.
        cfg (DictConfig): Configuration object.
        xyxys (Optional[List[Tuple[int, int, int, int]]]): Coordinates for validation patches.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        cfg: DictConfig,
        xyxys: Optional[List[Tuple[int, int, int, int]]] = None,
    ):
        self.images = images
        self.masks = masks
        self.cfg = cfg
        self.xyxys = xyxys
        self.transform = False

    def __len__(self):
        """Return the total number of samples."""
        return len(self.images)
    
    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Applies rotation transformations to the image.
        """
        # Assuming the rotate_transform expects (H, W, C) images
        image = image.transpose(2, 1, 0)  # (C, W, H)
        image = image.transpose(0, 2, 1)  # (C, H, W)
        image = image.transpose(0, 2, 1)  # Back to (C, W, H)
        return image.transpose(2, 1, 0)  # Finally (H, W, C)

    def _apply_temporal_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a custom temporal augmentation to the image.

        Args:
            image (np.ndarray): Input image of shape (H, W, C).

        Returns:
            np.ndarray: Augmented image.
        """
        image_tmp = image.copy() * 0  # Create a zeroed copy
        cropping_num = random.randint(18, 26)
        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = list(range(start_idx, start_idx + cropping_num))

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        paste_indices = list(range(start_paste_idx, start_paste_idx + cropping_num))
        random.shuffle(paste_indices)

        # Optionally apply cutout to some channels
        if random.random() > 0.4:
            cutout_idx = random.randint(0, 2)
            for idx in paste_indices[:cutout_idx]:
                image_tmp[..., idx] = 0

        # Paste the cropped channels into the new location
        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]
        return image_tmp

    def __getitem__(self, idx: int):
        """
        Retrieve an image and its mask, apply transformations, and return tensors.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Depending on whether it's training or validation:
                - For training: (image_tensor, mask_tensor)
                - For validation: (image_tensor, mask_tensor, (x1, y1, x2, y2))
        """
        image = self.images[idx]
        mask = self.masks[idx]
        if self.xyxys is not None:
            # Validation data: apply augmentation and return coordinates.
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"].unsqueeze(0)
                mask = augmented['mask']
                mask = F.interpolate(mask.unsqueeze(0), size=(self.cfg.size // 16, self.cfg.size // 16)).squeeze(0)
            return image, mask, self.xyxys[idx]
        else:
            # Training data: apply rotations and custom augmentation.
            image = self._apply_rotation(image)
            image = self._apply_temporal_augmentation(image)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image'].unsqueeze(0)
                mask = augmented['mask']
                mask = F.interpolate(mask.unsqueeze(0), size=(self.cfg.size // 16, self.cfg.size // 16)).squeeze(0)
            return image, mask