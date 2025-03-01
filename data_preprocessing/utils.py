import os
import random
from typing import List

import numpy as np
import torch

def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(directories: List[str]):
    """
    Create directories if they do not exist.

    Args:
        directories (List[str]): List of directory paths to create.
    """
    for dir_ in directories:
        os.makedirs(dir_, exist_ok=True)
