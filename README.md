# GP TimeSformer – Vesuvius Challenge Ink Detection Module

## Overview

This submodule is designed for **ink detection in volumetric scroll images**, as part of a broader project on deciphering ancient scrolls (the Vesuvius Challenge). Its main purposes are: 

- **Preprocessing:** Reading multi-layer *fragment* images of scrolls (e.g. CT scan slices) and their corresponding ink labels (masks), then generating training/validation patches.
- **Dataset Handling:** Providing a `ScrollDataset` class that loads image patches and masks, applies augmentations (rotations, temporal perturbations), and prepares data for model consumption [dataset.py].
- **Model Training:** Defining a PyTorch Lightning module `TimeSformerModel` that implements a TimeSformer-based architecture for segmentation. It integrates a vision transformer backbone (TimeSformer) to capture spatio-temporal features from the multi-channel input and outputs a predicted mask patch.
- **Evaluation:** During training, the module evaluates on a validation fragment by reconstructing the full mask from patch-wise predictions and computing losses. It logs metrics and predicted masks for inspection.

In the larger project context, this module handles the heavy-lifting for data preparation and model fine-tuning on the ink detection task, enabling integration with other components (such as ensemble pipelines or post-processing in the overall solution).

## Folder Structure

The repository is organized as follows:

```plaintext
gp_timesformer/               # Main package directory for this submodule
├── dataset.py           # Defines ScrollDataset for loading image & mask patches with augmentations
├── preprocessing.py     # Data loading and preprocessing functions (reads images/masks, creates patches)
├── model.py             # PyTorch Lightning Module (TimeSformerModel) with model architecture and training/val steps
├── scheduler.py         # Learning rate scheduler setup (warmup + cosine annealing) [scheduler.py]
├── utils.py             # Utility functions (random seed setting, directory creation)
├── main.py              # Entry-point script: orchestrates config loading, data prep, model instantiation, training loop
└── config.yaml          # Configuration file for hyperparameters, paths, and settings
setup.py                     # Installation script for the gp_timesformer package [setup.py]
README.md                    # Documentation (this file)
```

**Key components and their interactions:**

- *Preprocessing Pipeline:* `main.py` calls `prepare_datasets` from **preprocessing.py** to read raw data and generate train/validation splits [main.py]. This function uses `read_fragment_images_and_masks` to load the multi-layer images (scroll CT slices) and corresponding masks from the dataset, then tiles them into smaller patches [preprocessing.py]. Only patches containing valid fragment area and (for training) some ink are included. The result is a dictionary of training and validation patch arrays.
- *Dataset Class:* `ScrollDataset` (in **dataset.py**) wraps these patch arrays into a PyTorch `Dataset`. It handles on-the-fly transformations: random rotations and a custom *temporal augmentation* that shuffles subsets of the input channels (simulating misordered layers) On each call, it returns a tensor patch and its mask. Validation samples include coordinates to reconstruct their location in the full image 
- *Model Definition:* `TimeSformerModel` (in **model.py**) is a Lightning module encapsulating the neural network and training logic. It uses a **TimeSformer** backbone (Vision Transformer for temporal data) to process the stack of image layers and predict a downsampled mask. The model computes a **segmentation loss** (Dice loss + soft BCE) ([model.py] and implements training/validation steps with logging.
- *Training Orchestration:* `main.py` ties everything together. It uses **Hydra** to load `config.yaml` and sets up experiment directories and seeds. It creates `DataLoader` instances for train/val using `ScrollDataset`, initializes the `TimeSformerModel`, and configures a PyTorch Lightning `Trainer` with callbacks for model checkpointing and Weights & Biases (WandB) logging.

## Setup and Installation

To set up the environment and install this submodule, follow these steps:

1. **Clone the Repository:** Clone the project repository (ensure you have access to the larger project or this submodule directory).
2. **Create a Python Environment:** It’s recommended to use a Conda environment (or virtualenv) with Python 3.x. For example:  
   ```bash
   conda create -n vesuvius python=3.9 -y
   conda activate vesuvius
   ```  
   This creates and activates an environment named `vesuvius` (matching the project context).
3. **Install Dependencies:** Install the package and required libraries using `setup.py`. In the repository root (where `setup.py` is located), run:  
   ```bash
   pip install -e .
   ```  
   This will install **gp_timesformer** in editable mode along with its core dependencies (Torch, Hydra, WandB, NumPy, Zarr, etc.). Make sure to also install the following required packages if not already present:  
   - **PyTorch** (GPU-capable, with CUDA if available) and **torchvision** (for image transforms).  
   - **PyTorch Lightning** (for the high-level training loop).  
   - **Hydra** (`hydra-core`) and **OmegaConf** (for configuration management).  
   - **OpenCV** (`opencv-python`) for image loading.  
   - **Segmentation Models PyTorch** (`segmentation_models_pytorch`) for loss functions (Dice, BCE).  
   - **TimeSformer-pytorch** (a library implementing the TimeSformer model).  
   - **Warmup Scheduler** (`warmup_scheduler`) for learning rate scheduling.  
   - **wandb** (Weights & Biases) for logging.  
   These may be installed via pip if not covered by the `setup.py` requirements.
4. **Configure Weights & Biases (Optional):** If you plan to use experiment tracking, log in to WandB in your environment by running `wandb login` and following prompts. (If not, the code will still run, but you can disable WandB logging by adjusting the code or config.)
5. **Verify Installation:** After installation, you should be able to import the package in Python. For example: `python -c "import gp_timesformer; print('OK')"`. This should output "OK" without errors.

## Usage Instructions

This submodule provides an end-to-end pipeline for training the TimeSformer model on the scroll dataset. Typical usage involves the following steps:

1. **Prepare Data:** Ensure the **Vesuvius Challenge dataset** is downloaded and placed in the directory specified by `comp_dataset_path` in `config.yaml`. The expected structure is:  
   - For each **Fragment** (labeled data): a folder `Fragments/Fragment<id>` containing a `surface_volume` subfolder with multiple layer images (e.g., `00.tif, 01.tif, ...`), an `inklabels.png` (ground truth mask of ink, same size as one layer), and a `mask.png` (fragment mask indicating the valid area of the fragment).  
   - For **Segments** (unlabeled extra data from scrolls): folders `Scroll<scroll_id>/segments/<segment_id>/` with a `layers` subfolder for images and a `mask.png` for fragment area (these have no `inklabels.png`).  
   Make sure to update `config.yaml` with the correct path to this data. By default it uses a relative path (`../../../VesuviusDataDownload/`)
2. **Adjust Configuration (if needed):** Open `config.yaml` (or create a Hydra override file) to set desired parameters. For example, you might want to change the training `epochs`, learning rate, batch size, or specify different fragment IDs for training/validation. Key configuration options are explained in the **Configuration** section below.
3. **Run Preprocessing + Training:** The training script will automatically handle preprocessing and training. Simply execute the `main.py` script with Python. For example:  
   ```bash
   cd gp_timesformer      # if not already in the module directory
   python3 main.py
   ```  
   This will launch the Hydra-powered run using `config.yaml`. The script will print out information about the dataset (like number of patches for train/val) and then begin model training. Progress will be shown per epoch with training/validation loss.
4. **Monitoring Training:** If WandB logging is enabled and configured, each run will create an entry in your W&B project (by default project name is "vesuvius"). You can monitor metrics like **training loss** (`train/total_loss`) and **validation loss** (`val/total_loss`) in real time, and view the predicted mask images logged at each validation epoch.
5. **Evaluation:** During training, validation is performed on the specified validation fragment (`validation_fragment_id`). After training completes (or at any checkpoint), you can evaluate the results. The validation predictions for each patch are combined into a full-size mask and logged to WandB as an image called "masks" each epoch [model.py]. If you want to quantitatively evaluate, you could compare these predictions to the ground truth mask (e.g., compute Dice score) – this would require additional code, as the current module mainly focuses on training. For the Kaggle challenge, generating a submission would involve running the model on test fragments; you can adapt the provided code by loading a trained checkpoint and processing test data similarly to validation patches.
6. **Using Checkpoints:** The training script saves model checkpoints for each epoch in the directory specified by `cfg.model_dir` (default `./models`). You can find files like `timesformer_<fragmentID>_epoch{N}.ckpt` there. To resume training or run inference, load the appropriate `.ckpt` with PyTorch Lightning or the `TimeSformerModel` class.

**Example Command:** After setting up config and data, a one-command execution might look like:  
```bash
python gp_timesformer/main.py epochs=15 train_batch_size=128 wandb.name="run1_timesformer"
```  
(This uses Hydra overrides to train for 15 epochs with batch size 128 and set the W&B run name. You can also edit `config.yaml` directly instead of using CLI overrides.)

## Configuration

The training behavior and data settings are controlled via **config.yaml** (using Hydra/OmegaConf). Below are the key sections and parameters in the config and how to use them:

- **Project and Paths:**  
  - `comp_name`: Name of the competition or project (default `"vesuvius"`).  
  - `exp_name`: Experiment name or description (for logging).  
  - `comp_dataset_path`: Filesystem path to the dataset root. Update this to where your data is located. For example, `/path/to/VesuviusData/` containing the `Fragments/` and `Scroll*/segments/` subfolders.  
- **Runtime Settings:**  
  - `seed`: Random seed for reproducibility.  
  - `device`: `"cuda"` or `"cpu"` – typically `"cuda"` for GPU acceleration. (The code uses PyTorch Lightning’s device handling, so ensure this is set appropriately or let Lightning auto-select GPU.)  
  - `num_workers`: Number of worker processes for data loading. Set this based on CPU cores for optimal throughput (e.g., 4).  
- **Training Hyperparameters:**  
  - `lr`: Initial learning rate for the optimizer (AdamW).  
  - `min_lr`: Minimum learning rate for cosine annealing schedule (the lowest point it will decay to).  
  - `epochs`: Total number of training epochs.  
  - `warmup_epochs`: Number of warmup epochs for gradual LR ramp-up at the start.  
  - `weight_decay`: Weight decay (L2 regularization) factor for AdamW (if used).  
  - `scheduler_t_max`: The period (in epochs) for the cosine LR scheduler – typically set equal to `epochs` for a full decay cycle  
  - `train_batch_size` / `valid_batch_size`: Batch sizes for training and validation DataLoaders. These depend on your GPU memory and patch size; the default (196) works if using smaller patches and a GPU with sufficient memory. You may need to adjust down if you encounter OOM issues.  
- **Data Configuration:**  
  - `in_chans`: Number of input channels (layers) per sample. This should match the number of image layers loaded for each fragment. For instance, if using CT slice indices 22 through 37, that yields 16 layers (as in the default example). Ensure this value corresponds to `end_idx - start_idx` in the dataset parameters (described below).  
  - `size`: Patch size used for model input. This is the height/width of the small patches cut from each tile. Default 64 means each training patch is 64×64 pixels. **Important:** The model is configured to output a 4×4 prediction for a 64×64 input (since it processes 16×16 patches internally). If you change `size`, you may need to adjust model parameters accordingly (the current setup assumes 64).  
  - `tile_size`: Size of the sliding window tile used to scan the fragment image. Default 256 means the code will take 256×256 pixel windows across the large fragment. Within each such tile, it will further subdivide into 64×64 patches (if `size=64`).  
  - `stride`: Step size to move the 256×256 window when tiling. Default 32 means windows overlap heavily (creating many patches). A smaller stride yields more patches and more training data (with overlap), at the cost of redundancy and longer processing. Adjusting this trades off data volume vs. speed. Typically, `stride` might be set to `tile_size // 8` (which 32 is, for 256).  
  - `fragment_ids`: List of fragment indices to process for training/validation. For example `[1, 3]` means use Fragment1 and Fragment3 from the dataset. These correspond to subdirectories under `Fragments/`. All listed fragments will be processed; one of them can be held out as validation.  
  - `validation_fragment_id`: The fragment ID (from the above list) to reserve for validation. In the default config, fragment 3 is used as validation while 1 (and possibly others) are used for training. During preprocessing, patches from this fragment are separated as the validation set.  
  - `flip_ids`: (Optional) list of fragment IDs whose layer order should be reversed (flipped) when loaded. By default empty, but this can be used if certain fragments were scanned in reverse order of layers; flipping ensures consistency in channel ordering.  
  - `scrolls`: A list of *segments* to include for additional data. This is for using extra unlabeled scroll segments (each scroll contains many small segments). Each entry has `scroll_id` and `num_segments` to load from that scroll. For example, scroll 1 with `num_segments: 2` will randomly pick 2 segments from Scroll1’s data. If `num_segments: -1`, it will use all segments. The code will treat these as separate pieces of data (with no ink labels, masks filled with zeros). By default, some segments from scroll1 and scroll2 are listed. You can modify or remove this if you only want to train on fully labeled fragments.  
  - `validation_segments_mod`: Determines how segments are split into train/val. For instance, a value of 10 means every 10th segment is taken as validation. This is only relevant if using segments (unlabeled data) to possibly evaluate generalization; otherwise segments mostly serve as extra training data (with dummy masks).  
  - `use_cache`: If `True`, the preprocessing will save the processed patches to disk (under each fragment’s `cache/` directory using Zarr) and reuse them on subsequent runs. This speeds up repeated experiments on the same data. Keep it `True` to avoid re-processing each run. Set `False` if you suspect the preprocessing code changed or want to force re-read.  
- **Dataset Parameters:**  
  - `start_idx` / `end_idx`: The range of layer indices to read for each fragment or segment. These define the depth of the volume to use. By default `22` to `38` will read layers 22 through 37 (inclusive) which is 16 layers, matching `in_chans: 26` (though note, 22–37 is actually 16; adjust if needed). You can widen this range to use more slices (up to the full available, typically 0–64 in the competition data), but ensure `in_chans` is updated accordingly and your GPU can handle the larger model input.

You can modify the configuration by editing the YAML file or by overriding values via the command line when running `main.py`. For example, to change learning rate and number of epochs without editing the file, you can run:  

```bash
python main.py lr=5e-4 epochs=20
```  
Hydra will compose these overrides with the defaults. All configuration values in `config.yaml` can be customized this way.

## Model Details

**TimeSformerModel:** The core model is a **TimeSformer** (Time-Space Transformer) adapted for segmentation. TimeSformer is a transformer-based neural network originally designed for video classification, treating the temporal dimension with self-attention. In this project, we repurpose it to handle a stack of image layers (the “video” frames) and predict an ink mask. Key aspects of the model:

- **Architecture:** The model uses a TimeSformer backbone from the `timesformer_pytorch` library. It is configured with:
  - `num_frames = cfg.in_chans` – the number of input frames (or channels) matches the number of image layers (e.g., 16 or 26). Each layer of the fragment is treated like a frame in a sequence.
  - `image_size = 64` and `patch_size = 16` – the TimeSformer divides each 64×64 patch into 16×16 patches (resulting in a 4×4 grid of patches). This yields a sequence of 4*4*frames tokens for the transformer.
  - `dim = 512` – token embedding dimension.  
  - `depth = 8` – number of transformer layers (self-attention blocks).  
  - `heads = 6` – number of attention heads.  
  - `dim_head = 64` – dimension per head.  
  - `attn_dropout = 0.1` and `ff_dropout = 0.1` – dropout rates in attention and feedforward layers for regularization.  
  - `channels = 1` – since input images are single-channel (grayscale) per layer.  
  - `num_classes = 16` – this defines the output dimension of the model. Here it is set to 16, which corresponds to a flattened 4×4 output mask for each input patch. The model's forward output is later reshaped to (1,4,4) for each sample.
  These settings are tailored so that the TimeSformer processes the input patch and outputs a 4×4 prediction map. Each element of this 4×4 corresponds to a region of the input (effectively a 16×16 block, given the patch partitioning).

- **Preprocessing for Model:** Before feeding to TimeSformer, the input patch tensor is prepared in `TimeSformerModel.forward()`. If the data comes in as 4D (B, H, W, C), it’s reshaped to 5D (B, C, D, H, W) where D is number of frames. Then, if `with_norm` is enabled, a 3D batch normalization is applied across the input volume. The tensor is then permuted to shape (B, D, C, H, W) as expected by the TimeSformer library (frames first). The backbone’s output is a flat vector of length 16 per sample, which the model reshapes to a 1×4×4 mask patch.

- **Loss Functions:** The training uses a **hybrid loss** to handle the segmentation task:
  - **Dice Loss:** Measures overlap between predicted mask and true mask, focusing on region similarity (good for class imbalance).
  - **Soft BCE with Logits Loss:** A smoothed binary cross-entropy that provides pixel-wise classification loss. A smoothing factor (0.25) is used to avoid over-confidence. 
  The model combines these two losses equally: `total_loss = 0.5 * dice_loss + 0.5 * bce_loss`. This helps capture both regional accuracy and per-pixel accuracy. The losses are provided by `segmentation_models_pytorch` library for convenience.

- **Optimizer:** The optimizer is **AdamW** (Adam with weight decay) applied to all model parameters. The learning rate is taken from `cfg.lr` (e.g., 1e-3) and weight decay from `cfg.weight_decay`. AdamW is chosen to improve generalization with decoupled weight decay.

- **Learning Rate Scheduler:** A custom scheduler is set up to adjust the learning rate during training. It consists of:
  - A **CosineAnnealingLR** scheduler (from PyTorch) that will gradually decrease the LR from the initial value down to `cfg.min_lr` over `cfg.scheduler_t_max` epochs in a cosine curve 
  - A **Gradual Warmup** wrapper (`GradualWarmupSchedulerV2`) that linearly increases the LR from 0 (or a small base) up to the initial `lr` over the first `cfg.warmup_epochs` epochs. After the warmup period, it hands off to the cosine scheduler.  
  This means at the start of training the LR ramps up slowly (to stabilize training), then follows a cosine decay. Both `warmup_epochs` and `scheduler_t_max` can be tuned in config.

- **Training Loop (Lightning):** The `TimeSformerModel` class implements `training_step` and `validation_step` for the Lightning trainer. In each training step, it receives a batch of patches and masks `(x, y)`, performs a forward pass, computes the loss, and logs it. The validation step similarly computes loss on validation patches, but also does additional work to reconstruct the full-size mask:
  - The validation batch includes `(x, y, xyxys)` where `xyxys` are the coordinates of that patch in the original full image.
  - After getting model outputs for each patch in the batch, it applies a sigmoid to get predicted probabilities, then **upsamples** each 1×4×4 prediction back to 64×64 (the original patch size) using bilinear interpolation.
  - It accumulates these 64×64 predictions into a `mask_pred` array at the correct location (using the coords) and similarly accumulates a `mask_count` (to track overlaps). This is done for every validation batch.
  - At `on_validation_epoch_end`, the model divides `mask_pred / mask_count` to obtain the averaged prediction mask for the entire validation fragment. This combined mask is then logged to WandB as an image called `"masks"`. The arrays `mask_pred` and `mask_count` are then reset for the next epoch.
  - This mechanism ensures that if multiple predictions cover the same pixel (due to overlapping patches), the contributions are averaged for a smoother final prediction.

- **W&B Logging:** In addition to losses, the LightningModule uses `self.log` to record `train/total_loss` and `val/total_loss` for each epoch (and step-wise for training) which the W&B logger will capture. The `WandbLogger` is set up in `main.py` to watch the model (logging gradients and parameters). The custom mask image logging at epoch end (described above) uses the WandB API directly to log the `"masks"` image

- **Mixed Precision and Gradient Clipping:** The training is configured to use mixed precision (16-bit floating point) for efficiency, and gradient norms are clipped at 1.0. These settings are passed via the Lightning Trainer in `main.py`. Mixed precision can speed up training on modern GPUs and reduce memory usage, while gradient clipping helps stabilize training given the large gradients that can occur with transformers.

Overall, the model is an advanced architecture leveraging transformer-based spatial-temporal encoding to detect ink patterns in the 3D data. Despite the complex model, the training framework (Lightning) abstracts much of the boilerplate, allowing focus on experimentation with parameters and data.

## Checkpoints and Logging

**Model Checkpoints:** The training process automatically saves model checkpoints through PyTorch Lightning’s ModelCheckpoint callback. In `main.py`, a checkpoint callback is created to save every epoch’s model state. Key aspects of checkpointing:

- Checkpoints are saved in the directory specified by `cfg.model_dir` (default is `./models` in the project root). Ensure this directory exists or is created (the code calls `create_directories([cfg.model_dir])` to make it if missing).
- The filename format is `timesformer_{validation_fragment_id}_epoch{epoch}` with the epoch number appended. For example, if `validation_fragment_id` is 3, you will see files like `timesformer_3_epoch0.ckpt`, `timesformer_3_epoch1.ckpt`, etc.
- The checkpoint monitors the metric **train/total_loss** and saves all epochs (up to `save_top_k=cfg.epochs`). In this configuration, it does not early-stop; it simply keeps each epoch’s checkpoint (since the training is only for a fixed small number of epochs typically).
- Each checkpoint contains the model weights and optimizer states. You can load a `.ckpt` file with Lightning’s `LightningModule.load_from_checkpoint()` or manually load the `state_dict` into `TimeSformerModel`. This is useful for resuming training or doing inference later.

If you want to change checkpoint behavior (e.g., save only the best model according to validation loss, or save less frequently), you can modify the ModelCheckpoint parameters in `main.py`. For instance, monitoring `val/total_loss` and setting `mode="min"` would save the best validation-loss model.

**Weights & Biases Logging:** Training is instrumented with W&B for experiment tracking:

- A W&B logger is initialized at run start. By default it uses project name `"vesuvius"` and a run name that includes the fragment and patch size (constructed as `run_slug` in the code for uniqueness). You can change these in config or override via command line (for example, adding `wandb.project=<name>` or `wandb.name=<run_name>` if you extend the Hydra config).
- Metrics: The training and validation losses are automatically logged each epoch under the keys `train/total_loss` and `val/total_loss`. They will appear as charts in the W&B run dashboard. The code logs them with `on_epoch=True` and `prog_bar=True` so you see them in the console as well.
- Model gradients/parameters: `wandb_logger.watch(model, log="all", log_freq=100)` is used. This will log histograms of model weights and gradients every 100 training steps, which can be inspected in W&B to diagnose training (this can be memory intensive, so the frequency can be adjusted or turned off if not needed).
- **Predicted Mask Visualization:** One unique logging feature is that at the end of each validation epoch, the full predicted mask for the validation fragment is logged as an image to W&B. In the W&B run, you’ll find an image panel showing the model’s ink prediction across the entire validation fragment. Over epochs, you can compare these to see how the prediction improves. This is very helpful for qualitative evaluation.
- To disable W&B logging (if you prefer not to use it), you can set the `WandbLogger` to `offline` mode or remove the logger from the Trainer (e.g., comment out the WandbLogger lines and `logger=wandb_logger` in Trainer). The training will still run and print losses to console, and checkpoints will still be saved.

**Console Outputs:** Apart from W&B, the script prints some information to the terminal:
- When preparing datasets, it prints the count of train/val patches for each fragment/segment processed. For example, it might print `fragments - train_images: 5000` indicating 5000 training patches were extracted.
- During training, Lightning will by default show a progress bar for each epoch with the current loss values. At the end of each epoch, you’ll see logged metrics (train loss and val loss).
- The `print(f'x.shape: {x.shape}, y.shape: {y.shape}')` in validation step might output the shape of a batch’s tensors occasionally, which is mostly for debugging (e.g., confirming that x is [B, C, D, H, W] and y is [B,1,4,4] as expected).

After training completes, you should have: saved model checkpoints in the models folder, logged metrics (in W&B or console), and potentially a W&B run page with detailed logs and the final predicted mask for the validation fragment.

## Expected Outputs

After successfully running the preprocessing and training pipeline, you can expect the following outputs and results:

- **Model Checkpoints:** As described above, the `models/` directory will contain saved checkpoints for each epoch. The final checkpoint (e.g., `timesformer_3_epoch9.ckpt` for 10 epochs training) contains the trained model weights. You can use this for inference on new data or further fine-tuning.

- **Training Logs:** If using W&B, the metrics and images are logged there. You should see graphs of training loss decreasing over epochs and validation loss as well. The final training loss might be significantly lower than the starting loss, indicating the model has learned features to detect ink. Validation loss gives an indication of performance on the held-out fragment. In the console, you will see the loss values per epoch and possibly per batch (depending on Lightning’s logging frequency).

- **Reconstructed Validation Mask:** For the validation fragment, the model’s predictions are combined and you can visualize the ink detection result. Ideally, regions with ink (writing) will have higher predicted values. In W&B, under the run’s “media” or images section, there will be an image logged for `"masks"` each epoch. The final one corresponds to the model’s full prediction on the validation fragment. You can compare it with the ground truth (`inklabels.png` of that fragment) to qualitatively assess performance. 

- **Intermediate Data (if caching enabled):** If `use_cache=True`, the first run will produce cache files (Zarr format) inside each fragment’s directory (`cache/data.zarr`). These files store the extracted patches and can be reused on subsequent runs, saving preprocessing time. This is not directly a user-facing output, but you can find them in the dataset directories.

- **Potential Predictions on Test Data:** This module as-is does not automatically output test predictions (since Kaggle submissions require running on secret test fragments). However, you can adapt it to run on any fragment by treating it as a “validation” fragment in the config. For example, if you have a Fragment2 with no ground truth (as test), you could set `fragment_ids: [1,2]` and `validation_fragment_id: 2` to train on 1 and output predictions for 2. The predicted mask would be logged (and you could save it from W&B or modify code to save locally). This approach would yield the expected output mask for the test fragment which could be thresholded and submitted. This is an advanced usage, but it’s how you could get the model’s predictions for new data.

**License:** This project is released under the MIT License. You are free to use, modify, and distribute this code in your own projects, provided you include the proper attribution. Please see the `LICENSE` file for details.

