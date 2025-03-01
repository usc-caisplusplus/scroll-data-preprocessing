# Scroll Data Preprocessing

This repository contains data preprocessing scripts for handling scroll dataset images and segmentation masks. The pipeline includes dataset preparation, normalization, augmentation, and model training using **PyTorch Lightning** and **TimeSformer** for temporal segmentation.

## 📂 Repository Structure

```
├── data_preprocessing/
│   ├── dataset.py          # Custom PyTorch Dataset class for scroll images
│   ├── main.py             # Main script for training the segmentation model
│   ├── model.py            # Implementation of TimeSformer segmentation model
│   ├── preprocessing.py    # Prepares datasets from fragments and segments
│   ├── scheduler.py        # Learning rate scheduler with warm-up
│   ├── utils.py            # Utility functions for seed setting and directory creation
│   |── config/
│       ├── config.yaml         # Configuration file for Hydra
│── setup.py                # Installation script for pip
│── vesuvius.yaml           # Conda environment file
│── README.md               # Project documentation
```

## 🔧 Installation

### Using Conda (Recommended)
To create a Conda environment with all dependencies:

```bash
conda env create -f vesuvius.yaml
conda activate vesuvius
```
Then run the following to use the setup.py file.

```bash
pip install -e .
```

## 🚀 Usage

### 1️⃣ Dataset Preparation
Before training, preprocess the dataset using:

```bash
python data_preprocessing/preprocessing.py
```

This script processes image fragments and segmentation masks, splitting them into **train, validation, and test** sets.

#### Customizing Preprocessing Parameters
You can modify dataset parameters in config/config.yaml before running the script. Some key parameters you might want to tweak:

```yaml
# Fragments to process
fragment_ids: [1, 2, 3]  # List of fragment numbers to include

# Specify a single fragment for validation
validation_fragment_id: 1  # Set to "-1" to disable validation

# Segments to process
scrolls:
  - scroll_id: 2
    num_segments: -1  # "-1" means process all segments from this scroll
  # - scroll_id: 3
  #   num_segments: 10  # Reads 10 random segments from this scroll

# Validation segment frequency
val_seg_mod: 5  # Every 5th segment is used for validation (set "-1" to disable)

```

- `fragment_ids`: List of fragment numbers to process.
- `validation_fragment_id`: Select a single fragment for validation (-1 disables validation).
- `scrolls`: Specifies which scrolls to read and how many random segments to include.
- `num_segments`:
    - **-1** → Read all segments from this scroll
    - **n** → Select n random segments from this scroll (Replace **n** with an actual value)
- `val_seg_mod`: Controls how often a segment is assigned to validation.


🚨 Important: During data collection, no explicit validation data was specified so by default, everything is saved as training data. If needed, adjust `validation_fragment_id` and `val_seg_mod` to designate validation samples.


### 2️⃣ Training the Model
To train the segmentation model, run:

```bash
python data_preprocessing/main.py
```

This will:
- Load the dataset using `ScrollDataset`
- Train the `TimeSformerModel` with **PyTorch Lightning**
- Log training metrics and model checkpoints using **Weights & Biases (wandb)**

### 3️⃣ Model Architecture
The model is based on **TimeSformer**, a **video transformer** adapted for **scroll image segmentation**. The model structure is implemented in `data_preprocessing/model.py`.

## 🏗️ Code Components

### `data_preprocessing/dataset.py`
Defines `ScrollDataset`, a custom PyTorch Dataset class that:
- Loads image and mask patches
- Applies **temporal augmentations**
- Returns training/validation data

### `data_preprocessing/model.py`
Implements `TimeSformerModel`, a **spatiotemporal transformer** that:
- Uses `TimeSformer` for feature extraction
- Optimizes with **Dice Loss + BCE Loss**
- Implements **gradient clipping** for stability

### `data_preprocessing/preprocessing.py`
Handles dataset preparation:
- Loads images/masks from fragments & segments
- Applies **tiling** and **augmentation**
- Saves preprocessed data as NumPy arrays

### `data_preprocessing/main.py`
Main script that:
- Loads configuration via Hydra
- Initializes dataset and dataloaders
- Runs the training loop using **PyTorch Lightning**
- Logs to **Weights & Biases (wandb)**

### `data_preprocessing/utils.py`
Helper functions for:
- Setting **random seed** for reproducibility
- Creating **directories** dynamically

### `data_preprocessing/scheduler.py`
Implements a **warm-up learning rate scheduler** to stabilize training.

## 📝 Configuration
The pipeline is configured using Hydra (`config/config.yaml`). Modify hyperparameters such as:

```yaml
train_batch_size: 8
valid_batch_size: 4
epochs: 50
lr: 0.0001
tile_size: 256
stride: 128
```

## 📊 Logging & Visualization
Training logs are stored in **Weights & Biases (wandb)**. Ensure `wandb` is properly configured:

```bash
wandb login
```

To visualize results:

```bash
wandb online
```

## 📜 License
This project is licensed under the **MIT License**.

## 👥 Contributors
- **Aryan Gulati**
- **Aditya Kumar**
- **Jaiv Doshi**
- **Leslie Moreno**
