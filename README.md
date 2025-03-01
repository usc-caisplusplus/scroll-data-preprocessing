# Scroll Data Preprocessing

This repository provides a suite of scripts and utilities designed to preprocess data for the [Vesuvius Challenge](https://scrollprize.org/). The preprocessing steps encompass data normalization and preparation tailored for machine learning applications.

## Features

- **Data Normalization**: Standardizes datasets to have a mean of 0 and a standard deviation of 1, ensuring consistency across data inputs
- **Dataset Splitting**: Efficiently divides data into training, testing, and validation subsets, facilitating robust model evaluation

## Getting Started

### Prerequisites

Ensure the following dependencies are installed:

- Python 3.x.
- NumPy
- scikit-learn

Install the required packages using pip:

```bash
pip install numpy scikit-learn
```

### Usage

1. **Data Normalization**:

   - Organize your training data within a directory named `train`, ensuring each file is in NumPy `.npy` format
   - Place the scaler parameters file, `scaler_params.npy`, in the same directory. This file should contain the mean and scale values utilized for normalization
   - Execute the `preprocessing.py` script to normalize the data. The normalized dataset will be saved as `train_data_normalized.npy` within the `train` directory

   ```python
   import os
   import numpy as np
   from sklearn.preprocessing import StandardScaler

   # Load scaler parameters
   scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()
   scaler = StandardScaler()
   scaler.mean_ = scaler_params['mean']
   scaler.scale_ = scaler_params['scale']

   # Define the path to the training data directory
   train_folder = 'train'
   scaled_data_list = []

   # Process each .npy file in the training directory
   for filename in os.listdir(train_folder):
       if filename.endswith('.npy'):
           file_path = os.path.join(train_folder, filename)
           data = np.load(file_path)
           # Normalize the data
           scaled_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
           scaled_data_list.append(scaled_data)

   # Combine all normalized data
   train_data_normalized = np.stack(scaled_data_list, axis=0)

   # Save the normalized dataset
   save_path = os.path.join(train_folder, 'train_data_normalized.npy')
   np.save(save_path, train_data_normalized)
   ```

2. **Dataset Splitting**:

   - The script includes sections for splitting data into training, testing, and validation sets
   - Modify these sections as necessary to align with your dataset's structure

   ```python
   import os
   import numpy as np
   from sklearn.preprocessing import StandardScaler

   # Define the directory containing the dataset
   array_output_directory = "coarsened_array_outputs"

   # Retrieve filenames and extract time step indices
   filenames = os.listdir(array_output_directory)
   indices = [int(filename.split('_t')[1].split('.')[0]) for filename in filenames]

   # Shuffle indices to ensure random distribution
   np.random.shuffle(indices)

   # Determine the sizes for training, testing, and validation sets
   total_samples = len(indices)
   train_size = int(0.8 * total_samples)
   test_size = int(0.1 * total_samples)
   val_size = total_samples - train_size - test_size

   # Assign indices to each subset
   train_indices = np.array(indices[:train_size])
   test_indices = np.array(indices[train_size:train_size + test_size])
   val_indices = np.array(indices[train_size + test_size:])

   # Create directories for each subset
   output_dir_train = "train"
   output_dir_test = "test"
   output_dir_val = "val"
   os.makedirs(output_dir_train, exist_ok=True)
   os.makedirs(output_dir_test, exist_ok=True)
   os.makedirs(output_dir_val, exist_ok=True)

   # Move corresponding files to their respective directories
   for index in train_indices:
       filename = f'wind_speed_coarse_t{index}.npy'
       src_path = os.path.join(array_output_directory, filename)
       dst_path = os.path.join(output_dir_train, filename)
       os.rename(src_path, dst_path)

   for index in test_indices:
       filename = f'wind_speed_coarse_t{index}.npy'
       src_path = os.path.join(array_output_directory, filename)
       dst_path = os.path.join(output_dir_test, filename)
       os.rename(src_path, dst_path)

   for index in val_indices:
       filename = f'wind_speed_coarse_t{index}.npy'
       src_path = os.path.join(array_output_directory, filename)
       dst_path = os.path.join(output_dir_val, filename)
       os.rename(src_path, dst_path)

   # Load and normalize training data
   train_data = np.array([np.load(os.path.join(output_dir_train, filename)) for filename in os.listdir(output_dir_train)])
   scaler = StandardScaler()
   train_data_normalized = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)

   # Save scaler parameters for future use
   scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
   np.save('scaler_params.npy', scaler_params)

   # Apply normalization to testing and validation datasets
   test_data = np.array([np.load(os.path.join(output_dir_test, filename)) for filename in os.listdir(output_dir_test)])
   val_data = np.array([np.load(os.path.join(output_dir_val, filename)) for filename in os.listdir(output_dir_val)])
   test_data_normalized = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
   val_data_normalized = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)

   # Save the normalized datasets
   np.save(os.path.join(output_dir_train, "train_data_normalized.npy"), train_data_normalized)
   np.save(os.path.join(output_dir_test, "test_data_normalized.npy"), test_data_normalized)
   np.save(os.path.join(output_dir_val, "val_data_normalized.npy"), val_data_normalized)
   ```

   This script will organize your data into training, testing, and validation directories, normalize the training data, and apply the same normalization parameters to the testing and validation datasets.

