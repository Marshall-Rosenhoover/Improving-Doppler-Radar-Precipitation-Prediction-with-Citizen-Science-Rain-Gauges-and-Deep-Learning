###################################################################################################
# Project:  CIROH - Doppler Radar Precipitation Prediction
# Program:  DRPP_01_Dataset.py
# Author:   Marshall Rosenhoover (marshall.rosenhoover@uah.edu) (marshallrosenhoover@gmail.com)
# Created:  2025-03-26
# Modified: 2025-06-11
#
#    This file contains code for the creation of the dataset. 
#
####################################################################################################

from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

def convert_to_labels(data):
  data[data < 0.1] = 0
  data[(data >= 0.1) & (data < 1.0)] = 1
  data[(data >= 1.0) & (data < 2.5)] = 2
  data[data >= 2.5] = 3
  return data

def load_entries(directory, train_portion=True):
  dataset = np.load(directory)

  # ID = In Distribution (Generalization on Observed Spatial Clusters)
  # OD = Out of Distribution (Generalization to Unseen Spatial Clusters)

  if train_portion:
    x_train   = dataset['x_train']
    y_train   = dataset['y_train']    # Currently in Rain Rates
    x_train   = ((np.log1p(np.log1p(x_train))   / np.log1p(np.log1p(150))) * 2) - 1 
    y_train   = convert_to_labels(y_train)
    x_train   = torch.tensor(x_train.astype(np.float32))
    y_train   = torch.tensor(y_train.astype(np.int64))
    return x_train, y_train 

  else:
    x_test_ID = dataset['x_test_ID']
    y_test_ID = dataset['y_test_ID']   # Currently in Rain Rates
    x_test_OD = dataset['x_test_OD']
    y_test_OD = dataset['y_test_OD']

    # Convert to Rain Labels
    y_test_ID = convert_to_labels(y_test_ID)
    y_test_OD = convert_to_labels(y_test_OD)

    # Normalize the radar data (Note that the maximum radar value is 150)
    x_test_ID = ((np.log1p(np.log1p(x_test_ID)) / np.log1p(np.log1p(150))) * 2) - 1  
    x_test_OD = ((np.log1p(np.log1p(x_test_OD)) / np.log1p(np.log1p(150))) * 2) - 1 

    # Convert to Tensors
    x_test_ID = torch.tensor(x_test_ID)
    x_test_OD = torch.tensor(x_test_OD)
    y_test_ID = torch.tensor(y_test_ID)
    y_test_OD = torch.tensor(y_test_OD)
 
  return x_test_ID, y_test_ID, x_test_OD, y_test_OD


def get_dataset(dataset_path, train_portion=True, validation_percent=0.3, batchsize=256, num_workers=1):
  if train_portion:
    x_train, y_train = load_entries(dataset_path, train_portion=train_portion)
    valid_indices = torch.rand(len(x_train)) <= validation_percent
    train_indices = ~valid_indices

    x_valid = x_train[valid_indices]
    y_valid = y_train[valid_indices]

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    dataset_train = TensorDataset(x_train, y_train)
    dataset_valid = TensorDataset(x_valid, y_valid)

    # Create DataLoaders
    train_loader   = DataLoader(dataset_train, batch_size=batchsize, shuffle=True,  pin_memory=True , num_workers=num_workers, persistent_workers=True)
    valid_loader   = DataLoader(dataset_valid, batch_size=batchsize, shuffle=False, pin_memory=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, valid_loader

  else:
    x_test_ID, y_test_ID, x_test_OD, y_test_OD = load_entries(dataset_path, train_portion=train_portion)

    dataset_ID    = TensorDataset(x_test_ID, y_test_ID)
    dataset_OD    = TensorDataset(x_test_OD, y_test_OD)

    # Create DataLoaders
    test_ID_loader = DataLoader(dataset_ID,    batch_size=batchsize, shuffle=False, pin_memory=False, num_workers=num_workers, persistent_workers=True)
    test_OD_loader = DataLoader(dataset_OD,    batch_size=batchsize, shuffle=False, pin_memory=False, num_workers=num_workers, persistent_workers=True)

    return test_ID_loader, test_OD_loader


# End of Document