###################################################################################################
# Project:  CIROH - Doppler Radar Precipitation Prediction
# Program:  DRPP_11_Trainer.py
# Author:   Marshall Rosenhoover (marshall.rosenhoover@uah.edu) (marshallrosenhoover@gmail.com)
# Created:  2025-03-26
# Modified: 2025-06-11
#
#    This file contains the trainer class to train the neural network
#    
####################################################################################################

import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import itertools  # used for iterating over rows and columns

####################################################################################################
# Trainer
#
#    Takes the gnn and trains it for node prediction.  
#
#  Inputs:
#    model                  The model to train.
#    train_dataset          A dataloader with the train data.
#    valid_dataset          A dataloader with the valid data.
#    test_dataset           A dataloader with the test data.
#    optimizer              The optimizer to train the model.
#    save_every             Save the model after *save every* epochs
#    patience               Number of epochs to wait before early stopping
#    snapshot_path          The path to save the model. If the snapshot already exists, load the snapshot.
#    best_model_path        The path to save the best model.
#    
####################################################################################################

class Trainer:
  def __init__(self,
               model,
       train_dataset,   
       valid_dataset,     
           optimizer,
          save_every,
            patience,
       snapshot_path,
     best_model_path,
      ) -> None:
    
    # General Trainer Variables
    self.model             = model.cuda()
    self.train_dataset     = train_dataset
    self.valid_dataset     = valid_dataset
    self.optimizer         = optimizer
    self.save_every        = save_every
    self.patience          = patience
    self.criterion = nn.CrossEntropyLoss()
    self.best_valid_accuracy = 0

    self.patience_iter  = 0              
    self.epochs_run     = 0
    self.train_history  = [] 
    self.valid_history  = []
    
    self.best_model_path     = best_model_path

    self.snapshot_path   = snapshot_path
    if os.path.exists(snapshot_path):
      print("Loading snapshot")
      self._load_snapshot(self.snapshot_path)     

  ####################################################################################################
  # _save_snapshot
  #
  #    Takes the model with its current state and saves it to the path passed in to the trainer.
  #    The snapshot saves the state_dict of the model, the current number of epochs
  #    that have been completed, the training loss, and the validation loss.
  #
  #  Inputs:
  #    path               The path to save the model. 
  #
  ####################################################################################################

  def _save_snapshot(self, path):
    snapshot = {
      "MODEL_STATE"   : self.model.state_dict(),  
      "EPOCHS_RUN"    : self.epochs_run,
      "TRAIN_HISTORY" : self.train_history,
      "VALID_HISTORY" : self.valid_history,
      "PATIENCE_ITER" : self.patience_iter,
    }

    torch.save(snapshot, path)
    print(f"\nEpoch {self.epochs_run} | Training snapshot saved at {path}", end=" ")

  ####################################################################################################
  # _load_snapshot
  #
  #    Given the path passed into the trainer, it will load the given snapshot of the model.
  #
  #  Inputs:
  #    path               The path to the model to load.
  #
  ####################################################################################################

  def _load_snapshot(self, path):
    snapshot = torch.load(path, weights_only=True)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run     = snapshot["EPOCHS_RUN"]
    self.train_history  = snapshot["TRAIN_HISTORY"]
    self.valid_history  = snapshot["VALID_HISTORY"]
    self.patience_iter  = snapshot["PATIENCE_ITER"]
    self.best_valid_accuracy = max(self.valid_history)

    print(f"Loaded Model From {path}.")  
  
  ####################################################################################################
  # _save_check
  #
  #    Checks to see if the model should be saved due to validation loss and on epochs run. 
  #
  ####################################################################################################
  def _save_check(self):
    # Check to save model based on validation loss
    if self.valid_history[-1] > self.best_valid_accuracy:
      self.patience_iter = 0
      self.best_valid_accuracy = self.valid_history[-1]
      self._save_snapshot(self.best_model_path)

    # Check to save model based on training time
    if self.epochs_run % self.save_every == 0: 
      self._save_snapshot(self.snapshot_path)  
  
  ####################################################################################################
  # train
  #
  #    The function called to train the model. It will train the model with respect to the datasets 
  #    passed into the trainer class. It will train until it reached max_epochs. There is no other
  #    method for stopping the training for the model minus stopping the process running the training.
  #
  #  Inputs:
  #    max_epochs         The number of epochs to train the model.
  #
  ####################################################################################################

  def train(self, max_epochs: int):
    #lats, longs, elevation, station_locs, start_lat, start_long = self.train_dataset.dataset.get_auxilary_information()
    for epoch in range(self.epochs_run, max_epochs):
      self.epochs_run += 1

      # Train current epoch
      train_loss = self._train_epoch()     
      self.train_history.append(train_loss)

      # Validate current epoch
      F1 = self._evaluate_epoch(self.valid_dataset, "valid")         
      self.valid_history.append(F1)
      
      # Check to save best model
      self._save_check()

      # Check for early stopping
      if self.patience_iter >= self.patience:
        print(f"Early stopping at epoch {self.epochs_run}. No improvement in validation loss for {self.patience} consecutive epochs.")
        break
      
      self.patience_iter += 1

  ####################################################################################################
  # _train_epoch
  #
  #    Trains the model for one iteration of the training data. It sends off data in batches to train 
  #    and stores the overall loss. 
  #
  #  Returns:
  #    total_loss         Returns the overall loss with respect to the training data.
  #
  ####################################################################################################
  
  def _train_epoch(self):
    print(f"\nEpoch {self.epochs_run}", end=" ")  
    self.model.train()  # Set the model to train mode
    total_loss = 0.0
  
    for x, y in self.train_dataset:
      loss = self._train_batch(x, y)
      total_loss += loss

    total_loss /= len(self.train_dataset)

    print(f"| train loss {total_loss}", end=" ")
    return total_loss

  ####################################################################################################
  # _train_batch
  #
  #    Trains the model on the batch passed in. It resets the gradients, makes a prediction, then
  #    computes the loss and back propagates the information. 
  #
  #  Inputs:
  #    batch              The batch of data to run. It should contain the features and predictions
  #                       for all the nodes.
  #  Returns:
  #    loss               The error between the prediction of the model and the batch output.
  #
  ####################################################################################################

  def _train_batch(self, x, y):
    self.optimizer.zero_grad()       
    x = x.cuda()
    y = y.cuda()

    logits = self.model(x)
    loss = self.criterion(logits, y)

    loss.backward()
    self.optimizer.step()
    return loss.detach()

  ####################################################################################################
  # _evaluate_epoch
  #
  #    Evaluates the model for one iteration of the dataset passed in. It sends off data in batches to be
  #    evaluated and stores the overall loss. 
  #
  #  Inputs:
  #    dataset            The dataset to evaluate the model on.
  #  Returns:
  #    total_loss         Returns the overall loss with respect to the dataset.
  #
  ####################################################################################################
  
  def _evaluate_epoch(self, dataset, loss_name, plot=False):    
    self.model.eval()  # Set the model to evaluation mode
    all_predicted = []
    all_gt        = []
    all_radar     = []
    for x, y in dataset:
      predicted, gt, radar = self._evaluate_batch(x, y)
      all_predicted.append(predicted)
      all_gt.append(gt)
      all_radar.append(radar)

    all_predicted = torch.cat(all_predicted).cuda()
    all_gt = torch.cat(all_gt).cuda()
    all_radar = torch.cat(all_radar).cuda()


    if plot:
      macro_f1 = self.compute_metrics(all_gt, all_radar,     f"{loss_name} Radar vs Station", plot=plot)  # Plot and give information about the test dataset


    macro_f1 = self.compute_metrics(all_gt, all_predicted, f"{loss_name} Model vs Station", plot=plot)    # Information about the model
    print(f"| eval F1 {macro_f1}", end=" ")

    return macro_f1
  
  ####################################################################################################
  # _run_batch
  #
  #    Runs the model on the batch passed in and returns the loss for the batch. 
  #
  #  Inputs:
  #    batch              The batch of data to run. It should contain the features and predictions
  #                       for all the nodes.
  #  Returns:
  #    loss               The error between the prediction of the model and the batch output.
  #
  ####################################################################################################

  def _evaluate_batch(self, x, y):
    with torch.no_grad():
      x = x.cuda()
      y = y.cuda()
      logits       = self.model(x)               

      x = x[:, -1, 16, 16]
      x = torch.expm1(torch.exp(((x + 1) * np.log1p(np.log1p(150))) / 2) - 1)  # Unnormalize the radar data
      x = self.classify_x(x)                                                    # Classify the radar data
      _, predicted = torch.max(logits, dim=1)                                   # Get the class prediction from the model

    return predicted, y, x

  def classify_x(self, values):
    classes = torch.empty_like(values, dtype=torch.long)
    classes[values < 0.1] = 0
    classes[(values >= 0.1) & (values < 1.0)] = 1
    classes[(values >= 1.0) & (values < 2.5)] = 2
    classes[values >= 2.5] = 3
    return classes

  ####################################################################################################
  # compute_metrics
  #
  #    Computes the metrics of F1, precision, and recall for both micro and macro. Only returns
  #    macro F1. If plot is true, you will get the details of all the information. 
  #
  #  Inputs:
  #    true_labels        The ground truth to compare against
  #    predicted_labels   The predicted labels
  #    name               Used for plotting, name of what you are comparing
  #    num_classes        Number of classes
  #    plot               Plots out a confusion matrix and gives macro and micro details.
  #
  #  Returns:
  #    macro_F1           The class wide macro score.
  #
  ####################################################################################################

  def compute_metrics(self, true_labels, predicted_labels, name, num_classes=4, plot=False):
    # 1. Build an empty confusion matrix of shape [num_classes, num_classes].
    #    Rows = true class, Columns = predicted class.
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.long).cuda()

    # 2. Fill the confusion matrix
    for t, p in zip(true_labels, predicted_labels):
      conf_mat[t, p] += 1

    # 3. Accuracy: (sum of diagonal) / (sum of all cells)
    total_correct = conf_mat.trace().item()
    total_samples = conf_mat.sum().item()
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # 4. Precision and recall per class, then macro-average
    precision_per_class = []
    recall_per_class    = []
    f1_per_class        = []

    for c in range(num_classes):
      tp = conf_mat[c, c].item()
      fp = (conf_mat[:, c].sum() - tp).item()  # predicted as c that aren't actually c
      fn = (conf_mat[c, :].sum() - tp).item()  # actual c that weren't predicted as c

      # Avoid division by zero
      precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall_c    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

      # Compute F1 score for the class
      if (precision_c + recall_c) > 0:
        f1_c = 2 * (precision_c * recall_c) / (precision_c + recall_c)
      else:
        f1_c = 0.0

      precision_per_class.append(precision_c)
      recall_per_class.append(recall_c)
      f1_per_class.append(f1_c)

    macro_precision = sum(precision_per_class) / num_classes
    macro_recall    = sum(recall_per_class) / num_classes
    macro_f1        = sum(f1_per_class) / num_classes

    if plot:
      print(f"{name} -----------")

      print(f"| {name} accuracy        {accuracy} ")
      print(f"| {name} macro precision {macro_precision} ")
      print(f"| {name} macro recall    {macro_recall} ")
      print(f"| {name} macro F1        {macro_f1} ")

      for i, (precision, recall, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        print(f"| {name} class {i} precision {precision} ")
        print(f"| {name} class {i} recall    {recall} ")
        print(f"| {name} class {i} F1        {f1} ")

      classes = ['[0, 0.1)', '[0.1, 1.0)', '[1.0, 2.5)', '[2.5, Inf)']  # Change these to your actual class labels if needed
      self.plot_normalized_confusion_matrix(conf_mat, classes, name)

    return macro_f1

  def plot_normalized_confusion_matrix(self, conf_mat, classes, name):
    # Convert the tensor to a NumPy array if necessary
    conf_matrix_np = conf_mat.cpu().numpy()

    # Compute row sums and normalized percentages (avoid division by zero)
    row_sums = conf_matrix_np.sum(axis=1, keepdims=True)
    norm_conf_matrix = (conf_matrix_np / row_sums * 100)
    norm_conf_matrix = np.nan_to_num(norm_conf_matrix)  # replaces NaNs with zero

    # Plotting
    plt.figure(figsize=(8, 6))
    im = plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.title("Normalized Confusion Matrix (%)")
    cbar = plt.colorbar(im, ticks=[0, 25, 50, 75, 100])
    cbar.ax.set_ylabel("Fraction", rotation=270, labelpad=15)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Annotate each cell with both count and percentage
    thresh = norm_conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix_np.shape[0]), range(conf_matrix_np.shape[1])):
      plt.text(j, i, f"{conf_matrix_np[i, j]}\n({norm_conf_matrix[i, j]:.1f}%)",
                horizontalalignment="center",
                color="white" if norm_conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f"./{name}.png")

    # Clear the plot when done
    plt.close()

# End of Document