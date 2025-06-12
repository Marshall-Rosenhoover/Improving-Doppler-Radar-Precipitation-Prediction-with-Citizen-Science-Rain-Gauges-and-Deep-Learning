###################################################################################################
# Project:  CIROH - Doppler Radar Precipitation Prediction
# Program:  DRPP_00_Main.py
# Author:   Marshall Rosenhoover (marshall.rosenhoover@uah.edu) (marshallrosenhoover@gmail.com)
# Created:  2025-03-26
# Modified: 2025-06-11
#
#    This file contains code for the creation of the model. 
#    
#    Usage: python IC_00_Main.py
#
####################################################################################################

import torch
from DRPP_01_Dataset import get_dataset
from DRPP_10_Model import Resnet_Model
from DRPP_11_Trainer import Trainer


if __name__ == '__main__':

    # Variables for the Trainer
    save_every    = 10
    patience      = 100
    total_epochs  = 400
    snapshot_path = "./save_every_model.pt"
    best_path     = "./best_model.pt"
    batchsize     = 256
    valid_percent = 0.3

    # Dataset Variables
    dataset_path = "./Hawaii_PWS_Precipitation.npz"

    # ID = In Distribution (Generalization on Observed Spatial Clusters)
    # OD = Out of Distribution (Generalization to Unseen Spatial Clusters) 

    train, valid = get_dataset(dataset_path, train_portion=True, validation_percent=valid_percent, batchsize=batchsize) 

    model = Resnet_Model(in_channels=11, num_classes=4)

    optimizer = torch.optim.Adam(model.parameters(),  lr=0.0002)
    trainer = Trainer(model=model, train_dataset=train, valid_dataset=valid, optimizer=optimizer, save_every=save_every, patience=patience, snapshot_path=snapshot_path, best_model_path=best_path)

    trainer.train(total_epochs)
    print("Training Complete.")

    test_ID, test_OD = get_dataset(dataset_path, train_portion=False, validation_percent=valid_percent, batchsize=batchsize)

    # Evaluate the model
    trainer._load_snapshot(best_path)                    # Load the best model
    test_loss = trainer._evaluate_epoch(test_ID, "Test Dataset - Generalization on Observed Spatial Clusters")  # Test the model 
    print("---------------------")
    print(f"Evaluation of Model on Test Dataset - Generalization on Observed Spatial Clusters | F1 Score {test_loss}")
    test_loss = trainer._evaluate_epoch(test_OD,  "Test Dataset - Generalization to Unseen Spatial Clusters")  # Test the model 
    print("---------------------")
    print(f"Evaluation of Model on Test Dataset - Generalization to Unseen Spatial Clusters | F1 Score {test_loss}")


# End of Document