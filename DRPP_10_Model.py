###################################################################################################
# Project:  CIROH - Doppler Radar Precipitation Prediction
# Program:  DRPP_10_Model.py
# Author:   Marshall Rosenhoover (marshall.rosenhoover@uah.edu) (marshallrosenhoover@gmail.com)
# Created:  2025-02-03
# Modified: 2025-06-10
#
#    This file contains code for the creation of the model. 
#
####################################################################################################

import torch.nn as nn
import torchvision.models as models

class Resnet_Model(nn.Module):
  def __init__(self, in_channels : int, 
                     num_classes : int) -> None:
    super(Resnet_Model, self).__init__()
    self.initial_conv = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.SiLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
      nn.SiLU(inplace=True),  
    )

    self.resnet         = models.resnet101(pretrained=False)
    self.resnet.conv1   = nn.Identity()  # Remove the original first convolution
    self.resnet.maxpool = nn.Identity()  # Remove max pooling

    num_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_features, num_classes)


  def forward(self, x):  
    x = self.initial_conv(x)
    x = self.resnet(x)
    return x

# End of Document