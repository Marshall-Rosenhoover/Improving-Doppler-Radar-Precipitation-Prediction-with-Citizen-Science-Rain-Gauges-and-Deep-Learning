# Improving Doppler Radar Precipitation Prediction with Citizen Science Rain Gauges and Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository accompanies the paper:

**Improving Doppler Radar Precipitation Prediction with Citizen Science Rain Gauges and Deep Learning**  
Marshall Rosenhoover, John Rushing, John Beck, Kelsey White, and Sara Graves  
*Sensors (MDPI), 2025*

📄 **[Link to Paper (coming soon)]()**  
📊 **[Dataset on Kaggle](https://www.kaggle.com/datasets/rosenhoover/hawaii-radar-to-rain-gauge-rain-rate-prediction)**  

---

## 📘 Overview

This work presents a deep learning framework that improves real-time Doppler radar rainfall estimates using citizen science rain gauge data and a radar-guided accumulation fitting method.

Key contributions:
- A method for reconstructing continuous rainfall accumulation functions from discrete gauge and radar observations
  ![Accumulation ](images/Accumulation_Reconstruction.png)
- A quality control framework for validating citizen weather station data
- A deep ResNet-101 model trained to classify rain intensity from radar images
- Improved classification performance over NOAA’s operational Surface Precipitation Rate (SPR) radar product

---

## 📂 Contents
Since we only recieved permission from Weather Underground to publish out dataset, not the preprocessed information, we are only uploading the files to train the model associated with the dataset. The paper defines how to construct the rainfall accumulation curves from rain gauge observations and radar. Moreover, it describes how to quality control Personal Weather stations with only radar. 

In light of this, our project only has four files:
- `DRPP_00_Main.py`    - Main file to setting file locations and model parameters
- `DRPP_01_Dataset.py` - Loads in the dataset from the filepath
- `DRPP_10_Model.py`   - Imports the Resnet101 model and modifies the beginning convolutions
- `DRPP_11_Trainer.py` - Trains the model and allows the model to be evaluated.

---

## 🚀 Getting Started


