# Advanced Crack Detection in Concrete Structures Using Dense ResU-Net with T-Max-Avg Pooling

This project implements a Dense ResU-Net model with T-Max-Avg pooling to detect cracks in concrete structures. The model is trained and evaluated on a dataset of images to classify whether each image contains a crack or not.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Reference Paper](#reference-paper)

## Introduction
Crack detection in concrete structures is crucial for maintaining the structural integrity and safety of buildings, bridges, and other infrastructures. This project leverages a Dense ResU-Net model with a T-Max-Avg pooling layer for binary classification of images.

## Dataset
The dataset used in this project consists of 19,000 training images and a separate set of testing images. The images are organized into `train/` and `test/` directories, with subdirectories for `crack` and `no_crack` classes.

## Model Architecture
The model is based on a Dense ResU-Net architecture with a T-Max-Avg pooling layer. The architecture includes an encoder-decoder structure with skip connections to preserve spatial information.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Aadik1ng/crack_detection.git
    cd crack_detection
    ```

2. Install the required dependencies:
    ```bash
    pip install torch torchvision matplotlib numpy tqdm
    ```

3. Ensure your dataset is organized as follows:
    ```
    crack_detection/
    ├── train/
    │   ├── crack/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── no_crack/
    │       ├── img1.jpg
    │       ├── img2.jpg
    │       └── ...
    ├── test/
    │   ├── crack/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── no_crack/
    │       ├── img1.jpg
    │       ├── img2.jpg
    │       └── ...
    └── ...
    ```

## Training
To train the model, run the following command:
```bash
python main.py
```
The script will train the model for 50 epochs and print the training and validation loss and accuracy for each epoch.

## Training
After training, the model will be evaluated on the test dataset. The script will print the test loss and accuracy.

## Results
The training process will output the loss and accuracy for each epoch. The final model will be evaluated on the test set to provide the test loss and accuracy.

## Reference Paper
This implementation is based on the following paper:

Title: An Innovative Dense ResU-Net Architecture With T-Max-Avg Pooling for Advanced Crack Detection in Concrete Structures

Authors: Ali Sarhadi, Mehdi Ravanshadnia, Armin Monirabbasi, Milad Ghanbari

DOI: 10.1007/s41062-023-00847-0

This README file provides a comprehensive overview of the project, including installation instruction
