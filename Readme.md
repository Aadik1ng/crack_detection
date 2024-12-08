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
The model is based on a Dense ResU-Net architecture with a T-Max-Avg pooling layer. Here's a detailed explanation of each component:

### Encoder
The encoder part of the network extracts features from the input images. It consists of three convolutional blocks:
- Each block has two convolutional layers with ReLU activation, followed by a max pooling layer to downsample the feature maps.
- The number of filters increases with depth (64, 128, and 256).

### Bottleneck
The bottleneck layer captures the most abstract features of the input image:
- It consists of two convolutional layers with 512 filters each and ReLU activation.
- The T-Max-Avg pooling layer is applied after the convolutions to combine max and average pooling results, preserving important features while reducing spatial dimensions.

### Decoder
The decoder part of the network reconstructs the input image from the abstract features:
- It consists of three up-convolution (transposed convolution) blocks to upsample the feature maps.
- Skip connections from the encoder are concatenated with the upsampled feature maps to preserve spatial information from earlier layers.
- The number of filters decreases with depth (256, 128, and 64).

### Output Layer
The output layer produces a single scalar value for each input image, indicating the presence of a crack:
- A 1x1 convolutional layer reduces the feature maps to the desired number of output channels.
- Global average pooling is applied to produce a single value per feature map.
- A fully connected layer outputs the final binary classification.

### T-Max-Avg Pooling Layer
The T-Max-Avg pooling layer combines the strengths of both max pooling and average pooling:
- Max pooling captures the most prominent features by taking the maximum value within the pooling window.
- Average pooling provides a smoothing effect by averaging the values within the pooling window.
- The T-Max-Avg pooling layer combines these two pooling methods using a parameter T to balance their contributions.

### Model Summary
The architecture can be summarized as follows:
1. **Input Layer**: Input image of shape (256, 256, 3).
2. **Encoder**:
   - Conv Block 1: Conv2D(64) -> Conv2D(64) -> MaxPooling2D
   - Conv Block 2: Conv2D(128) -> Conv2D(128) -> MaxPooling2D
   - Conv Block 3: Conv2D(256) -> Conv2D(256) -> MaxPooling2D
3. **Bottleneck**:
   - Conv Block 4: Conv2D(512) -> Conv2D(512)
   - T-Max-Avg Pooling
4. **Decoder**:
   - UpConv Block 1: ConvTranspose2D(256) -> Concatenate -> Conv Block (256)
   - UpConv Block 2: ConvTranspose2D(128) -> Concatenate -> Conv Block (128)
   - UpConv Block 3: ConvTranspose2D(64) -> Concatenate -> Conv Block (64)
5. **Output Layer**:
   - Conv2D(128)
   - Global Average Pooling
   - Fully Connected Layer (1)



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
