# Car Detection and Classification using VGG16 and Selective Search

This project aims to detect and classify cars in images using a Convolutional Neural Network (CNN) based on the VGG16 architecture and Selective Search for region proposals.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses a pre-trained VGG16 model to classify regions of images as either containing a car or not. The regions are proposed using Selective Search, and the Intersection over Union (IoU) metric is used to evaluate the accuracy of the bounding boxes.

## Dataset

The dataset consists of images and corresponding bounding box annotations in a CSV file. The annotations file contains columns for the image name and the coordinates of the bounding boxes (xmin, ymin, xmax, ymax).

## Usage

1. Set the paths for the training images and annotations in the script:
    ```python
    train_path = 'path/to/training_images'
    annot = 'path/to/annotations.csv'
    ```

2. Run the script to start the training process:
    ```bash
    python train.py
    ```

## Model Training

The training process involves the following steps:

1. **Loading Data:** Read the images and annotations, and preprocess them for training.
2. **Selective Search:** Generate region proposals using Selective Search.
3. **Intersection over Union (IoU):** Compute IoU to evaluate the bounding boxes.
4. **Data Augmentation:** Apply data augmentation techniques to the training data.
5. **Model Training:** Train the VGG16 model with the proposed regions and their labels.

### Example Code

```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load annotations
train_path = 'C:/Users/shrey/Desktop/Ann/training_images'
annot = 'C:/Users/shrey/Desktop/Ann/train_solution_bounding_boxes (1).csv'
labels = pd.read_csv(annot)

# Function to display images with bounding boxes
def display_image(path, box=None, specific=False): 
    if not specific:
        img = Image.open(path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        rect = patches.Rectangle((int(box['xmin']), int(box['ymin'])), int(box['xmax']-box['xmin']), int(box['ymax']-box['ymin']), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
    else:
        plt.imshow(path)
        fig, ax = plt.subplots()
        ax.imshow(path)
        for i in box:
            rect = patches.Rectangle((int(i[0]-(i[2]/2)), int(i[1]-(i[3]/2))), int(i[2]), int(i[3]), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

# Example usage
display_image(train_path + '/' + labels['image'][0], labels[0:1])

# Create Selective Search object
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Function to calculate Intersection over Union (IoU)
def get_iou(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou
