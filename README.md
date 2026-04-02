# HEAL ITALIA PROJECT - Spoke 2
![Pipeline](Logo-Header.png)
# Lung Nodule Segmentation (2D)

Deep learning-based 2D lung nodule segmentation on CT scans using slice-wise training.

## Overview

This repository contains the implementation of 2D segmentation models for detecting lung nodules from CT images. The pipeline processes slices extracted from volumetric CT scans and trains convolutional neural networks for segmentation.

## Method

* Slice-wise processing of CT scans
* Attention Residual U-Net architecture
* Hybrid loss functions (Dice + BCE)
* Preprocessed lung CT inputs

## Features

* Data generator for 2D slices
* Model training and evaluation
* Prediction and visualization

## Repository Structure

```text
src/        # model, training, data generator
notebooks/  # experiments and visualization
results/    # outputs and figures
```

## Status

Initial version uploaded for project submission. Further improvements and cleanup will be added.

