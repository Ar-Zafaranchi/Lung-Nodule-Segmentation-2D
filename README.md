# Lung Nodule Segmentation (2D)

Deep learning-based 2D lung nodule segmentation on CT scans using slice-wise training.

## Overview

This repository contains the implementation of 2D segmentation models for detecting lung nodules from CT images. The pipeline processes slices extracted from volumetric CT scans and trains convolutional neural networks for segmentation.

## Method

* Slice-wise 2D training
* Attention Residual U-Net architecture
* Loss functions: Dice, BCE, and hybrid combinations
* Preprocessed inputs from CT pipeline

## Features

* Data generator for slice-based training
* Model training and evaluation scripts
* Visualization of predictions

## Structure

```text
src/        # models, training, generators
notebooks/  # experiments and visualization
results/    # outputs and figures
```

## Status

Initial version. Code cleanup and documentation are in progress.
