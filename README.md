# sargassum_RESNet

This repository contains code for training and validating a UNet model with a ResNet34 backbone, designed to identify Sargassum algae from multispectral images.

## Model Overview

The model uses a modified UNet architecture with ResNet34 as the encoder backbone. It leverages pre-trained weights from ImageNet and is designed for segmenting floating algae from multispectral satellite data.

### Features:
- **ResNet34 Backbone**: Pre-trained ResNet34 improves feature extraction.
- **Multi-channel Input**: Supports 8-channel input (excluding a specific band, configurable).
- **CUDA Support**: The model can be accelerated using GPU.
- **Custom Loss & Evaluation Metrics**: Includes F1 score, precision, recall, and loss plotting for comprehensive evaluation.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- CUDA-enabled GPU (optional for training)
- Rasterio (for reading multispectral images)

Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
