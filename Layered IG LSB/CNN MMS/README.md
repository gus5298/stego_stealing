# Steganography & Classification Pipeline

This project implements a steganographic system for hiding messages in grayscale images using a deep learning-based approach. It selects important pixels using Integrated Gradients (IG), embeds a secret message using Least Significant Bit (LSB) modification, and evaluates the impact on classification and image quality.

## Features

- 🔐 **Message Embedding**: Embed secret messages into selected pixels using LSB based on IG saliency.
- 🧠 **Model Inference**: Use a pretrained CNN to classify images (e.g., Attack vs. Normal).
- 📊 **Metrics & Visualization**:
  - Image quality: PSNR, SSIM, MSE, DCT difference
  - Classification: Accuracy, Confusion Matrix
  - Saliency Maps: Overlay of pixel importance
- 📁 **Output Generation**:
  - Stego images with embedded messages
  - Visualization images with red-marked embedded pixels
  - Saved confusion matrices in CSV and image formats

## Prerequisites

- Python 3.7+
- GPU recommended for faster inference with TensorFlow

## Installation

1. Clone this repository or download the folder.

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
