# CNN MMS - Steganography Classification with Convolutional Neural Networks

This Jupyter Notebook (`CNN MMS.ipynb`) implements a Convolutional Neural Network (CNN) for classifying grayscale images in the **MMS dataset**, specifically in the context of **steganography**. It evaluates the effectiveness of message embedding in images and their impact on classification performance.

## 🔍 Overview

The notebook performs the following tasks:

- Loads the MMS dataset with three classes: `Normal`, `Attack`, and `Faulty`
- Builds and trains a CNN classifier for image classification
- Evaluates model performance before and after message embedding
- Supports testing with images modified via steganographic methods
- Outputs confusion matrices and accuracy metrics

## 🧠 Model

- Architecture: Basic CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Input: Grayscale images (likely 64×64)
- Output: Multi-class classification (3 output neurons with softmax)

## 🧪 Evaluation Metrics

- Accuracy
- Confusion matrix (visual and CSV export)
- Optionally: Precision, recall, F1-score

## 📂 Directory Structure (Expected)

```
├── CNN MMS.ipynb
├── /dataset/
│   ├── Normal/
│   ├── Attack/
│   └── Faulty/
├── /stego_images/
│   ├── Normal/
│   ├── Attack/
│   └── Faulty/
├── /outputs/
│   ├── confusion_matrix.png
│   └── results.csv
```

## 🛠️ Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

## ▶️ How to Run

1. Place your dataset folders (`Normal`, `Attack`, `Faulty`) inside a `dataset/` directory.
2. Open and run all cells in `CNN MMS.ipynb`.
3. Optionally, place steganographically modified images inside `stego_images/` to test model robustness post-embedding.

## 📊 Results

- Classification accuracy is computed on both original and stego images.
- Confusion matrices are saved as `.png` and optionally `.csv` for analysis.

## 📌 Use Case

This notebook is designed to support research in:

- Information hiding and steganography
- Adversarial robustness
- Lightweight CNNs for industrial or embedded vision applications
