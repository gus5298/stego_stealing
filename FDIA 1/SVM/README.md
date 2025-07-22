# Anomaly Detection using One-Class SVM (FDIA Dataset)

This project performs anomaly detection on grayscale images using a **One-Class SVM** model. It is designed for applications like detecting False Data Injection Attacks (FDIAs) in critical infrastructure systems.

## 📁 Project Structure

- `RF FDIA 1.ipynb`: Main Jupyter notebook for training and testing the One-Class SVM.
- `requirements.txt`: Dependencies required to run the notebook.
- `results/`: Directory where output CSV with predictions is saved (created automatically).
- `data/normal/`: Folder containing training images of the "normal" class.
- `data/test/`: Folder containing test images (normal and anomalies).

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
