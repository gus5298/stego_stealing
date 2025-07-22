# Layered IG + LSB Steganography with Random Forest Classifier

This project implements a steganography system that embeds secret messages into grayscale images using the Least Significant Bit (LSB) method. Important pixels for embedding are selected using Integrated Gradients (IG). The modified images are then evaluated using a Random Forest (RF) classifier.

---

## 📦 Features

- Selects pixel locations for embedding using Integrated Gradients (IG)
- Embeds secret messages using LSB modification
- Uses a trained Random Forest model (`rf_model.pkl`) to classify stego images
- Calculates classification accuracy and confusion matrix
- Computes image quality metrics: PSNR, SSIM, MSE, DCT differences
- Exports results to Excel and visualizes embedded pixels

