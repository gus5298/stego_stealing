# PRNG-Based Steganography with Random Forest Classification

This project performs layered steganography using a Pseudorandom Number Generator (PRNG) to embed messages and secret seeds into grayscale images. A Random Forest (RF) classifier is used to analyze both original and stego images for classification. Image quality and message integrity are evaluated, and detailed results are exported including confusion matrices and metric plots.

## Features

- Seed-based PRNG pixel selection
- Two-stage message embedding using LSB
- Key and message recovery validation
- Random Forest model classification
- Evaluation metrics: MSE, PSNR, SSIM, DCT Difference
- Confusion matrix visualization
- Summary plots for key/message match

## Folder Structure

```
├── input/                  # Folder containing input images, subfolders = class names
├── model/                  # Folder containing trained RF model (.joblib)
├── stego_output/           # Output folder for stego images
├── results/                # Output folder for analysis and plots
├── main.ipynb              # Main notebook
├── README.md
├── requirements.txt
```

## Usage

1. Place test images in subfolders under `input/`, named after their class labels (e.g., `Normal/`, `Attack/`).
2. Put the trained RF model in `model/` folder.
3. Run the notebook `main.ipynb`. Update paths for:
   - `input_folder`
   - `model_path`
   - `stego_output_folder`
   - `results_output_folder`
   - `seed_secret_key`

## Example

```python
input_folder = "input"
model_path = "model/rf_model.joblib"
stego_output_folder = "stego_output"
results_output_folder = "results"
seed_secret_key = "Gustavo_Sanchez"

process_folder(input_folder, model_path, stego_output_folder, results_output_folder, seed_secret_key)
```

## Output

- Stego images saved in `stego_output/`
- Results in `results/`:
  - Excel report
  - Confusion matrix heatmaps
  - Match count plots

## Dependencies

See `requirements.txt` for the full list of dependencies.
