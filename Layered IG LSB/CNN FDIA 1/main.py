# === FIXED analyze_with_cnn-style Predictions for IG + LSB Pipeline ===

import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# === CONFIGURATION === #
IMG_SIZE = (128, 128)
NUM_PIXELS = 32
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2
MESSAGE_BITS = 4 * 8
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Attack']

# === MESSAGE UTILS === #
def generate_random_message():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

def string_to_bits(s):
    return [int(b) for c in s.encode('utf-8') for b in format(c, '08b')][:MESSAGE_BITS]

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

# === COORDINATE CONVERSION === #
def coords_to_bits(coords):
    bits = []
    for y, x in coords:
        yb = format(y, f'0{BITS_PER_COORD}b')
        xb = format(x, f'0{BITS_PER_COORD}b')
        bits.extend(int(b) for b in yb + xb)
    return bits

def bits_to_coords(bits):
    coords = []
    for i in range(0, len(bits), 32):
        y = int(''.join(map(str, bits[i:i+16])), 2)
        x = int(''.join(map(str, bits[i+16:i+32])), 2)
        coords.append((y, x))
    return coords

# === CLEANING FUNCTION === #
def clean_message(s):
    return ''.join(c for c in s if c.isprintable())

# === EMBEDDING === #
def embed_lsb(image, bits, coords):
    flat = image.flatten()
    for i, (y, x) in enumerate(coords):
        idx = y * image.shape[1] + x
        flat[idx] = (flat[idx] & ~1) | bits[i]
    return flat.reshape(image.shape)

def embed_metadata(image, meta_bits):
    flat = image.flatten()
    for i in range(len(meta_bits)):
        flat[i] = (flat[i] & ~(1 << 7)) | (meta_bits[i] << 7)
    return flat.reshape(image.shape)

# === IG-BASED IMPORTANT PIXELS === #
def get_ig_pixels(model, image_tensor, top_k=NUM_PIXELS):
    def loss_fn(output): return output[:, 0]  # Single output from sigmoid
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    saliency_map = saliency(loss_fn, image_tensor)
    saliency_map = np.abs(saliency_map[0])
    if saliency_map.ndim == 3:
        saliency_map = np.mean(saliency_map, axis=-1)
    flat = saliency_map.flatten()
    indices = np.argsort(flat)[-top_k:]
    ys, xs = np.unravel_index(indices, saliency_map.shape)
    return list(zip(ys, xs))

# === DECODING === #
def decode_message_and_coords(image):
    flat = image.flatten()
    coord_bits = [(flat[i] >> 7) & 1 for i in range(RESERVED_BITS)]
    coords = bits_to_coords(coord_bits)
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]
    return bits_to_string(msg_bits[:MESSAGE_BITS]), coords

# === QUALITY METRICS === #
def calculate_mse(original, stego):
    return np.mean((original.astype(np.float32) - stego.astype(np.float32)) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def calculate_ssim(original, stego):
    return ssim(original, stego)

def compute_dct_difference(original, stego):
    original_dct = dct(dct(original.T, norm='ortho').T, norm='ortho')
    stego_dct = dct(dct(stego.T, norm='ortho').T, norm='ortho')
    diff = np.abs(original_dct - stego_dct)
    return np.mean(diff), np.max(diff)

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel, stego_output_folder):
    model = load_model(model_path)
    results = []
    original_labels = []
    original_preds = []
    stego_preds = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, filename in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] Processing {filename}")
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Could not read {filename}. Skipping.")
            continue

        image = cv2.resize(image, IMG_SIZE)
        norm_img = image.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(norm_img, axis=(0, -1))

        prob_orig = float(model.predict(input_tensor, verbose=0)[0])
        pred_class_orig = CLASS_NAMES[int(prob_orig >= 0.5)]
        print(f"Original prediction prob: {prob_orig:.4f} → Predicted class: {pred_class_orig}")
        original_preds.append(pred_class_orig)

        folder_name = os.path.basename(os.path.normpath(input_folder)).lower()
        if folder_name == "normal":
            true_label = "Normal"
        elif folder_name == "attack":
            true_label = "Attack"
        else:
            raise ValueError(f"Cannot determine true label from folder name: {folder_name}")
        original_labels.append(true_label)

        ig_pixels = get_ig_pixels(model, input_tensor, top_k=NUM_PIXELS)
        original_msg = generate_random_message()
        msg_bits = string_to_bits(original_msg)
        coord_bits = coords_to_bits(ig_pixels)

        msg_encoded_img = embed_lsb(image.copy(), msg_bits, ig_pixels)
        final_stego = embed_metadata(msg_encoded_img, coord_bits)

        cnn_input = np.expand_dims(final_stego.astype(np.float32) / 255.0, axis=(0, -1))
        prob_stego = float(model.predict(cnn_input, verbose=0)[0])
        pred_class_after = CLASS_NAMES[int(prob_stego >= 0.5)]
        stego_preds.append(pred_class_after)

        decoded_msg, decoded_coords = decode_message_and_coords(final_stego)
        original_clean = clean_message(original_msg)
        decoded_clean = clean_message(decoded_msg)

        print(f"Encoded Message: {original_clean}")
        print(f"Decoded Message: {decoded_clean}")
        print(f"Messages Match: {'Matched' if original_clean == decoded_clean else 'Mismatched'}\n")

        mse_value = calculate_mse(image, final_stego)
        psnr_value = calculate_psnr(mse_value)
        ssim_value = calculate_ssim(image, final_stego)
        mean_dct_diff, max_dct_diff = compute_dct_difference(image, final_stego)

        stego_image_filename = f"stego_{filename}"
        stego_image_path = os.path.join(stego_output_folder, stego_image_filename)
        cv2.imwrite(stego_image_path, final_stego)
        print(f"Stego image saved to {stego_image_path}\n")

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "CNN Predicted Class": pred_class_after,
            "Confidence": round(prob_stego if pred_class_after == "Attack" else 1 - prob_stego, 4),
            "Encoded Coords": str(ig_pixels),
            "Decoded Coords Match": "Matched" if ig_pixels == decoded_coords else "Mismatched",
            "True Label": true_label,
            "MSE": round(mse_value, 4),
            "PSNR": round(psnr_value, 2),
            "SSIM": round(ssim_value, 4),
            "Mean DCT Diff": round(mean_dct_diff, 4),
            "Max DCT Diff": round(max_dct_diff, 4)
        })

    df = pd.DataFrame(results)

    cm_original = confusion_matrix(original_labels, original_preds, labels=CLASS_NAMES)
    cm_stego = confusion_matrix(original_labels, stego_preds, labels=CLASS_NAMES)

    output_dir = os.path.dirname(output_excel)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_original, annot=True, fmt='d', cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix BEFORE Embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_original.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_stego, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix AFTER Embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_stego.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Match")
    plt.title("Message Match Count")
    plt.ylabel("Number of Images")
    plt.xlabel("Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_match_count.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Decoded Coords Match")
    plt.title("Coordinate Match Count")
    plt.ylabel("Number of Images")
    plt.xlabel("Coordinate Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coordinate_match_count.png"))
    plt.close()

    summary_df = pd.DataFrame({
        "Average MSE": [df["MSE"].mean()],
        "Average PSNR (dB)": [df["PSNR"].mean()],
        "Average SSIM": [df["SSIM"].mean()],
        "Average Mean DCT Diff": [df["Mean DCT Diff"].mean()],
        "Average Max DCT Diff": [df["Max DCT Diff"].mean()],
        "Message Match Rate (%)": [(df["Match"] == "Matched").mean() * 100],
        "Coordinate Match Rate (%)": [(df["Decoded Coords Match"] == "Matched").mean() * 100],
        "CNN Accuracy After Embedding (%)": [(df["True Label"] == df["CNN Predicted Class"]).mean() * 100]
    })

    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name='Per Image Results')
        summary_df.to_excel(writer, index=False, sheet_name='Summary Averages')

    print("\n✅ Done! Results saved to", output_excel)
    print("\n📊 AVERAGE METRICS ACROSS ALL IMAGES:")
    for col in summary_df.columns:
        print(f"{col}: {summary_df[col].iloc[0]:.4f}" if isinstance(summary_df[col].iloc[0], float) else f"{col}: {summary_df[col].iloc[0]}")

# === RUN === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/CNN FDIA 1/Normal"
    model_path = "Layered IG LSB/CNN FDIA 1/final_cnn_model_grayscale.h5"
    output_excel = "Layered IG LSB/CNN FDIA 1/Final Results (Normal)/ig_layered_stego_results.xlsx"
    stego_output_folder = "Layered IG LSB/CNN FDIA 1/Final Results (Normal)/stego_images"

    if not os.path.exists(stego_output_folder):
        os.makedirs(stego_output_folder)

    process_folder(input_folder, model_path, output_excel, stego_output_folder)


