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
IMG_SIZE = (64, 64)
BITS_PER_COORD = 16
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Attack', 'Faulty']

# === MESSAGE UTILS === #
def generate_random_message(k):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=k))

def string_to_bits(s):
    return [int(b) for c in s.encode('utf-8') for b in format(c, '08b')]

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
def get_ig_pixels(model, image_tensor, label_index, top_k):
    def loss_fn(output): return output[:, label_index]
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
def decode_message_and_coords(image, num_pixels):
    flat = image.flatten()
    reserved_bits = num_pixels * BITS_PER_COORD * 2
    coord_bits = [(flat[i] >> 7) & 1 for i in range(reserved_bits)]
    coords = bits_to_coords(coord_bits)
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:num_pixels]]
    return bits_to_string(msg_bits), coords

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

# === VISUALIZATION === #
def save_pixel_overlay(image, coords, save_path):
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in coords:
        cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite(save_path, vis_image)

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel, stego_output_folder, embed_visuals_folder, num_letters):
    message_bits = num_letters * 8
    num_pixels = message_bits  # 1 bit per pixel
    reserved_bits = num_pixels * BITS_PER_COORD * 2

    model = load_model(model_path)
    results = []
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    matched_count = 0
    mismatched_count = 0

    original_labels = []
    original_preds = []

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

        pred_probs = model.predict(input_tensor, verbose=0)[0]
        label_index = int(np.argmax(pred_probs))
        true_label = os.path.basename(input_folder)
        if true_label not in CLASS_NAMES:
            true_label = 'Unknown'

        original_labels.append(true_label)
        original_preds.append(CLASS_NAMES[label_index])

        ig_pixels = get_ig_pixels(model, input_tensor, label_index, top_k=num_pixels)

        original_msg = generate_random_message(num_letters)
        msg_bits = string_to_bits(original_msg)[:message_bits]
        coord_bits = coords_to_bits(ig_pixels)

        msg_encoded_img = embed_lsb(image.copy(), msg_bits, ig_pixels)
        cnn_input = np.expand_dims(msg_encoded_img.astype(np.float32) / 255.0, axis=(0, -1))
        pred_probs_after_lsb = model.predict(cnn_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(pred_probs_after_lsb))
        pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else 'Unknown'
        confidence = float(pred_probs_after_lsb[pred_class_idx])

        final_stego = embed_metadata(msg_encoded_img, coord_bits)
        stego_image_filename = f"stego_{filename}"
        stego_image_path = os.path.join(stego_output_folder, stego_image_filename)
        cv2.imwrite(stego_image_path, final_stego)
        print(f"Stego image saved to {stego_image_path}")

        visual_filename = f"embed_vis_{filename}"
        visual_path = os.path.join(embed_visuals_folder, visual_filename)
        save_pixel_overlay(image, ig_pixels, visual_path)
        print(f"Visualization saved to {visual_path}")

        decoded_msg, decoded_coords = decode_message_and_coords(final_stego, num_pixels)
        original_clean = clean_message(original_msg)
        decoded_clean = clean_message(decoded_msg)

        print(f"Encoded Message: {original_clean}")
        print(f"Decoded Message: {decoded_clean}")
        print(f"Messages Match: {'Matched' if original_clean == decoded_clean else 'Mismatched'}\n")

        if original_clean == decoded_clean:
            matched_count += 1
        else:
            mismatched_count += 1

        mse_value = calculate_mse(image, final_stego)
        psnr_value = calculate_psnr(mse_value)
        ssim_value = calculate_ssim(image, final_stego)
        mean_dct_diff, max_dct_diff = compute_dct_difference(image, final_stego)

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "CNN Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
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

    # === Visualization === #
    output_dir = os.path.dirname(output_excel)

    sns.set(style="whitegrid")
    cm_original = confusion_matrix(original_labels, original_preds, labels=CLASS_NAMES)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_original, annot=True, fmt='d', cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix BEFORE Embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_original.png"))
    plt.close()

    cm = confusion_matrix(df["True Label"], df["CNN Predicted Class"], labels=CLASS_NAMES)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix AFTER Embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
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


# === Run === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/CNN MMS/Normal"
    model_path = "Layered IG LSB/CNN MMS/cnn_best_model.h5"
    output_excel = "Layered IG LSB/CNN MMS/Custom Message Encoder/Normal/ig_layered_stego_results (5 letters).xlsx"

    # Ask user how many letters to encode
    num_letters = int(input("🔤 How many letters do you want to encode per image? "))

    # Output folders
    stego_output_folder = "Layered IG LSB/CNN MMS/Custom Message Encoder/Normal/stego_images (5 letters)"
    embed_visuals_folder = "Layered IG LSB/CNN MMS/Custom Message Encoder/Normal/embed_visuals (5 letters)"

    os.makedirs(stego_output_folder, exist_ok=True)
    os.makedirs(embed_visuals_folder, exist_ok=True)

    process_folder(input_folder, model_path, output_excel, stego_output_folder, embed_visuals_folder, num_letters)
