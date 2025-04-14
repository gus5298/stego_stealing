import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import hashlib
import joblib
import math
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (128, 128)
CHANNELS = 1
CLASS_NAMES = None
SEED_RESERVED_PIXELS = 300
SEED_SECRET_KEY = "my_shared_passphrase"

# === DYNAMIC USER INPUT === #
num_chars_to_embed = int(input("Enter number of characters to embed in each image (e.g., 3): "))
MESSAGE_BITS = num_chars_to_embed * 8
NUM_PIXELS = MESSAGE_BITS
print(f"Embedding {num_chars_to_embed} characters → {MESSAGE_BITS} bits")

# === MESSAGE UTILS === #
def generate_random_message(chars):
    return ''.join(random.choices(string.ascii_letters, k=chars))

def string_to_bits(s):
    return [int(b) for c in s.encode('ascii') for b in format(c, '08b')]

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def clean_excel_string(s):
    return ''.join(c for c in s if c.isalpha())

# === PRNG SEED-HIDING BASED ON IMAGE CONTENT + SHARED KEY === #
def get_seed_hiding_key(secret_key: str, image: np.ndarray):
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    combined = secret_key + image_hash
    digest = hashlib.sha256(combined.encode()).hexdigest()
    return int(digest, 16) % (2**32)

# === EMBEDDING === #
def embed_lsb(image, bits, coords):
    flat = image.flatten()
    for i, (y, x) in enumerate(coords):
        if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            continue
        idx = y * image.shape[1] + x
        if idx < SEED_RESERVED_PIXELS:
            continue
        flat[idx] = (flat[idx] & ~1) | bits[i]
    return flat.reshape(image.shape)

# === PRNG Seed Embedding === #
def get_seed_embedding_coords(image, total_reserved=SEED_RESERVED_PIXELS, bits=32, seed_secret_key=SEED_SECRET_KEY):
    seed_key = get_seed_hiding_key(seed_secret_key, image)
    np.random.seed(seed_key)
    coords = np.random.choice(range(total_reserved), size=bits, replace=False)
    return coords

def embed_seed_value(image, seed_bits, seed_secret_key, coords=None):
    if coords is None:
        coords = get_seed_embedding_coords(image.copy(), seed_secret_key=seed_secret_key)
    flat = image.flatten()
    for i, idx in enumerate(coords):
        flat[idx] = (flat[idx] & ~1) | seed_bits[i]
    return flat.reshape(image.shape), coords

def extract_seed_value(image, seed_secret_key, coords=None):
    if coords is None:
        coords = get_seed_embedding_coords(image.copy(), seed_secret_key=seed_secret_key)
    flat = image.flatten()
    seed_bits = [flat[idx] & 1 for idx in coords]
    seed_value = int(''.join(map(str, seed_bits)), 2)
    return seed_value

# === SEED GENERATORS === #
def get_seed_from_image(secret_key: str, image: np.ndarray):
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    combined = secret_key + image_hash + "_main_seed"
    digest = hashlib.sha256(combined.encode()).hexdigest()
    return int(digest, 16) % (2**32)

# === PRNG PIXEL SELECTION === #
def generate_prng_pixel_positions(image_shape, count, seed_value):
    np.random.seed(seed_value)
    h, w = image_shape
    total_pixels = h * w
    excluded_pixels = set(range(SEED_RESERVED_PIXELS))
    valid_indices = list(set(range(total_pixels)) - excluded_pixels)
    indices = np.random.choice(valid_indices, size=count, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    return list(zip(ys, xs))

# === RF FEATURE EXTRACTION === #
def extract_rf_features(image):
    return (image.flatten() / 255.0)

# === RF ANALYSIS === #
def analyze_with_rf(image, rf_model):
    features = extract_rf_features(image).reshape(1, -1)
    prediction = rf_model.predict(features)[0]
    proba = rf_model.predict_proba(features)[0]
    confidence = max(proba)
    return prediction, confidence

# === IMAGE QUALITY METRICS === #
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
def mark_embedded_pixels(image, coords, output_path):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in coords:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            color_img[y, x] = [0, 0, 255]  # Red in BGR
    cv2.imwrite(output_path, color_img)

# === DECODING === #
def decode_message(image, coords, num_bits):
    flat = image.flatten()
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:num_bits]]
    return bits_to_string(msg_bits)

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, stego_output_folder, results_output_folder, seed_secret_key):
    global CLASS_NAMES
    rf_model = joblib.load(model_path)
    CLASS_NAMES = list(rf_model.classes_)
    results = []

    if not os.path.exists(stego_output_folder):
        os.makedirs(stego_output_folder)

    visualization_folder = os.path.join(results_output_folder, "visualizations")
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)

    image_files = []
    true_labels = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_files.append(image_path)
                label = os.path.basename(root).capitalize()
                true_labels.append(label)

    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        true_label = true_labels[i]

        print(f"Processing: {filename} | True Label: {true_label}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, IMG_SIZE)
        image = image.astype(np.uint8)

        original_pred_class, original_confidence = analyze_with_rf(image, rf_model)

        seed_value = get_seed_from_image(seed_secret_key, image)
        seed_bits = [int(b) for b in format(seed_value, '032b')]

        stego_img, seed_coords = embed_seed_value(image.copy(), seed_bits, seed_secret_key)
        extracted_seed = extract_seed_value(stego_img, seed_secret_key, coords=seed_coords)
        key_match_status = "Matched" if extracted_seed == seed_value else "Mismatched"

        prng_pixels = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value)
        original_msg = generate_random_message(num_chars_to_embed)
        msg_bits = string_to_bits(original_msg)
        final_stego = embed_lsb(stego_img.copy(), msg_bits, prng_pixels)

        decoded_msg = decode_message(final_stego, prng_pixels, MESSAGE_BITS)
        msg_match_status = "Matched" if clean_excel_string(decoded_msg) == clean_excel_string(original_msg) else "Mismatched"

        pred_class, confidence = analyze_with_rf(final_stego, rf_model)

        stego_image_filename = f"stego_{filename}"
        stego_image_path = os.path.join(stego_output_folder, stego_image_filename)
        cv2.imwrite(stego_image_path, final_stego)

        visualization_path = os.path.join(visualization_folder, f"viz_{filename}")
        mark_embedded_pixels(final_stego, prng_pixels, visualization_path)

        mse_value = calculate_mse(image, final_stego)
        psnr_value = calculate_psnr(mse_value)
        ssim_value = calculate_ssim(image, final_stego)
        mean_dct_diff, max_dct_diff = compute_dct_difference(image, final_stego)

        results.append({
            "Image": filename,
            "True Label": true_label,
            "Original Message": original_msg,
            "Decoded Message": decoded_msg,
            "Message Match": msg_match_status,
            "Seed": seed_value,
            "Extracted Seed": extracted_seed,
            "Key Match": key_match_status,
            "RF Predicted Class": pred_class,
            "RF Predicted Class (Original)": original_pred_class,
            "Confidence": confidence,
            "Confidence (Original)": original_confidence,
            "Stego Image Path": stego_image_path,
            "MSE": round(mse_value, 4),
            "PSNR": round(psnr_value, 2),
            "SSIM": round(ssim_value, 4),
            "Mean DCT Diff": round(mean_dct_diff, 4),
            "Max DCT Diff": round(max_dct_diff, 4)
        })

    df = pd.DataFrame(results)
    avg_metrics = {
        "Image": "AVERAGE",
        "MSE": df["MSE"].mean(),
        "PSNR": df["PSNR"].mean(),
        "SSIM": df["SSIM"].mean(),
        "Mean DCT Diff": df["Mean DCT Diff"].mean(),
        "Max DCT Diff": df["Max DCT Diff"].mean()
    }

    df_filtered = df[df["Image"] != "AVERAGE"]
    cm_stego = confusion_matrix(df_filtered["True Label"], df_filtered["RF Predicted Class"], labels=CLASS_NAMES)
    cm_original = confusion_matrix(df_filtered["True Label"], df_filtered["RF Predicted Class (Original)"], labels=CLASS_NAMES)

    output_excel = os.path.join(results_output_folder, "stego_analysis_results_with_metrics.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="Per Image Results")
        pd.DataFrame([avg_metrics]).to_excel(writer, index=False, sheet_name="Summary Averages")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_stego, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Stego Images")
    plt.tight_layout()
    plt.savefig(os.path.join(results_output_folder, "stego_confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_original, annot=True, fmt='d', cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Original Images")
    plt.tight_layout()
    plt.savefig(os.path.join(results_output_folder, "original_confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Key Match")
    plt.title("Seed Match Count (Initial PRNG)")
    plt.ylabel("Number of Images")
    plt.xlabel("Key Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(results_output_folder, "initial_prng_key_match_count.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Message Match")
    plt.title("Message Match Count (Second PRNG)")
    plt.ylabel("Number of Images")
    plt.xlabel("Message Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(results_output_folder, "second_prng_message_match_count.png"))
    plt.close()

    print(f"\u2705 All results saved to: {results_output_folder}")

# === INPUT PATHS === #
input_folder = "PRNG/FDIA 1/RF/Images FDIA 1/Attack"
model_path = "PRNG/FDIA 1/RF/best_rf_model.pkl"
stego_output_folder = "PRNG/FDIA 1/RF/Custom Message Encoder/Attack/stego_images (600 letters)"
results_output_folder = "PRNG/FDIA 1/RF/Custom Message Encoder/Attack/Final Results (600 letters)"
seed_secret_key = "Gustavo_Sanchez"

process_folder(input_folder, model_path, stego_output_folder, results_output_folder, seed_secret_key)
