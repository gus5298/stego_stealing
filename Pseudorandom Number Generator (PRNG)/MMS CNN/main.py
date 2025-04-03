import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import hashlib
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct
import math

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (64, 64)
NUM_PIXELS = 24
MESSAGE_BITS = NUM_PIXELS
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Faulty', 'Attack']
SEED_MODE = 'hash'
SEED_RESERVED_PIXELS = 150
SEED_SECRET_KEY = "my_shared_passphrase"

# === MESSAGE UTILS === #
def generate_random_message(chars=3):
    return ''.join(random.choices(string.ascii_letters[:26] + string.ascii_letters[26:], k=chars))

def string_to_bits(s):
    return [int(b) for c in s.encode('ascii') for b in format(c, '08b')][:MESSAGE_BITS]

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
    used_indices = set()
    for i, (y, x) in enumerate(coords):
        if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            continue
        idx = y * image.shape[1] + x
        if idx < SEED_RESERVED_PIXELS:
            continue
        flat[idx] = (flat[idx] & ~1) | bits[i]
        used_indices.add(idx)
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

def get_random_seed():
    return random.randint(0, 2**32 - 1)

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

# === CNN FEATURE EXTRACTION === #
def extract_features(image):
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# === DECODING === #
def decode_message(image, coords):
    flat = image.flatten()
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]
    return bits_to_string(msg_bits)

# === CNN ANALYSIS === #
def analyze_with_cnn(stego_image, model):
    cnn_input = extract_features(stego_image)
    prediction = model.predict(cnn_input, verbose=0)[0]
    class_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[class_index]
    confidence = float(prediction[class_index])
    print(f"CNN Prediction: {predicted_class} (Confidence: {confidence:.4f})")
    return predicted_class, confidence

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
def process_folder(input_folder, model_path, stego_output_folder, results_output_folder, seed_secret_key):
    model = load_model(model_path)
    results = []

    if not os.path.exists(stego_output_folder):
        os.makedirs(stego_output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, filename in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] Processing: {filename}")
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, IMG_SIZE)
        image = image.astype(np.uint8)

        seed_value = get_seed_from_image(seed_secret_key, image)
        seed_bits = [int(b) for b in format(seed_value, '032b')]

        stego_img, seed_coords = embed_seed_value(image.copy(), seed_bits, seed_secret_key)
        extracted_seed = extract_seed_value(stego_img, seed_secret_key, coords=seed_coords)
        key_match_status = "Matched" if extracted_seed == seed_value else "Mismatched"

        prng_pixels = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value)
        original_msg = generate_random_message(chars=3)
        msg_bits = string_to_bits(original_msg)
        final_stego = embed_lsb(stego_img.copy(), msg_bits, prng_pixels)

        decoded_msg = decode_message(final_stego, prng_pixels)
        msg_match_status = "Matched" if clean_excel_string(decoded_msg) == clean_excel_string(original_msg) else "Mismatched"

        pred_class, confidence = analyze_with_cnn(final_stego, model)

        stego_image_filename = f"stego_{filename}"
        stego_image_path = os.path.join(stego_output_folder, stego_image_filename)
        cv2.imwrite(stego_image_path, final_stego)

        mse_value = calculate_mse(image, final_stego)
        psnr_value = calculate_psnr(mse_value)
        ssim_value = calculate_ssim(image, final_stego)
        mean_dct_diff, max_dct_diff = compute_dct_difference(image, final_stego)

        results.append({
            "Image": filename,
            "Original Message": original_msg,
            "Decoded Message": decoded_msg,
            "Message Match": msg_match_status,
            "Seed": seed_value,
            "Extracted Seed": extracted_seed,
            "Key Match": key_match_status,
            "CNN Predicted Class": pred_class,
            "Confidence": confidence,
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

    if "True Label" not in df.columns:
        df["True Label"] = "Normal"

    df_filtered = df[df["Image"] != "AVERAGE"]
    cm = confusion_matrix(df_filtered["True Label"], df_filtered["CNN Predicted Class"], labels=CLASS_NAMES)

    output_excel = os.path.join(results_output_folder, "stego_analysis_results_with_metrics.xlsx")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_filtered.to_excel(writer, index=False, sheet_name="Per Image Results")
        pd.DataFrame([avg_metrics]).to_excel(writer, index=False, sheet_name="Summary Averages")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Stego Images")
    plt.tight_layout()
    plt.savefig(os.path.join(results_output_folder, "stego_confusion_matrix.png"))
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

# === RUN === #
input_folder = "PRNG/MMS/MMS Images/Normal"
model_path = "PRNG/MMS/MMS Images/cnn_3_class_grayscale_model_64x64.h5"
stego_output_folder = "PRNG/MMS/Double PRNG/stego_images"
results_output_folder = "PRNG/MMS/Double PRNG"
seed_secret_key = "Gustavo_Sanchez"

process_folder(input_folder, model_path, stego_output_folder, results_output_folder, seed_secret_key)
