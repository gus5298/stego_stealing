import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import joblib  # For loading Random Forest model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (128, 128)  # Updated to 128x128 for the new Random Forest model
NUM_PIXELS = 24  # Number of PRNG selected pixels for embedding
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2  # Coordinate storage size
MESSAGE_BITS = NUM_PIXELS  # 24 bits for message (3 characters)
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Attack']  # Updated for your RF model's classes (Normal and Attack)

# === MESSAGE UTILS === #
def generate_random_message(chars=3):
    # Generate a 3-character message with only alphabetic letters
    return ''.join(random.choices(string.ascii_letters, k=chars))

def string_to_bits(s):
    # Convert the message into bits (24 bits for 3 characters)
    return [int(b) for c in s.encode('utf-8') for b in format(c, '08b')][:MESSAGE_BITS]

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def clean_excel_string(s):
    # Clean the string to include only letters (no spaces, no numbers)
    return ''.join(c for c in s if c.isalpha())

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
        flat[i] = (flat[i] & ~1) | meta_bits[i]
    return flat.reshape(image.shape)

# === DECODING === #
def decode_message_and_coords(image):
    flat = image.flatten()
    coord_bits = [flat[i] & 1 for i in range(RESERVED_BITS)]
    coords = bits_to_coords(coord_bits)
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]
    return bits_to_string(msg_bits), coords

# === PRNG-BASED PIXEL SELECTION === #
def generate_prng_pixel_positions(image_shape, count, seed_value):
    np.random.seed(seed_value)
    h, w = image_shape
    indices = np.random.choice(h * w, size=count, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    return list(zip(ys, xs))

# === FEATURE EXTRACTION FOR RF === #
def extract_features(image):
    # Flatten image to a feature vector (you can add more advanced features here)
    return image.flatten()

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel):
    # Load Random Forest model (instead of CNN)
    model = joblib.load(model_path)  # Assuming the RF model is saved as a .pkl file using joblib
    results = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for idx, filename in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] Processing {filename}")
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Could not read {filename}. Skipping.")
            continue

        image = cv2.resize(image, IMG_SIZE)
        
        # Feature extraction for Random Forest
        features = extract_features(image)

        # Predict with Random Forest
        pred_class_idx = model.predict([features])[0]  # Random Forest predicts directly
        confidence = model.predict_proba([features])[0][pred_class_idx]  # Confidence score

        # Map pred_class_idx back to class name
        pred_class = CLASS_NAMES[pred_class_idx]

        prng_pixels = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value=12345 + idx)

        # Generate a 24-bit message
        original_msg = generate_random_message(chars=MESSAGE_BITS // 8)

        # Convert message to bits
        msg_bits = string_to_bits(original_msg)

        # Convert PRNG-selected coordinates to bits
        coord_bits = coords_to_bits(prng_pixels)

        # Embed the message in the image using LSB
        msg_encoded_img = embed_lsb(image.copy(), msg_bits, prng_pixels)

        # Embed the coordinates as metadata
        final_stego = embed_metadata(msg_encoded_img, coord_bits)

        # Decode the message and coordinates
        decoded_msg, decoded_coords = decode_message_and_coords(final_stego)

        # Clean up the message for the Excel file
        original_clean = clean_excel_string(original_msg)
        decoded_clean = clean_excel_string(decoded_msg)

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "RF Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
            "Encoded Coords": str(prng_pixels),
            "Decoded Coords Match": "Matched" if prng_pixels == decoded_coords else "Mismatched"
        })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Done! Results saved to {output_excel}")

    # === Visualization === #
    sns.set(style="whitegrid")
    df["True Label"] = "Normal"  # Assuming the true labels are always "Normal"
    cm = confusion_matrix(df["True Label"], df["RF Predicted Class"], labels=CLASS_NAMES)

    output_dir = os.path.dirname(output_excel)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
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


if __name__ == "__main__":
    input_folder = "PRNG/FDIA 1/RF/Images FDIA 1/Normal"
    model_path = "PRNG/FDIA 1/RF/rf_model.pkl"  # Path to your Random Forest model
    output_excel = "PRNG/FDIA 1/RF/ig_layered_stego_results_rf.xlsx"
    process_folder(input_folder, model_path, output_excel)
