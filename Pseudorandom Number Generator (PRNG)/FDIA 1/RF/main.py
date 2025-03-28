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
import hashlib

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (128, 128)
NUM_PIXELS = 24
MESSAGE_BITS = NUM_PIXELS
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Attack']
SEED_MODE = 'hash'  # Options: 'hash' or 'random'
RESERVED_PIXELS = 800

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

# === EMBEDDING === #
def embed_lsb(image, bits, coords):
    flat = image.flatten()
    used_indices = set()
    for i, (y, x) in enumerate(coords):
        if not (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
            print(f"⚠️ Invalid coordinate: ({y}, {x})")
            continue
        idx = y * image.shape[1] + x
        if idx < RESERVED_PIXELS:
            print(f"⚠️ Skipping reserved pixel at index {idx} (coord: {y}, {x})")
            continue
        flat[idx] = (flat[idx] & ~1) | bits[i]
        used_indices.add(idx)
    return flat.reshape(image.shape)

def seed_to_bits(seed_value, bits_per_seed=32):
    return [int(b) for b in format(seed_value, f'0{bits_per_seed}b')]

def embed_seed_value(image, seed_bits):
    flat = image.flatten()
    for i in range(len(seed_bits)):
        flat[i] = (flat[i] & ~1) | seed_bits[i]
    return flat.reshape(image.shape)

def extract_seed_value(image, bits_per_seed=32):
    flat = image.flatten()
    seed_bits = [flat[i] & 1 for i in range(bits_per_seed)]
    seed_value = int(''.join(map(str, seed_bits)), 2)
    return seed_value

# === SEED GENERATORS === #
def get_seed_from_filename(filename):
    hash_digest = hashlib.md5(filename.encode()).hexdigest()
    return int(hash_digest, 16) % (2**32)

def get_random_seed():
    return random.randint(0, 2**32 - 1)

# === PRNG PIXEL SELECTION === #
def generate_prng_pixel_positions(image_shape, count, seed_value):
    np.random.seed(seed_value)
    h, w = image_shape
    total_pixels = h * w
    excluded_pixels = set(range(RESERVED_PIXELS))
    valid_indices = list(set(range(total_pixels)) - excluded_pixels)
    indices = np.random.choice(valid_indices, size=count, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    return list(zip(ys, xs))

# === FEATURE EXTRACTION FOR RF === #
def extract_features(image):
    return image.flatten().reshape(1, -1)  # Flatten the image into a feature vector

# === DECODING === #
def decode_message(image, coords):
    flat = image.flatten()
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]
    print(f"Decoded Message Bits: {msg_bits}")
    return bits_to_string(msg_bits)

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel):
    # Load the Random Forest model
    model = joblib.load(model_path)
    results = []
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize counters for matched and mismatched messages
    matched_count = 0
    mismatched_count = 0

    for idx, filename in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] Processing {filename}")
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"⚠️ Could not read {filename}. Skipping.")
            continue

        image = cv2.resize(image, IMG_SIZE)
        image = image.astype(np.uint8)

        if SEED_MODE == 'hash':
            seed_value = get_seed_from_filename(filename)
        elif SEED_MODE == 'random':
            seed_value = get_random_seed()
        else:
            raise ValueError("Invalid SEED_MODE. Choose 'hash' or 'random'.")

        print(f"Encoded Seed: {seed_value}")

        seed_bits = seed_to_bits(seed_value)
        stego_img = embed_seed_value(image.copy(), seed_bits)

        prng_pixels = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value=seed_value)
        original_msg = generate_random_message(chars=MESSAGE_BITS // 8)
        msg_bits = string_to_bits(original_msg)
        print(f"Original Message Bits: {msg_bits}")

        final_stego = embed_lsb(stego_img.copy(), msg_bits, prng_pixels)

        # === RF Prediction BEFORE decoding === #
        rf_features = extract_features(final_stego)
        pred_class_idx = model.predict(rf_features)[0]  # Predict using the RF model
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = model.predict_proba(rf_features)[0][pred_class_idx]

        # === Now Decode Message AFTER RF Prediction === #
        decoded_seed_value = extract_seed_value(final_stego)
        print(f"Decoded Seed: {decoded_seed_value}")

        prng_pixels_decoded = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value=decoded_seed_value)
        decoded_msg = decode_message(final_stego, prng_pixels_decoded)

        original_clean = clean_excel_string(original_msg)
        decoded_clean = clean_excel_string(decoded_msg)

        match_status = "Matched" if original_clean == decoded_clean else "Mismatched"
        print(f"Original Message: {original_clean} | Decoded Message: {decoded_clean} => {match_status}")

        # Increment the counters based on whether the messages match
        if match_status == "Matched":
            matched_count += 1
        else:
            mismatched_count += 1

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": match_status,
            "RF Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
            "Seed": seed_value
        })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Done! Results saved to {output_excel}")

    # Print total matched and mismatched messages
    print(f"\nTotal Matched Messages: {matched_count}")
    print(f"Total Mismatched Messages: {mismatched_count}")

    # === Visualization === #
    sns.set(style="whitegrid")
    df["True Label"] = "Normal"
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



# === Run === #
if __name__ == "__main__":
    input_folder = "PRNG/FDIA 1/RF/Images FDIA 1/Normal"
    model_path = "PRNG/FDIA 1/RF/rf_model.pkl"  # Path to your Random Forest model
    output_excel = "PRNG/FDIA 1/RF/ig_layered_stego_results_rf_with_seed.xlsx"
    process_folder(input_folder, model_path, output_excel)
