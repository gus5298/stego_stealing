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
from sklearn.ensemble import RandomForestClassifier  # Use for loading RF model
import joblib  # Used to load the Random Forest model (saved as .pkl file)

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (128, 128)  # Updated to 128x128 for the new RF model (grayscale)
NUM_PIXELS = 24  # Number of PRNG selected pixels for embedding
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2  # Coordinate storage size
MESSAGE_BITS = NUM_PIXELS  # 24 bits for message (3 characters)
CHANNELS = 1  # Grayscale image has 1 channel
CLASS_NAMES = ['Normal', 'Attack']  # Updated for the RF model's classes (Normal and Attack)

# === FIXED SEED === #
FIXED_SEED = 12345  # A fixed seed value that will be used for all images

# === MESSAGE UTILS === #
def generate_random_message(chars=3):
    return ''.join(random.choices(string.ascii_letters, k=chars))

def string_to_bits(s):
    return [int(b) for c in s.encode('utf-8') for b in format(c, '08b')][:MESSAGE_BITS]

def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) == 8:
            chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

def clean_excel_string(s):
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

# === EMBEDDING SEED VALUE === #
def seed_to_bits(seed_value, bits_per_seed=32):
    return [int(b) for b in format(seed_value, f'0{bits_per_seed}b')]

def embed_seed_value(image, seed_bits):
    flat = image.flatten()
    # Debugging: Print seed bits before embedding
    print(f"Seed bits before embedding: {seed_bits}")
    # Embed the seed into the first 32 bits of the image
    for i in range(len(seed_bits)):
        flat[i] = (flat[i] & ~1) | seed_bits[i]
    return flat.reshape(image.shape)

# === EXTRACTING SEED VALUE === #
def extract_seed_value(image, bits_per_seed=32):
    flat = image.flatten()
    # Extract the first 32 bits for the seed
    seed_bits = [flat[i] & 1 for i in range(bits_per_seed)]
    
    # Debugging: Print the extracted seed bits to verify
    print(f"Extracted seed bits: {seed_bits[:32]}")  # Only print the first 32 bits

    seed_value = int(''.join(map(str, seed_bits)), 2)

    # Debugging: Print the extracted seed value
    print(f"Extracted seed value: {seed_value}")
    return seed_value

# === PRNG-BASED PIXEL SELECTION === #
def generate_prng_pixel_positions(image_shape, count, seed_value):
    np.random.seed(seed_value)
    h, w = image_shape
    total_pixels = h * w

    # Exclude the first 128 pixels from the PRNG location generation
    excluded_pixels = set(range(128))  # Exclude the first 128 pixels

    # Generate valid indices, excluding the first 128 pixels
    valid_indices = list(set(range(total_pixels)) - excluded_pixels)

    # Randomly select locations from the valid indices
    indices = np.random.choice(valid_indices, size=count, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    return list(zip(ys, xs))

# === FEATURE EXTRACTION FOR RF === #
def extract_features(image):
    # Flatten the image to a 1D vector (suitable for Random Forest model)
    return image.flatten().reshape(1, -1)

# === DECODING === #
def decode_message_and_coords(image):
    flat = image.flatten()
    
    # Extract the seed bits
    coord_bits = [flat[i] & 1 for i in range(RESERVED_BITS)]  # Extracting the coordinate bits (for LSB encoding)
    coords = bits_to_coords(coord_bits)  # Convert the coordinate bits back to coordinates
    
    # Now extract the message bits from the PRNG-selected coordinates
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]  # Extract the LSB of the selected pixels
    return bits_to_string(msg_bits), coords

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel):
    # Load the Random Forest model
    model = joblib.load(model_path)  # Use joblib to load .pkl RF model
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
        
        # Use the same fixed seed for all images
        seed_value = FIXED_SEED  # Same seed for all images

        # Convert seed to bits
        seed_bits = seed_to_bits(seed_value)

        # Embed the seed value in the image
        msg_encoded_img_with_seed = embed_seed_value(image.copy(), seed_bits)

        # Generate PRNG locations excluding the first 128 pixels
        prng_pixels = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value=seed_value)

        # Generate a 24-bit message
        original_msg = generate_random_message(chars=MESSAGE_BITS // 8)

        # Convert message to bits
        msg_bits = string_to_bits(original_msg)

        # Convert PRNG-selected coordinates to bits
        coord_bits = coords_to_bits(prng_pixels)

        # Embed the message in the image using LSB
        msg_encoded_img_with_seed = embed_lsb(msg_encoded_img_with_seed, msg_bits, prng_pixels)

        # Embed the coordinates as metadata
        final_stego_with_seed = embed_metadata(msg_encoded_img_with_seed, coord_bits)

        # === Debugging: Check the Seed during Encoding ===
        print(f"Seed used during encoding: {seed_value}")
        print(f"PRNG pixel locations used during encoding: {prng_pixels}")
        print(f"Original message: {original_msg}")

        # Decode the message and coordinates
        decoded_seed_value = extract_seed_value(final_stego_with_seed)
        print(f"Extracted seed during decoding: {decoded_seed_value}")

        prng_pixels_decoded = generate_prng_pixel_positions(image.shape, NUM_PIXELS, seed_value=decoded_seed_value)
        decoded_msg, decoded_coords = decode_message_and_coords(final_stego_with_seed)

        # Clean up the message for the Excel file
        original_clean = clean_excel_string(original_msg)
        decoded_clean = clean_excel_string(decoded_msg)

        # === Debugging: Check the Decoded Message ===
        print(f"Decoded message: {decoded_clean}")
        print(f"Match: {'Matched' if original_clean == decoded_clean else 'Mismatched'}")

        # RF prediction
        rf_input = extract_features(image)
        pred_class_idx = int(model.predict(rf_input)[0])  # RF model predicts the class directly
        pred_class = CLASS_NAMES[pred_class_idx]

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "RF Predicted Class": pred_class,
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


# === Run === #
if __name__ == "__main__":
    input_folder = "PRNG/FDIA 1/RF/Images FDIA 1/Normal"
    model_path = "PRNG/FDIA 1/RF/rf_model.pkl"  # Path to your Random Forest model
    output_excel = "PRNG/FDIA 1/RF/ig_layered_stego_results_rf_with_seed.xlsx"
    process_folder(input_folder, model_path, output_excel)
