import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import joblib  # for loading the RF model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib as mpl

# Set font to support Unicode characters like check marks
mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (128, 128)
NUM_PIXELS = 40
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2  # 1280 bits for coordinate encoding
MESSAGE_BITS = NUM_PIXELS  # One bit per pixel
CHANNELS = 1  # Grayscale
CLASS_NAMES = ['Normal', 'Attack']  # Update as needed

# === MESSAGE UTILS === #
def generate_random_message(bits=40):
    chars = bits // 8
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

# === DECODING === #
def decode_message_and_coords(image):
    flat = image.flatten()
    coord_bits = [flat[i] & 1 for i in range(RESERVED_BITS)]
    coords = bits_to_coords(coord_bits)

    msg_bits = []
    for y, x in coords[:NUM_PIXELS]:
        idx = y * image.shape[1] + x
        msg_bits.append(flat[idx] & 1)

    message = bits_to_string(msg_bits)
    return message, coords

# === IG-LIKE PIXEL SELECTION (Random sampling for RF placeholder) === #
def get_random_pixels(image, top_k=NUM_PIXELS):
    h, w = image.shape
    coords = set()
    while len(coords) < top_k:
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        coords.add((y, x))
    return list(coords)

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, rf_model_path, output_excel):
    model = joblib.load(rf_model_path)
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

        # Step 1: Predict class using RF
        flat_img = image.flatten().reshape(1, -1)
        pred_class_idx = model.predict(flat_img)[0]
        pred_probs = model.predict_proba(flat_img)[0]
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(pred_probs[pred_class_idx])

        # Step 2: Get pixels for embedding (using random selection for RF)
        selected_pixels = get_random_pixels(image, top_k=NUM_PIXELS)

        # Step 3: Generate message and embed
        original_msg = generate_random_message()
        msg_bits = string_to_bits(original_msg)
        coord_bits = coords_to_bits(selected_pixels)
        msg_encoded_img = embed_lsb(image.copy(), msg_bits, selected_pixels)

        # Step 4: Embed coordinates
        final_stego = embed_metadata(msg_encoded_img, coord_bits)

        # Step 5: Decode both message and coordinates
        decoded_msg, decoded_coords = decode_message_and_coords(final_stego)

        # Step 6: Compare decoded results
        original_clean = clean_excel_string(original_msg)
        decoded_clean = clean_excel_string(decoded_msg)

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "RF Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
            "Encoded Coords": str(selected_pixels),
            "Decoded Coords Match": "Matched" if selected_pixels == decoded_coords else "Mismatched"
        })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Done! Results saved to {output_excel}")

    # === Create plots === #
    sns.set(style="whitegrid")
    df["True Label"] = df["Image"].apply(lambda x: "Normal" if "Normal" in x else "Attack")
    cm = confusion_matrix(df["True Label"], df["RF Predicted Class"], labels=["Normal", "Attack"])

    output_dir = os.path.dirname(output_excel)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Message Match
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Match")
    plt.title("Message Match Count")
    plt.ylabel("Number of Images")
    plt.xlabel("Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_match_count.png"))
    plt.close()

    # Coordinate Match
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Decoded Coords Match")
    plt.title("Coordinate Match Count")
    plt.ylabel("Number of Images")
    plt.xlabel("Coordinate Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coordinate_match_count.png"))
    plt.close()

# === RUN SCRIPT === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/RF FDIA 1/Normal (FDIA 1)"
    rf_model_path = "Layered IG LSB/RF FDIA 1/rf_model.pkl"
    output_excel = "Layered IG LSB/RF FDIA 1/rf_layered_stego_results.xlsx"
    process_folder(input_folder, rf_model_path, output_excel)
