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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

mpl.rcParams['font.family'] = 'DejaVu Sans'

# === CONFIGURATION === #
IMG_SIZE = (64, 64)
NUM_PIXELS = 40
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2
MESSAGE_BITS = NUM_PIXELS
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Faulty', 'Attack']

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
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]
    return bits_to_string(msg_bits), coords

# === IG-BASED IMPORTANT PIXELS === #
def get_ig_pixels(model, image_tensor, label_index, top_k=NUM_PIXELS):
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

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel):
    model = load_model(model_path)
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
        norm_img = image.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(norm_img, axis=(0, -1))  # shape: (1, 64, 64, 1)

        pred_probs = model.predict(input_tensor, verbose=0)[0]
        label_index = int(np.argmax(pred_probs))
        ig_pixels = get_ig_pixels(model, input_tensor, label_index, top_k=NUM_PIXELS)

        original_msg = generate_random_message()
        msg_bits = string_to_bits(original_msg)
        coord_bits = coords_to_bits(ig_pixels)
        msg_encoded_img = embed_lsb(image.copy(), msg_bits, ig_pixels)

        cnn_input = np.expand_dims(msg_encoded_img.astype(np.float32) / 255.0, axis=(0, -1))  # shape: (1, 64, 64, 1)
        pred_probs_after_lsb = model.predict(cnn_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(pred_probs_after_lsb))
        pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else 'Unknown'
        confidence = float(pred_probs_after_lsb[pred_class_idx])

        final_stego = embed_metadata(msg_encoded_img, coord_bits)
        decoded_msg, decoded_coords = decode_message_and_coords(final_stego)

        original_clean = clean_excel_string(original_msg)
        decoded_clean = clean_excel_string(decoded_msg)

        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "CNN Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
            "Encoded Coords": str(ig_pixels),
            "Decoded Coords Match": "Matched" if ig_pixels == decoded_coords else "Mismatched"
        })

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Done! Results saved to {output_excel}")

    # === Visualization === #
    sns.set(style="whitegrid")
    df["True Label"] = "Normal"
    cm = confusion_matrix(df["True Label"], df["CNN Predicted Class"], labels=CLASS_NAMES)

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
    input_folder = "Layered IG LSB/CNN MMS/Normal"
    model_path = "Layered IG LSB/CNN MMS/cnn_3_class_grayscale_model_64x64.h5"
    output_excel = "Layered IG LSB/CNN MMS/ig_layered_stego_results.xlsx"
    process_folder(input_folder, model_path, output_excel)
