import os
import cv2
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# === CONFIGURATION === #
IMG_SIZE = (64, 64)
NUM_PIXELS = 16  # We will use top 16 important pixels for encoding the message
BITS_PER_COORD = 16
RESERVED_BITS = NUM_PIXELS * BITS_PER_COORD * 2  # For embedding coordinates of the top 16 pixels
MESSAGE_BITS = 16 * 8  # 2 letter message, each letter is 8 bits
CHANNELS = 1
CLASS_NAMES = ['Normal', 'Faulty', 'Attack']  # Actual classes

# === MESSAGE UTILS === #
def generate_random_message():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=2))

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

# === CLEANING FUNCTION FOR EXCEL === #
def clean_message(s):
    """
    Clean message to remove illegal characters (non-printable characters).
    """
    return ''.join(c for c in s if c.isprintable())

# === EMBEDDING === #
def embed_lsb(image, bits, coords):
    """
    First layer: Embed the message (LSB) into the image.
    """
    flat = image.flatten()
    for i, (y, x) in enumerate(coords):
        idx = y * image.shape[1] + x
        flat[idx] = (flat[idx] & ~1) | bits[i]  # Modify LSB for message embedding
    return flat.reshape(image.shape)

def embed_metadata(image, meta_bits):
    """
    Second layer: Embed coordinates or metadata into higher bit planes (e.g., 8th bit).
    """
    flat = image.flatten()
    for i in range(len(meta_bits)):
        flat[i] = (flat[i] & ~(1 << 7)) | (meta_bits[i] << 7)  # Set bit 7 for coordinates (higher bit)
    return flat.reshape(image.shape)

# === IG-BASED IMPORTANT PIXELS === #
def get_ig_pixels(model, image_tensor, label_index, top_k=NUM_PIXELS):
    def loss_fn(output): return output[:, label_index]
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    saliency_map = saliency(loss_fn, image_tensor)
    saliency_map = np.abs(saliency_map[0])
    if saliency_map.ndim == 3:
        saliency_map = np.mean(saliency_map, axis=-1)
    flat = saliency_map.flatten()
    indices = np.argsort(flat)[-top_k:]  # Get the top k important pixels
    ys, xs = np.unravel_index(indices, saliency_map.shape)
    return list(zip(ys, xs))

# === DECODING === #
def decode_message_and_coords(image):
    flat = image.flatten()

    # Extract coordinates from the higher bit plane (bit 7 for example)
    coord_bits = [(flat[i] >> 7) & 1 for i in range(RESERVED_BITS)]  # Extract higher bit for coordinates
    coords = bits_to_coords(coord_bits)  # Convert bits to coordinates

    # Extract message bits from the LSBs of the image pixels
    msg_bits = [flat[y * image.shape[1] + x] & 1 for y, x in coords[:NUM_PIXELS]]  # Extract LSB for message
    return bits_to_string(msg_bits), coords

# === MAIN PROCESSING FUNCTION === #
def process_folder(input_folder, model_path, output_excel):
    model = load_model(model_path)
    results = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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
        norm_img = image.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(norm_img, axis=(0, -1))  # shape: (1, 64, 64, 1)

        pred_probs = model.predict(input_tensor, verbose=0)[0]
        label_index = int(np.argmax(pred_probs))
        ig_pixels = get_ig_pixels(model, input_tensor, label_index, top_k=NUM_PIXELS)

        # Encode a random 2-letter message
        original_msg = generate_random_message()
        msg_bits = string_to_bits(original_msg)
        coord_bits = coords_to_bits(ig_pixels)

        # Embed the message and the coordinates
        msg_encoded_img = embed_lsb(image.copy(), msg_bits, ig_pixels)
        cnn_input = np.expand_dims(msg_encoded_img.astype(np.float32) / 255.0, axis=(0, -1))  # shape: (1, 64, 64, 1)

        # CNN Prediction after embedding the message
        pred_probs_after_lsb = model.predict(cnn_input, verbose=0)[0]
        pred_class_idx = int(np.argmax(pred_probs_after_lsb))
        pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else 'Unknown'
        confidence = float(pred_probs_after_lsb[pred_class_idx])

        # Embed metadata (coordinates) into the image
        final_stego = embed_metadata(msg_encoded_img, coord_bits)

        # Decode the message and coordinates from the stego image
        decoded_msg, decoded_coords = decode_message_and_coords(final_stego)

        # Clean the messages (for Excel storage)
        original_clean = clean_message(original_msg)  # Clean original message
        decoded_clean = clean_message(decoded_msg)  # Clean decoded message

        # Print encoded and decoded message, and whether they match
        print(f"Encoded Message: {original_clean}")
        print(f"Decoded Message: {decoded_clean}")
        print(f"Messages Match: {'Matched' if original_clean == decoded_clean else 'Mismatched'}\n")

        # Count matched and mismatched messages
        if original_clean == decoded_clean:
            matched_count += 1
        else:
            mismatched_count += 1

        # Extract True Label from folder name (e.g., "Normal", "Faulty", "Attack")
        true_label = os.path.basename(input_folder)  # Use basename to get the folder name directly
        if true_label not in CLASS_NAMES:
            print(f"Warning: '{true_label}' not found in CLASS_NAMES. Using 'Unknown' instead.")
            true_label = 'Unknown'  # Default if label isn't found

        # Store results for Excel
        results.append({
            "Image": filename,
            "Original Message": original_clean,
            "Decoded Message": decoded_clean,
            "Match": "Matched" if original_clean == decoded_clean else "Mismatched",
            "CNN Predicted Class": pred_class,
            "Confidence": round(confidence, 4),
            "Encoded Coords": str(ig_pixels),
            "Decoded Coords Match": "Matched" if ig_pixels == decoded_coords else "Mismatched",
            "True Label": true_label  # Add True Label to DataFrame
        })

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Done! Results saved to {output_excel}")

    # Print total matched and mismatched messages
    print(f"\nTotal Matched Messages: {matched_count}")
    print(f"Total Mismatched Messages: {mismatched_count}")

    # === Visualization === #
    sns.set(style="whitegrid")
    cm = confusion_matrix(df["True Label"], df["CNN Predicted Class"], labels=CLASS_NAMES)

    output_dir = os.path.dirname(output_excel)

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Message Match Count Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Match")
    plt.title("Message Match Count")
    plt.ylabel("Number of Images")
    plt.xlabel("Match Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_match_count.png"))
    plt.close()

    # Coordinate Match Count Plot
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
