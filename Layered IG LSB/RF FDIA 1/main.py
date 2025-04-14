import os
import numpy as np
import pandas as pd
import shap
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import random
import string
import joblib
import math
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct

# === CONFIG === #
IMG_SIZE = (128, 128)
MESSAGE_LENGTH = 4  # 4 characters
BITS_PER_CHAR = 8
TOTAL_BITS = MESSAGE_LENGTH * BITS_PER_CHAR
CLASS_NAMES = None
RESERVED_BITS = 1024  # First 1024 pixels reserved for coordinate metadata

# === UTILS === #
def generate_random_message(length=4):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def sanitize_string(input_string):
    return ''.join([char if ord(char) < 128 else '' for char in input_string]).strip()

def normalize_label(label):
    return str(label).strip().lower()

# === EMBEDDING & DECODING === #
def lsb_encode(image, message, positions):
    encoded_image = image.copy()
    message_bits = ''.join(format(ord(char), '08b') for char in message)
    bit_index, used_pixels, embedded_bits = 0, [], []

    for x, y in positions:
        if bit_index >= len(message_bits): break
        bit = int(message_bits[bit_index])
        encoded_image[x, y] = (encoded_image[x, y] & ~1) | bit
        used_pixels.append((x, y))
        embedded_bits.append(bit)
        bit_index += 1

    return encoded_image, used_pixels, embedded_bits

def lsb_decode(image, positions, length):
    bit_string = ''
    decoded_message = []
    total_bits = length * 8
    for i, (x, y) in enumerate(positions):
        if i >= total_bits: break
        bit_string += str(image[x, y] & 1)
        if len(bit_string) == 8:
            decoded_message.append(chr(int(bit_string, 2)))
            bit_string = ''
    return ''.join(decoded_message)

# === COORDINATE METADATA UTILS === #
def coords_to_bits(coords, bits_per_coord=16):
    bits = []
    for y, x in coords:
        y_bits = format(y, f'0{bits_per_coord}b')
        x_bits = format(x, f'0{bits_per_coord}b')
        bits.extend(int(b) for b in y_bits + x_bits)
    return bits

def bits_to_coords(bits, bits_per_coord=16):
    coords = []
    for i in range(0, len(bits), 2 * bits_per_coord):
        y = int(''.join(map(str, bits[i:i+bits_per_coord])), 2)
        x = int(''.join(map(str, bits[i+bits_per_coord:i+2*bits_per_coord])), 2)
        coords.append((y, x))
    return coords

def embed_metadata(image, meta_bits):
    flat = image.flatten()
    for i in range(len(meta_bits)):
        flat[i] = (flat[i] & ~(1 << 7)) | (meta_bits[i] << 7)
    return flat.reshape(image.shape)

def extract_metadata(image, reserved_bits=RESERVED_BITS):
    flat = image.flatten()
    bits = [(flat[i] >> 7) & 1 for i in range(reserved_bits)]
    return bits_to_coords(bits)

# === QUALITY METRICS === #
def calculate_mse(original, stego):
    return np.mean((original.astype(np.float32) - stego.astype(np.float32)) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    return float('inf') if mse == 0 else 20 * math.log10(max_pixel / math.sqrt(mse))

def calculate_ssim(original, stego):
    return ssim(original, stego)

def compute_dct_difference(original, stego):
    original_dct = dct(dct(original.T, norm='ortho').T, norm='ortho')
    stego_dct = dct(dct(stego.T, norm='ortho').T, norm='ortho')
    diff = np.abs(original_dct - stego_dct)
    return np.mean(diff), np.max(diff)

# === SHAP PIXEL SELECTION === #
def shap_pixel_selection(image, model, num_pixels=32):
    image_flattened = image.flatten().reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(image_flattened, check_additivity=False)[0]
    flat_shap_values = shap_values[:, 0] if shap_values.ndim == 2 else shap_values[0]
    top_indices = np.argsort(flat_shap_values)[-num_pixels:]
    return [(idx // image.shape[1], idx % image.shape[1]) for idx in top_indices]

# === IMAGE I/O === #
def load_images_from_folder(folder_path):
    images, filenames = [], []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            filenames.append(filename)
    return images, filenames

# === CONFUSION MATRIX PLOTTING === #
def save_confusion_matrix(y_true, y_pred, label_map, title, save_path, cmap="Blues"):
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    display_labels = [label_map.get(l, l.capitalize()) for l in present_labels]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=cmap, values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === MAIN PROCESS === #
def process_images(folder_path, model_path, output_excel, stego_output_folder, generate_confusion_matrix=True):
    images, filenames = load_images_from_folder(folder_path)
    rf_model = joblib.load(model_path)

    rf_estimator = rf_model.named_steps['rf'] if hasattr(rf_model, 'named_steps') else rf_model

    global CLASS_NAMES
    CLASS_NAMES = list(rf_model.classes_)
    class_labels_lower = [c.lower() for c in CLASS_NAMES]
    label_map = dict(zip(class_labels_lower, CLASS_NAMES))

    results = []
    folder_path_lower = os.path.basename(folder_path).lower()
    if folder_path_lower == 'normal':
        true_label = 'normal'
    elif folder_path_lower == 'attack':
        true_label = 'attack'
    else:
        raise ValueError(f"❌ Could not determine true label from folder name: {folder_path_lower}")

    for i, image in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {filenames[i]}")

        pred_orig = rf_model.predict([image.flatten()])[0]
        original_pred = normalize_label(pred_orig)

        selected_pixels = shap_pixel_selection(image, rf_estimator, num_pixels=TOTAL_BITS)
        message = generate_random_message()
        encoded_image, used_pixels, embedded_bits = lsb_encode(image, message, selected_pixels)
        coord_bits = coords_to_bits(selected_pixels)
        encoded_image = embed_metadata(encoded_image, coord_bits)

        stego_path = os.path.join(stego_output_folder, f"stego_{filenames[i]}")
        cv2.imwrite(stego_path, encoded_image)

        decoded_coords = extract_metadata(encoded_image, reserved_bits=RESERVED_BITS)
        decoded_message = lsb_decode(encoded_image, decoded_coords, len(message))

        mse_val = calculate_mse(image, encoded_image)
        psnr_val = calculate_psnr(mse_val)
        ssim_val = calculate_ssim(image, encoded_image)
        dct_mean, dct_max = compute_dct_difference(image, encoded_image)

        pred_stego = rf_model.predict([encoded_image.flatten()])[0]
        stego_pred = normalize_label(pred_stego)

        results.append({
            "Image": filenames[i],
            "Original Message": sanitize_string(message),
            "Decoded Message": sanitize_string(decoded_message),
            "Match": "Matched" if message == decoded_message else "Mismatched",
            "RF Predicted Class (Original)": original_pred,
            "RF Predicted Class": stego_pred,
            "True Label": true_label,
            "Used Pixels": str(selected_pixels),
            "Decoded Coords Match": "Matched" if selected_pixels == decoded_coords else "Mismatched",
            "MSE": mse_val,
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "Mean DCT Diff": dct_mean,
            "Max DCT Diff": dct_max
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
        "RF Accuracy After Embedding (%)": [(df["True Label"] == df["RF Predicted Class"]).mean() * 100],
        "RF Accuracy Before Embedding (%)": [(df["True Label"] == df["RF Predicted Class (Original)"]).mean() * 100]
    })

    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name='Per Image Results')
        summary_df.to_excel(writer, index=False, sheet_name='Summary Averages')

    output_dir = os.path.dirname(output_excel)

    if generate_confusion_matrix:
        save_confusion_matrix(
            y_true=df["True Label"],
            y_pred=df["RF Predicted Class"],
            label_map=label_map,
            title="Confusion Matrix (After Embedding)",
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
            cmap="Blues"
        )

        save_confusion_matrix(
            y_true=df["True Label"],
            y_pred=df["RF Predicted Class (Original)"],
            label_map=label_map,
            title="Confusion Matrix (Original Images)",
            save_path=os.path.join(output_dir, "confusion_matrix_original.png"),
            cmap="Greens"
        )

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Match")
    plt.title("Message Match Count")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_match_count.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Decoded Coords Match")
    plt.title("Coordinate Match Count")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coordinate_match_count.png"))
    plt.close()


# === RUN === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/RF FDIA 1/Normal"
    model_path = "Layered IG LSB/RF FDIA 1/best_rf_model.pkl"
    output_excel = "Layered IG LSB/RF FDIA 1/Final Results (Normal)/final_rf_layered_results.xlsx"
    stego_output_folder = "Layered IG LSB/RF FDIA 1/Final Results (Normal)/stego_images"

    if not os.path.exists(stego_output_folder):
        os.makedirs(stego_output_folder)

    process_images(input_folder, model_path, output_excel, stego_output_folder, generate_confusion_matrix=False)
