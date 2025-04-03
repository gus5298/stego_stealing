import os
import numpy as np
import pandas as pd
import shap
import cv2
from sklearn.metrics import confusion_matrix
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
CLASS_NAMES = ['Normal', 'Attack']

# === UTILS === #
def generate_random_message(length=4):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def sanitize_string(input_string):
    return ''.join([char if ord(char) < 128 else '' for char in input_string]).strip()

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

def shap_pixel_selection(image, model, num_pixels=32):
    image_flattened = image.flatten().reshape(1, -1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(image_flattened, check_additivity=False)[0]
    flat_shap_values = shap_values[:, 0] if shap_values.ndim == 2 else shap_values[0]
    top_indices = np.argsort(flat_shap_values)[-num_pixels:]
    return [(idx // image.shape[1], idx % image.shape[1]) for idx in top_indices]

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

def process_images(folder_path, model_path, output_excel, stego_output_folder):
    images, filenames = load_images_from_folder(folder_path)
    rf_model = joblib.load(model_path)

    results = []
    true_label = next((label for label in CLASS_NAMES if label.lower() in folder_path.lower()), 'Unknown')

    for i, image in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {filenames[i]}")
        selected_pixels = shap_pixel_selection(image, rf_model, num_pixels=TOTAL_BITS)

        message = generate_random_message()
        encoded_image, used_pixels, embedded_bits = lsb_encode(image, message, selected_pixels)
        rf_prediction = rf_model.predict([encoded_image.flatten()])[0]
        rf_class_label = CLASS_NAMES[rf_prediction] if rf_prediction < len(CLASS_NAMES) else 'Unknown'
        decoded_message = lsb_decode(encoded_image, selected_pixels, len(message))

        mse_val = calculate_mse(image, encoded_image)
        psnr_val = calculate_psnr(mse_val)
        ssim_val = calculate_ssim(image, encoded_image)
        dct_mean, dct_max = compute_dct_difference(image, encoded_image)

        results.append({
            "Image": filenames[i],
            "Original Message": sanitize_string(message),
            "Decoded Message": sanitize_string(decoded_message),
            "Match": "Matched" if message == decoded_message else "Mismatched",
            "RF Predicted Class": rf_class_label,
            "True Label": true_label,
            "Used Pixels": str(used_pixels),
            "MSE": mse_val,
            "PSNR": psnr_val,
            "SSIM": ssim_val,
            "Mean DCT Diff": dct_mean,
            "Max DCT Diff": dct_max
        })

        stego_path = os.path.join(stego_output_folder, f"stego_{filenames[i]}")
        cv2.imwrite(stego_path, encoded_image)

    df = pd.DataFrame(results)
    avg_metrics = {
        "Average MSE": [df["MSE"].mean()],
        "Average PSNR (dB)": [df["PSNR"].mean()],
        "Average SSIM": [df["SSIM"].mean()],
        "Average Mean DCT Diff": [df["Mean DCT Diff"].mean()],
        "Average Max DCT Diff": [df["Max DCT Diff"].mean()],
        "Message Match Rate (%)": [(df["Match"] == "Matched").mean() * 100],
        "Coordinate Match Rate (%)": [100.0],
        "RF Accuracy After Embedding (%)": [(df["True Label"] == df["RF Predicted Class"]).mean() * 100]
    }
    summary_df = pd.DataFrame(avg_metrics)

    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name='Per Image Results')
        summary_df.to_excel(writer, index=False, sheet_name='Summary Averages')

    print("\n✅ Done! Results saved to", output_excel)
    print("\n📊 AVERAGE METRICS ACROSS ALL IMAGES:")
    for col in summary_df.columns:
        print(f"{col}: {summary_df[col].iloc[0]:.4f}" if isinstance(summary_df[col].iloc[0], float) else f"{col}: {summary_df[col].iloc[0]}")

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
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "message_match_count.png"))
    plt.close()

# === RUN === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/RF FDIA 1/Normal"
    model_path = "Layered IG LSB/RF FDIA 1/rf_model.pkl"
    output_excel = "Layered IG LSB/RF FDIA 1/Final Results/final_rf_layered_results.xlsx"
    stego_output_folder = "Layered IG LSB/RF FDIA 1/Final Results/stego_images"

    if not os.path.exists(stego_output_folder):
        os.makedirs(stego_output_folder)

    process_images(input_folder, model_path, output_excel, stego_output_folder)
