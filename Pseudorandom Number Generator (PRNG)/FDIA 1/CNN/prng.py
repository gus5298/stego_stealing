import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# ---------------- Parameters ---------------- #
input_folder = "PRNG/Images FDIA 1/Normal"
attacked_folder = "PRNG/attacked_images"
model_path = "PRNG/cnn_2_class_model_grayscale.h5"
output_excel = "PRNG/prng_pixel_encoding_results.xlsx"

img_input_size = (128, 128)
secret_seed = 12345
prng_pixel_count = 25
bits_per_coord = 16
total_bits = prng_pixel_count * bits_per_coord * 2  # 800 bits

# Create attacked folder
os.makedirs(attacked_folder, exist_ok=True)

# ---------------- PRNG & LSB Functions ---------------- #
def generate_prng_pixel_positions(image_shape, count, seed_value):
    np.random.seed(seed_value)
    h, w = image_shape
    indices = np.random.choice(h * w, size=count, replace=False)
    rows, cols = np.unravel_index(indices, (h, w))
    return list(zip(rows, cols))

def coords_to_bits(coords):
    bit_list = []
    for row, col in coords:
        row_bits = format(row, f'0{bits_per_coord}b')
        col_bits = format(col, f'0{bits_per_coord}b')
        bit_list.extend(int(b) for b in row_bits + col_bits)
    return bit_list

def bits_to_coords(bit_list):
    coords = []
    for i in range(0, len(bit_list), bits_per_coord * 2):
        row_bits = bit_list[i:i+bits_per_coord]
        col_bits = bit_list[i+bits_per_coord:i+bits_per_coord*2]
        row = int(''.join(map(str, row_bits)), 2)
        col = int(''.join(map(str, col_bits)), 2)
        coords.append((row, col))
    return coords

def encode_bits_lsb(image_array, bit_values):
    flat = image_array.flatten()
    for i, bit in enumerate(bit_values):
        flat[i] = (flat[i] & ~1) | bit
    return flat.reshape(image_array.shape)

def decode_bits_lsb(image_array, num_bits):
    flat = image_array.flatten()
    return [flat[i] & 1 for i in range(num_bits)]

# ---------------- Load CNN Model ---------------- #
if not os.path.exists(model_path):
    print("❌ CNN model not found!")
    exit()
model = load_model(model_path)

# ---------------- Process Images ---------------- #
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg'))]
results = []

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Generate PRNG coords and encode bits
    img_shape = img.shape
    prng_coords = generate_prng_pixel_positions(img_shape, prng_pixel_count, secret_seed + idx)
    encoded_bits = coords_to_bits(prng_coords)
    attacked_img = encode_bits_lsb(img, encoded_bits)

    # Save attacked image
    save_path = os.path.join(attacked_folder, img_file)
    cv2.imwrite(save_path, attacked_img)

    # Prepare for CNN prediction (grayscale model)
    resized = cv2.resize(attacked_img, img_input_size).astype(np.float32) / 255.0
    input_img = np.expand_dims(resized, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    # CNN prediction
    pred_prob = float(model.predict(input_img, verbose=0)[0][0])
    pred_class = "Normal" if pred_prob > 0.5 else "Faulty"

    # Decode LSB bits
    decoded_bits = decode_bits_lsb(attacked_img, total_bits)
    decoded_coords = bits_to_coords(decoded_bits)

    # Compare bits
    bit_match_count = sum(e == d for e, d in zip(encoded_bits, decoded_bits))
    bit_match_percent = round((bit_match_count / total_bits) * 100, 2)
    bit_match_result = "Match ✅" if bit_match_percent == 100.0 else "Mismatch ❌"

    # Compare coords
    coords_match = prng_coords == decoded_coords
    coords_match_result = "Match ✅" if coords_match else "Mismatch ❌"

    # Save all data
    results.append({
        "Image": img_file,
        "Prediction": pred_class,
        "Confidence": round(pred_prob, 4),
        "Encoded Bits": str(encoded_bits),
        "Decoded Bits": str(decoded_bits),
        "Bit Match (%)": bit_match_percent,
        "Message Match": bit_match_result,
        "Original PRNG Coords": str(prng_coords),
        "Decoded Coords": str(decoded_coords),
        "Coords Match": coords_match_result
    })

# ---------------- Save to Excel ---------------- #
df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)
print(f"\n✅ Results saved to '{output_excel}' with full encoded messages.")
