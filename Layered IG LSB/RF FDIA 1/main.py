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

# Helper function to generate random 4-letter message (32 bits)
def generate_random_message(length=4):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Helper function to perform LSB (Least Significant Bit) encoding
def lsb_encode(image, message, positions):
    encoded_image = image.copy()
    message_bits = ''.join(format(ord(char), '08b') for char in message)  # Convert message to bits
    bit_index = 0  # Track the bit position in the message
    
    used_pixels = []  # Store the pixels used for LSB encoding
    embedded_bits = []  # Store the bits being embedded
    
    for i, pos in enumerate(positions):
        if bit_index >= len(message_bits):  # Stop encoding when all message bits are encoded
            break
        
        x, y = pos
        bit = int(message_bits[bit_index])  # Get the next bit of the message
        encoded_image[x, y] = (encoded_image[x, y] & ~1) | bit  # Encode the bit in the LSB
        
        # Track the pixel location and the embedded bit
        used_pixels.append((x, y))
        embedded_bits.append(bit)
        
        bit_index += 1  # Move to the next bit
    
    return encoded_image, used_pixels, embedded_bits

# Helper function to perform LSB decoding
def lsb_decode(image, positions, message_length):
    decoded_message = []
    bit_string = ""
    total_bits = message_length * 8  # Total bits in the message (4 characters * 8 bits each)
    bit_index = 0  # Track the bit position in the message
    
    for i, pos in enumerate(positions):
        if bit_index >= total_bits:  # Stop decoding when all message bits are decoded
            break
        
        x, y = pos
        bit = image[x, y] & 1  # Get the least significant bit from the pixel
        bit_string += str(bit)  # Add the bit to the bit string
        
        if len(bit_string) == 8:  # Every 8 bits form one byte (character)
            decoded_message.append(chr(int(bit_string, 2)))  # Convert bits to character
            bit_string = ""  # Reset for the next byte
        
        bit_index += 1  # Move to the next bit
    
    return ''.join(decoded_message)

# Function to sanitize strings for illegal characters in Excel
def sanitize_string(input_string):
    sanitized = ''.join([char if ord(char) >= 32 else '' for char in input_string])
    return sanitized

# Step 1: Input folder with images
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Step 2: Load pre-trained RF model and SHAP analysis
def load_rf_model(model_path):
    rf_model = joblib.load(model_path)
    return rf_model

# Step 3: Randomly select pixels for encoding
def get_random_pixels(image, num_pixels=32):
    height, width = image.shape
    random_pixels = [(random.randint(0, height - 1), random.randint(0, width - 1)) for _ in range(num_pixels)]
    return random_pixels

# Step 4: Perform layered steganography (Encode the message in LSB)
def layered_steganography(image, message, random_pixels):
    encoded_image, used_pixels, embedded_bits = lsb_encode(image, message, random_pixels)
    return encoded_image, used_pixels, embedded_bits

# Step 5: Evaluate RF model on the new images
def rf_predict(rf_model, images):
    X = np.array([img.flatten() for img in images])
    predictions = rf_model.predict(X)
    return predictions

# Step 6: Generate results and save to Excel
def create_results_dataframe(filenames, encoded_messages, decoded_messages, rf_predictions, matches, used_pixels, embedded_bits, excel_path):
    sanitized_encoded_messages = [sanitize_string(msg) for msg in encoded_messages]
    sanitized_decoded_messages = [sanitize_string(msg) for msg in decoded_messages]
    
    data = {
        'Image': filenames,
        'Encoded Message': sanitized_encoded_messages,
        'Decoded Message': sanitized_decoded_messages,
        'RF Prediction': rf_predictions,
        'Messages Match': matches,
        'Used Pixels': used_pixels,
        'Embedded Bits': embedded_bits
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)

# Step 7: Visualize results with confusion matrix and bar graphs
def visualize_results(filenames, predictions, matches, random_pixels, decoded_messages):
    cm = confusion_matrix([0 if "Normal" in fname else 1 for fname in filenames], predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    correct_positions = [i for i, match in enumerate(matches) if match]
    plt.bar(range(len(correct_positions)), correct_positions)
    plt.title("Correct Message Locations")
    plt.xlabel("Image Index")
    plt.ylabel("Pixel Location")
    plt.show()

    match_counts = [matches.count(True), matches.count(False)]
    plt.bar(['Matches', 'Mismatches'], match_counts)
    plt.title("Message Matching Accuracy")
    plt.ylabel("Count")
    plt.show()

# Main function to run the entire process
def process_images(folder_path, model_path, excel_path):
    images, filenames = load_images_from_folder(folder_path)
    rf_model = load_rf_model(model_path)

    encoded_messages = []
    decoded_messages = []
    rf_predictions = []
    matches = []
    used_pixels_all = []
    embedded_bits_all = []

    for i, image in enumerate(images):
        print(f"Processing image: {filenames[i]}")
        
        random_pixels = get_random_pixels(image, num_pixels=32)

        message = generate_random_message()
        encoded_messages.append(message)

        encoded_image, used_pixels, embedded_bits = layered_steganography(image, message, random_pixels)
        
        rf_prediction = rf_predict(rf_model, [encoded_image.flatten()])[0]
        rf_predictions.append(rf_prediction)

        decoded_message = lsb_decode(encoded_image, random_pixels, len(message))
        decoded_messages.append(decoded_message)
        
        match = decoded_message == message
        matches.append(match)
        
        used_pixels_all.append(used_pixels)
        embedded_bits_all.append(embedded_bits)
        
        print(f"Encoded message: {message}")
        print(f"Decoded message: {decoded_message}")
        print(f"Do the messages match? {'Yes' if match else 'No'}\n")

    total_matched = matches.count(True)
    total_mismatched = matches.count(False)

    print(f"Total Matched Messages: {total_matched}")
    print(f"Total Mismatched Messages: {total_mismatched}")

    create_results_dataframe(filenames, encoded_messages, decoded_messages, rf_predictions, matches, used_pixels_all, embedded_bits_all, excel_path)
    visualize_results(filenames, rf_predictions, matches, random_pixels, decoded_messages)


# === RUN SCRIPT === #
if __name__ == "__main__":
    input_folder = "Layered IG LSB/RF FDIA 1/Normal"
    model_path = "Layered IG LSB/RF FDIA 1/rf_model.pkl"
    output_excel = "Layered IG LSB/RF FDIA 1/rf_layered_stego_results.xlsx"
    process_images(input_folder, model_path, output_excel)
