import os
import cv2
import numpy as np
import joblib  # For saving/loading model
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA  # For dimensionality reduction
from concurrent.futures import ThreadPoolExecutor  # For faster image loading
import pandas as pd  # For saving test results

# Constants
IMAGE_SIZE = (128, 128)  # Resize all images to this size
N_COMPONENTS = 500  # PCA feature reduction
NU_VALUE = 0.02  # Adjust sensitivity (lower = strict, higher = relaxed)


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image (grayscale, resize, normalize, flatten).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Warning: Unable to read {image_path}")
            return None
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return img.flatten()  # Flatten to 1D feature vector
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def load_images_from_folder(folder):
    """
    Load and preprocess all images from a folder using multithreading.
    """
    images = []
    filenames = []

    if not os.path.exists(folder):
        print(f"❌ Error: Folder '{folder}' not found!")
        return np.array(images), filenames

    file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".png", ".jpg"))]

    if not file_list:
        print(f"⚠️ Warning: No images found in '{folder}'")
        return np.array(images), filenames

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_and_preprocess_image, file_list))

    for i, result in enumerate(results):
        if result is not None:
            images.append(result)
            filenames.append(os.path.basename(file_list[i]))

    return np.array(images), filenames


def train_one_class_svm(normal_folder, model_path="one_class_svm_model.pkl"):
    """
    Train One-Class SVM on normal images and save the trained model.
    """
    print("\n📂 Loading normal images for training...")
    X_train, _ = load_images_from_folder(normal_folder)

    if X_train.shape[0] == 0:
        print("❌ Error: No training images found!")
        return

    print(f"✅ Loaded {X_train.shape[0]} normal images for training.")
    print(f"Feature vector size: {X_train.shape[1]}")

    # Reduce dimensions using PCA
    print("\n🧠 Applying PCA for feature reduction...")
    pca = PCA(n_components=min(N_COMPONENTS, X_train.shape[1]))
    X_train_pca = pca.fit_transform(X_train)

    # Train One-Class SVM
    print("\n🚀 Training One-Class SVM...")
    svm_model = OneClassSVM(kernel="rbf", gamma="auto", nu=NU_VALUE)
    svm_model.fit(X_train_pca)

    # Save the trained model & PCA transformer
    joblib.dump((svm_model, pca), model_path)
    print(f"✅ Model trained and saved at {model_path}")


def test_one_class_svm(test_folder, model_path="one_class_svm_model.pkl", output_csv="anomaly_results.csv"):
    """
    Load a trained One-Class SVM model and test it on new images.
    """
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found! Train the model first.")
        return

    print("\n📂 Loading test images...")
    X_test, filenames = load_images_from_folder(test_folder)

    if X_test.shape[0] == 0:
        print("❌ No test images found!")
        return

    print(f"✅ Loaded {X_test.shape[0]} test images.")

    # Load trained model & PCA
    svm_model, pca = joblib.load(model_path)

    # Apply PCA to test data
    X_test_pca = pca.transform(X_test)

    # Predict anomalies (-1 = Anomaly, 1 = Normal)
    print("\n🔍 Predicting anomalies...")
    predictions = svm_model.predict(X_test_pca)

    # Save results
    results = pd.DataFrame({"Filename": filenames, "Prediction": predictions})
    results["Prediction"] = results["Prediction"].map({1: "Normal ✅", -1: "Anomaly ❌"})
    results.to_csv(output_csv, index=False)

    print(f"\n📄 Results saved to {output_csv}")

    # Display summary
    print("\n🔹 **Anomaly Detection Summary** 🔹")
    print(results.value_counts("Prediction"))
    print("\n✅ Detection completed!")


# ---------------------- MAIN EXECUTION ---------------------- #

# Define dataset paths
normal_data_folder = "One_Class_SVM/Normal"  # Folder containing only normal images
test_data_folder = "One_Class_SVM/Attack"  # Folder containing both normal and anomalous images
svm_model_path = "One_Class_SVM/one_class_svm_model.pkl"
output_results_csv = "One_Class_SVM/anomaly_results.csv"

# Step 1: Train the model
train_one_class_svm(normal_data_folder, svm_model_path)

# Step 2: Test on new images
test_one_class_svm(test_data_folder, svm_model_path, output_results_csv)
