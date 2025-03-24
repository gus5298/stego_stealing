import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Parameters
img_dir = "MMS RF/Images"  # Folder structure: MMS RF/Images/Attack/, Normal/
img_size = (128, 128)
class_names = ['Attack', 'Normal']

# Load images and labels
print("🔄 Loading and processing images...")
X, y = [], []

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(img_dir, class_name)
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(class_folder, img_file)
        img = Image.open(img_path).resize(img_size)
        img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
        X.append(img_array)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} images with {X.shape[1]} features each.")

# Train-Test Split
print("🔀 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train Random Forest
print("🌲 Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=200, random_state=6553)
rf.fit(X_train, y_train)
print("✅ Model training complete.")

# Predict
print("🔍 Making predictions on test set...")
y_pred = rf.predict(X_test)

# Evaluate
print("\n📊 Evaluation Results:")
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

print("🧩 Confusion Matrix (Raw):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plot and save confusion matrix
print("🖼️ Saving confusion matrix as an image...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved as confusion_matrix.png")

print("🎉 Done!")
