import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Dataset paths
base_dir = "MMS Images"  # Structure: train/, val/, test/ inside MMS Images
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Parameters
img_height, img_width = 64, 64
batch_size = 32
num_classes = 3
learning_rate = 0.001
epochs = 42
class_names = ['Normal', 'Attack', 'Faulty']

# Data preprocessing (rescaling only)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    classes=class_names)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Important for matching order in y_true
    classes=class_names)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("🚀 Training CNN model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1)

# Evaluate on test data
print("\n🔍 Evaluating on test data...")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

# Predict on test set
print("🔮 Predicting class labels...")
y_true = test_generator.classes  # Ground truth labels
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class indices

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot and save confusion matrix
print("🖼️ Saving confusion matrix...")
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved as cnn_confusion_matrix.png")

# Classification report
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save model
model.save("cnn_3_class_model.h5")
print("💾 Model saved as cnn_3_class_model.h5")
print("🎉 Done!")
