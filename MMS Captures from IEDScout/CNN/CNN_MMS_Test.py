import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
model_path = "cnn_3_class_model(93.45).h5"
test_dir = "MMS Images/test"  # Path to your test dataset
class_names = ['Normal', 'Attack', 'Faulty']
img_size = (64, 64)
batch_size = 32

# Load model
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded.")

# Data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=class_names)

# Predict
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot & save
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_from_weights.png")
plt.close()
print("✅ Confusion matrix saved as confusion_matrix_from_weights.png")

# Classification report
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
