import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('license.h5')

# Class mapping remains the same
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}

predicted_labels = []

# Loop through image files
for i in range(1, 8):  # For imagen1.jpg to imagen7.jpg
    img_path = f'imagen{i}.jpg'
    img = Image.open(img_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match model input
    img = Image.fromarray(255 - np.array(img))  # Invert colors
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28)  # Reshape for the model
    
    prediction = model.predict(img)
    label_pred = class_mapping[np.argmax(prediction)]
    predicted_labels.append(label_pred)
    
    # Optionally display each image with its prediction
    plt.imshow(img[0], cmap='binary_r')
    plt.title(f"Prediction for imagen{i}.jpg: {label_pred}")
    plt.show()

# Print out values gathered from each prediction
for i, label in enumerate(predicted_labels, start=1):
    print(f"Prediction for imagen{i}.jpg: {label}")

# Concatenate all labels into a single line
full_prediction = ''.join(predicted_labels)
print(f"Full sequence: {full_prediction}")
