import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as im
import random 
import numpy as np
import seaborn as sns
from emnist import extract_training_samples, extract_test_samples




model = tf.keras.models.load_model('license.h5')

mnist = tf.keras.datasets.fashion_mnist


x_train, y_train = extract_training_samples('balanced')

x_test, y_test = extract_test_samples('balanced')
n = random.randint(0,len(x_test))


class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
    40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
}



#!Normalizacion
x_test = x_test/255


predictions = model.predict(x_test)

conf_matrix = tf.math.confusion_matrix(y_test, np.argmax(predictions, axis=1))
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Reales')
plt.show()


plt.imshow(x_test[n], cmap="binary_r")
plt.xlabel(f'Real: {class_mapping[y_test[n]]} , Pred: {class_mapping[np.argmax(predictions[n])]}')
plt.show()

