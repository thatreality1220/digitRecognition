import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.express as px

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = (X_train / 255.0)[..., np.newaxis]
X_test = (X_test / 255.0)[..., np.newaxis]

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomRotation(0.02),
    keras.layers.RandomZoom(0.1),
    
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

def predict(data):
    data