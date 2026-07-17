import tensorflow as tf
from tensorflow import keras
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() # load the keras mnist dataset into a test set and training data set

X_train = (X_train / 255.0)[..., np.newaxis]
X_test = (X_test / 255.0)[..., np.newaxis] #normalize our training and testing data + add a new axis to match keras' syntax

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)), #build our input layer that takes in our 28x28 pixel image

    keras.layers.RandomTranslation(0.1, 0.1), 
    keras.layers.RandomRotation(0.02),
    keras.layers.RandomZoom(0.1), #randomly alter the image so that it's more realistic
    
    keras.layers.Conv2D(32, (3, 3), activation="relu"), #places a 3x3 pixel filter throughout our image
    keras.layers.MaxPooling2D((2, 2)), #take the largest values from our 3x3 filter

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(), # change our 28x28 image into a 1D array of numbers (0-1)
    keras.layers.Dense(64, activation="relu"), # 1 normal hidden layer with 64 neurons
    keras.layers.Dense(10, activation="softmax") #gives us 10 probabilties for each of the digits
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # configues our model's optimizer and our way of calculating loss
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
) #stops training our model when the accuracy doesn't decrease for 5 iterations
history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, verbose=1, callbacks=[early_stopping]) # splits our dataset into 90% training data and 10% testing data
#and adjusts how many iterations we train our model for

y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fig = px.imshow(cm,
                x=class_names,
                y=class_names,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted",
                            y="Actual",
                            color="Count",
                            title="Confusion Matrix"))

fig.show()
#Draws a confusion matrix that shows what the model gets wrong
model.save("digit_classifier.keras")
# saves our trained model into a seperate .keras file
