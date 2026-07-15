import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 

penguins = sns.load_dataset("penguins").dropna()
fig = px.scatter(
    penguins,
    x = "flipper_length_mm",
    y = "bill_length_mm",
    color = "species",
    title = "Penguins by flipper and bill length",

)
fig.show()

feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

X = penguins[feature_cols].values
y = LabelEncoder().fit_transform(penguins["species"]) #transforms our species data in individual numbers or labels for the model to process

print("Features shape: ", X.shape)
print("First 10 labels: ", y[:150])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splits our data. Random state shuffles our data the same way every time we run the code
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Input(shape=(4, )), #gives 4 neurons for the input layer which correspond to each of our x values.
    keras.layers.Dense(16, activation="relu"), #makes our first hidden layer with 16 neurons. Dense = hidden layer
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(3, activation="softmax"), #our output that turns the numbers into probabilities for each of the 3 species

])
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, verbose=1) #trains our model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.2%}")

predictions = model.predict(X_test) #The model makes a prediction based on our x test values
y_pred = np.argmax(predictions, axis=1) #converts all of our predictions to 0 or 1
cm = confusion_matrix(y_test, y_pred) #compares our y test values to our model's predictions of the y values based on our X test values
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Adelie", "Chinstrap", "Gentoo"], yticklabels=["Adelie", "Chinstrap", "Gentoo"]) # creates labels on our confusion matrix for the x and y values
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Penguin Species")

plt.show()
