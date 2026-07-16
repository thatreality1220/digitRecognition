import tkinter as tk
from tkinter import Tk, Canvas
from PIL import Image, ImageGrab, ImageOps
from tensorflow import keras
import numpy as np

root = Tk()
root.title("Digit Classifier!")
root.geometry("600x600")
canvas = Canvas(root, bg="white", highlightthickness=0)
pen_color = "black"
brush_size = 12;
last_x = None
last_y = None

pred_label = tk.Label(root, text="")
pred_label.pack(side="bottom", anchor="se")

conf_label = tk.Label(root, text="")
conf_label.pack(side="bottom", anchor="se")
def draw(event):
    global last_x, last_y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, event.x, event.y, fill=pen_color, width=brush_size, capstyle=tk.ROUND, smooth=tk.TRUE)
    last_x = event.x
    last_y = event.y

def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y

def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None

def erase_all():
    canvas.delete("all")
def model_result():
    x1 = canvas.winfo_rootx()
    y1 = canvas.winfo_rooty()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()

    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    screenshot = (screenshot.convert('L')).resize((28, 28))
    screenshot = ImageOps.invert(screenshot)

    prediction, confidence = predict(screenshot)

    pred_label.config(text=f"Prediction: {prediction}")
    conf_label.config(text=f"Confidence: {confidence:.2%}")

def predict(data):
    CLASS_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    model = keras.models.load_model("digit_classifier.keras")
    data = np.array(data)

    data = (data.astype("float32") / 255.0)[..., np.newaxis]
    data = np.expand_dims(data, axis=0)
    predictions = model.predict(data, batch_size=None, verbose=0, steps=None, callbacks=None)
    
    prediction = np.argmax(predictions, axis=-1)[0]
    predicted_digit = CLASS_LABELS[prediction]
    confidence = predictions[0][prediction]

    return predicted_digit, confidence
canvas.pack(fill=tk.BOTH, expand=True)
eraseButton = tk.Button(root, text="Erase", width=10, command=erase_all)
eraseButton.pack(side="bottom", anchor="sw")

predictButton = tk.Button(root, text="Predict", width=10, command=model_result)
predictButton.pack(side="bottom", anchor="sw")

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

root.mainloop()
