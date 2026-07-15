import tkinter as tk
from tkinter import Tk, Canvas
from PIL import Image, ImageGrab, ImageOps
import model
from model import predict


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

canvas.pack(fill=tk.BOTH, expand=True)
eraseButton = tk.Button(root, text="Erase", width=10, command=erase_all)
eraseButton.pack(side="bottom", anchor="sw")

predictButton = tk.Button(root, text="Predict", width=10, command=model_result)
predictButton.pack(side="bottom", anchor="sw")

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

root.mainloop()
