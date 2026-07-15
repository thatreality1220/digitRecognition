import tkinter as tk
from tkinter import Tk, Canvas
from model import predict

root = Tk()
root.title("Digit Classifier!")
root.geometry("600x600")
canvas = Canvas(root, bg="white", highlightthickness=0)
pen_color = "black"
brush_size = 5;
last_x = None
last_y = None

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
    print("In progress...")

canvas.pack(fill=tk.BOTH, expand=True)
eraseButton = tk.Button(root, text="Erase", width=10, command=erase_all)
eraseButton.pack(side="bottom", anchor="sw")

predictButton = tk.Button(root, text="Predict", width=10, command=model_result)
predictButton.pack(side="bottom", anchor="sw")

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

root.mainloop()
