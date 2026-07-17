import tkinter as tk
from tkinter import Tk, Canvas
from PIL import Image, ImageGrab, ImageOps
from tensorflow import keras
import numpy as np

root = Tk()
root.title("Digit Classifier!")
root.geometry("600x600")
canvas = Canvas(root, bg="white", highlightthickness=0) # set up our tkinter interface
pen_color = "black"
brush_size = 12
last_x = None
last_y = None

pred_label = tk.Label(root, text="")
pred_label.pack(side="bottom", anchor="se")

conf_label = tk.Label(root, text="")
conf_label.pack(side="bottom", anchor="se") #adds our prediction and confidence labels to update later
def draw(event):
    global last_x, last_y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, event.x, event.y, fill=pen_color, width=brush_size, capstyle=tk.ROUND, smooth=tk.TRUE)
    # draw a line using our current and previous positions if the values of last_x and last_y are not None
    last_x = event.x
    last_y = event.y 
    # set our previous values to be equal to the current values to keep drawing a line
def start_draw(event):
    global last_x, last_y
    last_x = event.x # sets our last mouse positions to the current ones since its the first position
    last_y = event.y

def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None

def erase_all():
    canvas.delete("all") #deletes everything drawn on the canvas
def model_result():
    x1 = canvas.winfo_rootx()
    y1 = canvas.winfo_rooty()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()
    # calculate the bounding box dimensions for our tkinter screen
    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2)) #saves a copy of an image of the area inside our box
    screenshot = (screenshot.convert('L')).resize((28, 28)) # converts the image into grayscale and resizes it into a 28x28 pixel image
    screenshot = ImageOps.invert(screenshot) #inverts the colors so that the drawing is white and the background is black (to make the screenshot similar to our model's training data)


    prediction, confidence = predict(screenshot) # call our predict function

    pred_label.config(text=f"Prediction: {prediction}") #update the labels with our prediction and confidence 
    conf_label.config(text=f"Confidence: {confidence:.2%}")

def predict(data):
    model = keras.models.load_model("digit_classifier.keras") #load our model from our saved file
    data = np.array(data) # convert our screenshot into a numpy array

    data = (data.astype("float32") / 255.0)[..., np.newaxis]
    data = np.expand_dims(data, axis=0) # fit the image data to give our model
    predictions = model.predict(data, batch_size=None, verbose=0, steps=None, callbacks=None) # model gives us probabilities for each digit
    
    predicted_digit = np.argmax(predictions, axis=-1)[0] #gives us a clear prediction based on our digits
    confidence = predictions[0][predicted_digit] #gives us the model's calculated probability for our predicted number

    return predicted_digit, confidence
canvas.pack(fill=tk.BOTH, expand=True)
eraseButton = tk.Button(root, text="Erase", width=10, command=erase_all)
eraseButton.pack(side="bottom", anchor="sw")

predictButton = tk.Button(root, text="Predict", width=10, command=model_result)
predictButton.pack(side="bottom", anchor="sw")
#add our predict and erase buttons with their corresponding functions above
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw) #applies our functions to each of the inputs

root.mainloop()
#runs the code indefinitely until the user exits out of the program
