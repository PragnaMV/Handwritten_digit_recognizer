from keras.models import load_model # type: ignore
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

model = load_model('mnist_improved_v2.h5')  # Use the updated model file

def predict_digit(img):
    # Resize image to 28x28 pixels
    img = img.resize((28, 28))
    # Convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # Reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = (255 - img) / 255.0

    # Predicting the class
    res = model.predict([img])[0]
    top_3_indices = np.argsort(res)[-3:][::-1]
    top_3_probs = res[top_3_indices]
    return top_3_indices, top_3_probs

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # Get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # Get the coordinate of the canvas
        a, b, c, d = rect
        rect = (a + 4, b + 4, c - 4, d - 4)
        im = ImageGrab.grab(rect)

        digits, accs = predict_digit(im)
        text = f"{digits[0]}: {int(accs[0] * 100)}%, {digits[1]}: {int(accs[1] * 100)}%, {digits[2]}: {int(accs[2] * 100)}%"
        self.label.configure(text=text)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
