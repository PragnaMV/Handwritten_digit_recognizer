from imp import load_module
from tkinter import W, mainloop
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Flatten, BatchNormalization # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D # type: ignore

from keras.optimizers import Adam # type: ignore
import numpy as np # type: ignore
# from keras.models import load_model
# from tkinter import *
# import tkinter as tk
# import win32gui
# from PIL import ImageGrab, Image
# import numpy as np

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=12,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)
datagen.fit(x_train)

# Model parameters
batch_size = 128
num_classes = 10
epochs = 50

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with a lower learning rate
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
hist = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                 steps_per_epoch=len(x_train) // batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))

print("The model has successfully trained")

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('mnist_improved_v2.h5')
print("Saving the model as mnist_improved_v2.h5")

# Load the model
model = load_module('mnist_improved_v2.h5')

def predict_digit(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = (255-img)/255.0

    res = model.predict([img])[0]
    top_3_indices = np.argsort(res)[-3:][::-1]
    top_3_probs = res[top_3_indices]
    return top_3_indices, top_3_probs

class App(tk.Tk): # type: ignore
    def __init__(self):
        tk.Tk.__init__(self) # type: ignore

        self.x = self.y = 0
        
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross") # type: ignore
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48)) # type: ignore
        self.classify_btn = tk.Button(self, text = "Recognize", command = self.classify_handwriting)    # type: ignore
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all) # type: ignore
       
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND) # type: ignore
        a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect) # type: ignore

        digits, accs = predict_digit(im)
        text = f"{digits[0]}: {int(accs[0]*100)}%, {digits[1]}: {int(accs[1]*100)}%, {digits[2]}: {int(accs[2]*100)}%"
        self.label.configure(text= text)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
