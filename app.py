import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('your_model_path.h5')

def classify_image():
    # Check if an image is selected
    if image_path:
        # Load and preprocess the image
        image = Image.open(image_path)
        image = image.resize((224, 224))  # Resize the image
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform classification
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        class_label = class_names[class_index]

        # Update the result label
        result_label.config(text=f"Classification Result: {class_label}")
    else:
        result_label.config(text="No image selected")

def open_image():
    global image_path
    image_path = filedialog.askopenfilename()
    image = Image.open(image_path)
    image.thumbnail((300, 300))  # Resize the image to fit in the label
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Create the main window
window = tk.Tk()
window.title("Diabetic Retinopathy Detection")

# Set the background image
background_image = Image.open("bg1.webp")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create the heading label
heading_label = tk.Label(window, text="Diabetic Retinopathy Detection", font=("Helvetica", 27, "bold"))
heading_label.pack(pady=10)

# Create the image label
image_label = tk.Label(window)
image_label.pack(pady=10)

# Create the open button
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack(pady=5)

# Create the classify button
classify_button = tk.Button(window, text="Classify", command=classify_image)
classify_button.pack(pady=5)

# Create the result label
result_label = tk.Label(window, text="Classification Result: ")
result_label.pack(pady=5)

# List of class names for the model output
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Create the bottom label
bottom_label = tk.Label(window, text="Done By - Anirudh Bojji & Siddharth Bollu", font=("Helvetica", 16))
bottom_label.pack(pady=10)

# Initialize the image path
image_path = None

# Run the main window loop
window.mainloop()
