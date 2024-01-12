import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import urllib.request
import os
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


mnist_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
local_mnist_path = "mnist.npz"

if not os.path.exists(local_mnist_path):
    print(f"Downloading data from {mnist_url}")
    urllib.request.urlretrieve(mnist_url, local_mnist_path)


with np.load(local_mnist_path, allow_pickle=True) as data:
    x_train, y_train, x_test, y_test = (
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
    )


model = load_model(
    "/Users/kemillamouri/Desktop/HETIC1erannee/CoursDATAHETIC1/Python/Projet perso/first_cnn/first_chiffre_cnn.h5"
)


class CNNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Model Prediction")

        self.root.geometry("500x700")
        self.root.resizable(False, False)

        self.load_random_image()

    def load_random_image(self):
        self.random_index = np.random.randint(0, x_test.shape[0])
        image = Image.fromarray(np.squeeze(x_test[self.random_index]) * 255.0).convert(
            "RGB"
        )
        image = image.resize((450, 425))

        tk_image = ImageTk.PhotoImage(image)
        self.image_label = ttk.Label(
            self.root,
            image=tk_image,
            borderwidth=6,
            relief="solid",
        )
        self.image_label.image = tk_image
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        preprocessed_image = np.squeeze(x_test[self.random_index])
        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_digit = np.argmax(prediction)

        result_label = ttk.Label(
            self.root,
            text=f"Predicted Digit: {predicted_digit}",
            font=("Helvetica", 25),
        )
        result_label.grid(row=2, column=0, pady=10)

        load_new_button = ttk.Button(
            self.root,
            text="Load New Image",
            command=self.load_random_image,
            style="TButton",
        )
        load_new_button.grid(row=1, column=0, pady=10)

        self.root.style = ttk.Style()
        self.root.style.configure(
            "TButton",
            font=("Helvetica", 14),
            padding=10,
            background="#4CAF50",
            foreground="white",
        )


root = tk.Tk()
app = CNNApp(root)
root.mainloop()
