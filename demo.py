import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import joblib
predicted_class = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary']
model = load_model('unet-model.h5', compile=False)
svm = joblib.load('svm_model.pkl')
pca = joblib.load('pca_model.pkl')
def browse_file():
    filename = filedialog.askopenfilename()
    if filename:
        img = Image.open(filename)
        img = img.convert('RGB').resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32')
        img_array /= 255.0

        prediction = model.predict(img_array)
        prediction = np.where(prediction > 0.5, 1, 0)

        predicted_mask = prediction.squeeze()
        predicted_mask_img = Image.fromarray((predicted_mask * 255).astype(np.uint8))

        mask_tk = ImageTk.PhotoImage(predicted_mask_img)
        label_prediction.config(image=mask_tk)
        label_prediction.image = mask_tk

        temp_prediction = prediction.reshape(1, -1)
        tumor_pca = pca.transform(temp_prediction)
        mask_prediction = svm.predict(tumor_pca)
        tumor_prediction.config(text=f"Predicted Class: {predicted_class[mask_prediction[0]]}")

root = tk.Tk()
root.title("File Browser")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = 800
window_height = 800

position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)

root.attributes("-fullscreen", True)

root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
btn_browse = tk.Button(root, text="Browse", command=browse_file)
btn_browse.pack(pady=20)

label_img = tk.Label(root)
label_img.pack(pady=20)

label_prediction = tk.Label(root)
label_prediction.pack(pady=20)

tumor_prediction = tk.Label(root, text="Predicted Class")
tumor_prediction.pack(pady=20)
root.mainloop()
