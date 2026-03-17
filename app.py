from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import requests
from tensorflow.keras.preprocessing import image

MODEL_URL = "https://huggingface.co/jerin96/face-mask-model/resolve/main/facemask.h5"
MODEL_PATH = "facemask.h5"

app = Flask(__name__)

# ===== Upload Folder =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = None   # ⭐ lazy global model


def load_model_once():
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            r = requests.get(MODEL_URL)
            open(MODEL_PATH, "wb").write(r.content)

        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")


def predict_mask(img_path):
    load_model_once()

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return f"No Mask ({pred*100:.2f}%)"
    else:
        return f"Mask ({(1-pred)*100:.2f}%)"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_file = None

    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            result = predict_mask(save_path)
            img_file = "static/uploads/" + file.filename

    return render_template("index.html", result=result, img_path=img_file)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)