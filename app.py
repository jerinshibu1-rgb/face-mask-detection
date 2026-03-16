from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import urllib.request
from tensorflow.keras.preprocessing import image

# ===== MODEL DOWNLOAD FROM HUGGINGFACE =====
MODEL_URL = "https://huggingface.co/jerin96/face-mask-model/resolve/main/cnn_facemask.keras"
MODEL_PATH = "cnn_facemask.keras"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from HuggingFace...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ===== FLASK APP =====
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_mask(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        confidence = pred * 100
        return f"No Mask ({confidence:.2f}%)"
    else:
        confidence = (1 - pred) * 100
        return f"Mask ({confidence:.2f}%)"


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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))