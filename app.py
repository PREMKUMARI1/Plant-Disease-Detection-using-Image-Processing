import os
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
REMEDIES_DIR = os.path.join(BASE_DIR, "remedies")
CLASSES_DIR = os.path.join(BASE_DIR, "class_maps")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Crop-to-file mapping
MODEL_FILES = {
    "potato": "potato_cnn.h5",
    "chilli": "chilli_cnn.h5",
    "maize": "maize_cnn.h5"
}

REMEDY_FILES = {
    "potato": "potato_remedies.json",
    "chilli": "chilli_remedies.json",
    "maize": "maize_remedies.json"
}

CLASS_FILES = {
    "potato": "potato_classes.json",
    "chilli": "chilli_classes.json",
    "maize": "maize_classes.json"
}


# Cache so models load only once
model_cache = {}
class_cache = {}
remedy_cache = {}


def load_crop_model(crop):
    if crop not in model_cache:
        path = os.path.join(MODELS_DIR, MODEL_FILES[crop])
        model_cache[crop] = load_model(path, compile=False)
    return model_cache[crop]


def load_classes(crop):
    if crop not in class_cache:
        path = os.path.join(CLASSES_DIR, CLASS_FILES[crop])
        with open(path, "r") as f:
            class_cache[crop] = json.load(f)
    return class_cache[crop]


def load_remedies(crop):
    if crop not in remedy_cache:
        path = os.path.join(REMEDIES_DIR, REMEDY_FILES[crop])
        with open(path, "r") as f:
            remedy_cache[crop] = json.load(f)
    return remedy_cache[crop]


def predict_image(crop, img_path):
    model = load_crop_model(crop)
    classes = load_classes(crop)
    remedies = load_remedies(crop)

    _, H, W, _ = model.input_shape

    img = image.load_img(img_path, target_size=(H, W))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    conf = float(np.max(preds[0]))
    cls = classes[idx]

    if conf < 0.50:
        severity = "Low"
    elif conf < 0.80:
        severity = "Medium"
    else:
        severity = "High"

    return cls, round(conf * 100, 2), severity, remedies.get(cls, [])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    crop = request.form.get("crop")

    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="Please select an image")

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    disease, confidence, severity, remedy_list = predict_image(crop, path)

    return render_template(
        "index.html",
        crop=crop,
        disease=disease,
        confidence=confidence,
        severity=severity,
        remedies=remedy_list,
        image_path=path.replace(BASE_DIR, "")
    )


if __name__ == "__main__":
    app.run(debug=True)

