from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from PIL import Image
import webbrowser
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Paths ---
resnet_path = os.path.join(BASE_DIR, "best_resnet50_model.h5")
efficientnet_path = os.path.join(BASE_DIR, "EfficientNETB3.h5")
densenet_path = os.path.join(BASE_DIR, "best_densenet121_model.h5")

print("🔹 Loading models, please wait...")

# --- Load ResNet50 (RGB) ---
resnet_model = tf.keras.models.load_model(resnet_path)
print("✅ ResNet50 model loaded.")

# --- Load DenseNet121 (RGB) ---
densenet_model = tf.keras.models.load_model(densenet_path)
print("✅ DenseNet121 model loaded.")

# --- Load EfficientNetB3 (Grayscale-safe) ---
try:
    efficientnet_model = tf.keras.models.load_model(efficientnet_path)
    print("✅ EfficientNetB3 loaded normally.")
except ValueError:
    print("⚠️ Shape mismatch detected. Rebuilding EfficientNetB3 for grayscale input...")
    input_tensor = Input(shape=(224, 224, 1))
    base_model = EfficientNetB3(include_top=False, weights=None, input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(4, activation='softmax')(x)
    efficientnet_model = Model(inputs=base_model.input, outputs=x)
    efficientnet_model.load_weights(efficientnet_path, by_name=True, skip_mismatch=True)
    print("✅ EfficientNetB3 loaded with grayscale input safely!")

print("✅ All models loaded successfully!\n")

# Eye disease labels
CATEGORIES = ['NORMAL', 'DRUSEN', 'DME', 'CNV']
IMG_SIZE = 224


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    resnet_pred = None
    efficientnet_pred = None
    densenet_pred = None

    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file)

        # ---------- For ResNet50 (RGB) ----------
        rgb_image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        rgb_array = np.expand_dims(np.array(rgb_image) / 255.0, axis=0)
        resnet_prediction = resnet_model.predict(rgb_array)
        resnet_pred = CATEGORIES[np.argmax(resnet_prediction)]

        # ---------- For DenseNet121 (RGB) ----------
        densenet_prediction = densenet_model.predict(rgb_array)
        densenet_pred = CATEGORIES[np.argmax(densenet_prediction)]

        # ---------- For EfficientNetB3 (Grayscale) ----------
        gray_image = image.convert("L").resize((IMG_SIZE, IMG_SIZE))
        gray_array = np.expand_dims(np.array(gray_image) / 255.0, axis=(0, -1))
        efficientnet_prediction = efficientnet_model.predict(gray_array)
        efficientnet_pred = CATEGORIES[np.argmax(efficientnet_prediction)]

        return render_template(
            'index.html',
            resnet_pred=resnet_pred,
            efficientnet_pred=efficientnet_pred,
            densenet_pred=densenet_pred
        )

    return render_template(
        'index.html',
        resnet_pred=None,
        efficientnet_pred=None,
        densenet_pred=None
    )


if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"Server running! Click here to open: {url}\n")
    webbrowser.open(url)
    app.run(debug=True, port=port)
