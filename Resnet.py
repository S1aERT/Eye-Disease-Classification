import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import zipfile
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from collections import Counter
from google.colab import drive
drive.mount('/content/drive')
ZIP_PATH = "/content/drive/MyDrive/Machine Learning/ml_dataset.zip"
IMG_SIZE = 224
BATCH_SIZE = 16
CLASSES = ['NORMAL', 'DRUSEN', 'DME', 'CNV']
EPOCHS = 5
def create_datasets(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        all_files = []

        for f in zip_file.namelist():
            if f.startswith('Project Exhibition 2 Dataset/train/') and \
               any(f'/{cls}/' in f for cls in CLASSES) and \
               f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(f)

        if len(all_files) == 0:
            raise ValueError("No image files found in the zip file.")

        np.random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        def get_label(file):
            for i, cls in enumerate(CLASSES):
                if f'/train/{cls}/' in file:
                    return i
            return 0

        train_labels = [get_label(f) for f in train_files]
        val_labels = [get_label(f) for f in val_files]

        # -------- Class Weights --------
        label_counts = Counter(train_labels)
        total_samples = len(train_labels)
        n_classes = len(CLASSES)

        class_weights = {
            i: total_samples / (n_classes * label_counts.get(i, 1))
            for i in range(n_classes)
        }

        # -------- Generator --------
        def generator(files, labels):
            with zipfile.ZipFile(zip_path, 'r') as zip_f:
                for file, label in zip(files, labels):
                    try:
                        with zip_f.open(file) as f:
                            img = Image.open(io.BytesIO(f.read()))
                            img = img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                            img = preprocess_input(np.array(img))
                        yield img, label
                    except Exception:
                        yield np.zeros((IMG_SIZE, IMG_SIZE, 3)), label

        train_ds = tf.data.Dataset.from_generator(
            lambda: generator(train_files, train_labels),
            output_signature=(
                tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
                tf.TensorSpec((), tf.int32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_generator(
            lambda: generator(val_files, val_labels),
            output_signature=(
                tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
                tf.TensorSpec((), tf.int32)
            )
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_weights, len(train_files), len(val_files)

train_ds, val_ds, class_weights, train_count, val_count = create_datasets(ZIP_PATH)

train_steps = max(1, train_count // BATCH_SIZE)
val_steps = max(1, val_count // BATCH_SIZE)

# -------------------------------
# RESNET-50 MODEL
# -------------------------------
base_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# CALLBACKS
# -------------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint(
        '/content/drive/MyDrive/Machine Learning/best_resnet50_model.h5',
        save_best_only=True
    )
]

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------
# EVALUATE
# -------------------------------
final_loss, final_acc = model.evaluate(val_ds)
print(f"\nFinal Validation Accuracy: {final_acc:.4f}")

# -------------------------------
# PLOTS
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Progression')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Progression')
plt.legend()

plt.show()

print("✅ ResNet-50 Training Complete!")
