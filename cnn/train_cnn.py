# train_cnn.py
#
# Reentrenamiento (fine-tuning) de una CNN para clasificación de motivos
# Se parte del modelo MyCNN.h5 proporcionado por el profesor
# Se utiliza data augmentation debido al tamaño reducido del dataset
#
# Autor: Raquel Montoliu y Ana Asenjo    Fecha: diciembre 2025

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# -----------------------------
# Parámetros
# -----------------------------
TRAIN_DIR = "Motifs_train"
OUTPUT_MODEL = "MyCNN_finetuned.h5"

IMG_SIZE = (120, 120)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# -----------------------------
# Cargar modelo base
# -----------------------------
print("Cargando modelo base...")
model = load_model("MyCNN.h5")
model.summary()

# -----------------------------
# Compilar modelo
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Generadores de datos
# -----------------------------
print("Preparando generadores de datos...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    OUTPUT_MODEL,
    monitor="accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -----------------------------
# Entrenamiento
# -----------------------------
print("Comenzando entrenamiento...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# Guardado final
# -----------------------------
model.save(OUTPUT_MODEL)
print(f"\nModelo entrenado guardado como: {OUTPUT_MODEL}")
