# predict_cnn.py
#
# Predicción de motivos usando una CNN preentrenada
# Modelo proporcionado por el profesor: MyCNN.h5
#
# Autor: (tu nombre)
# Fecha: diciembre 2025

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Parámetros
# -----------------------------
MODEL_PATH = "MyCNN_finetuned.h5"
TEST_DIR = "Motifs_test"

IMG_SIZE = (120, 120)
BATCH_SIZE = 32

# -----------------------------
# Cargar modelo
# -----------------------------
print("Cargando modelo...")
model = load_model(MODEL_PATH)
model.summary()

# -----------------------------
# Generador de datos (solo reescalado)
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# Evaluación del modelo
# -----------------------------
print("\nEvaluando modelo sobre el conjunto de test...")
loss, accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nAccuracy en test: {accuracy:.4f}")

# -----------------------------
# Predicciones
# -----------------------------
print("\nGenerando predicciones...")
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# -----------------------------
# Resultados
# -----------------------------
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred_classes))
