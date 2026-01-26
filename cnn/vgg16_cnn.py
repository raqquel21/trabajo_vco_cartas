# vgg16_fast_cnn.py
#
# Entrenamiento rápido de VGG16 usando fine-tuning solo de las últimas capas
# Autor: Raquel Montoliu y Ana Asenjo    Fecha: diciembre 2025

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Parámetros
# -----------------------------
TRAIN_DIR = "Motifs_train"
TEST_DIR = "Motifs_test"
OUTPUT_MODEL = "VGG16_fast.h5"

IMG_SIZE = (120, 120)
BATCH_SIZE = 32
EPOCHS = 10          # Menos épocas para entrenar rápido
LEARNING_RATE = 1e-4

# -----------------------------
# Data augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_generator.num_classes

# -----------------------------
# Crear modelo VGG16 (congelar base)
# -----------------------------
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
for layer in base_model.layers:
    layer.trainable = False   # Congelar todas las capas base

# Añadir capas finales entrenables
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# Compilar modelo
# -----------------------------
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Entrenamiento
# -----------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)

# Guardar modelo
model.save(OUTPUT_MODEL)
print(f"Modelo VGG16 rápido guardado como {OUTPUT_MODEL}")

# -----------------------------
# Visualización métricas
# -----------------------------
plt.style.use("ggplot")
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Training Loss - VGG16 Fast")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Training Accuracy - VGG16 Fast")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# -----------------------------
#
