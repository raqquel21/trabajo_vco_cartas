# train_cnn.py
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Configuración
TRAIN_DIR = "Motifs_train/"
VAL_DIR = "Motifs_val/"
IMG_SIZE = (120, 120)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "MyCNN_trained.h5"

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Definir modelo (ejemplo secuencial similar al original)
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(120,120,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(200, activation='relu'),
    Dense(18, activation='softmax')  # 18 clases
])

# Compilar
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Guardar modelo
model.save(MODEL_PATH)
print(f"Modelo entrenado guardado en {MODEL_PATH}")
