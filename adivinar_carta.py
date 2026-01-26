import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# -----------------------------
# 1. Configuración
# -----------------------------
MODEL_PATH = "cnn/MyCNN_finetuned.h5"
IMAGE_PATH = "images/test/IMG_20210321_122121.jpg"
TRAIN_DIR = "cnn/Motifs_train"
IMG_SIZE = (120, 120)

# -----------------------------
# 2. Carga de Modelo y Clases
# -----------------------------
class_names = sorted(os.listdir(TRAIN_DIR))
model = load_model(MODEL_PATH)

# -----------------------------
# 3. Procesamiento de la Imagen Original
# -----------------------------
img_original = cv2.imread(IMAGE_PATH)
img_cv = cv2.resize(img_original, (800, 800)) # Un poco más grande para ver mejor
output_img = img_cv.copy()

# Detección de múltiples contornos
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Se han detectado {len(contours)} posibles objetos.")

# -----------------------------
# 4. Bucle para procesar cada carta
# -----------------------------
for c in contours:
    # Filtrar por área mínima para evitar ruido (ajusta el 5000 si no detecta algo)
    if cv2.contourArea(c) > 5000:
        x, y, w, h = cv2.boundingRect(c)
        
        # --- Recortar la carta detectada ---
        roi = img_cv[y:y+h, x:x+w]
        
        # --- Preprocesar el recorte para la CNN ---
        # Pasamos de OpenCV (BGR) a formato Keras (RGB)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, IMG_SIZE)
        roi_array = image.img_to_array(roi_resized) / 255.0
        roi_array = np.expand_dims(roi_array, axis=0)
        
        # --- Predicción Individual ---
        predictions = model.predict(roi_array, verbose=0)
        idx = np.argmax(predictions[0])
        pred_name = class_names[idx]
        confianza = predictions[0][idx] * 100
        
        # --- Dibujar en la imagen final ---
        color = (0, 0, 255) # Rojo
        cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 3)
        
        label = f"{pred_name} ({confianza:.1f}%)"
        cv2.putText(output_img, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# -----------------------------
# 5. Mostrar Resultado
# -----------------------------
cv2.imshow("Deteccion Multiple de Cartas", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()