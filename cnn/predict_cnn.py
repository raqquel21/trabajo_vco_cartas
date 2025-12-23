# predict_cnn.py - versión corregida
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
MODEL_PATH = "MyCNN.h5"
TEST_DIR = "Motifs_test/"
IMG_SIZE = (120, 120)
BATCH_SIZE = 32

# Lista de clases que el modelo conoce (18 clases)
class_labels = [
    "000", "002", "003", "004", "005", "006", "007", "008",
    "009", "00A", "00J", "00K", "00Q", "corazones", "picas",
    "rombos", "treboles", "variados"
]

# Cargar modelo
model = load_model(MODEL_PATH)

# Crear generador solo para test
test_datagen = ImageDataGenerator(rescale=1./255)

# Filtrar las subcarpetas de test para que coincidan con class_labels
test_subdirs = [d for d in os.listdir(TEST_DIR) if d in class_labels]

# Crear un directorio temporal virtual con solo estas carpetas
# (Keras no necesita moverlas; solo pasamos subset)
# El truco: flow_from_directory con subset de clases
# Alternativa: eliminar carpetas extra temporalmente o crear generator por carpeta

# Generador por carpeta usando class_mode='categorical'
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_labels,
    class_mode='categorical',
    shuffle=False
)

# Predicción
preds = model.predict(test_generator, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes

# Accuracy global
accuracy = accuracy_score(true_classes, pred_classes)
print(f"\n=== ACCURACY GLOBAL ===\nAccuracy: {accuracy:.4f}")

# Matriz de confusión
print("\n=== MATRIZ DE CONFUSIÓN ===")
cm = confusion_matrix(true_classes, pred_classes)
print(cm)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Dibujar la matriz con colores
plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Matriz de Confusión Normalizada")
plt.xlabel("Predicción")
plt.ylabel("Clase Real")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Reporte por clase
print("\n=== REPORTE POR CLASE ===")
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print(report)
