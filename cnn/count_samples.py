# count_samples.py
import os
import matplotlib.pyplot as plt

# Carpeta de entrenamiento
TRAIN_DIR = "./Motifs_train"

# Listar las clases (subcarpetas)
classes = sorted(os.listdir(TRAIN_DIR))

# Contar número de imágenes por clase
counts = []
for c in classes:
    class_path = os.path.join(TRAIN_DIR, c)
    num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    counts.append(num_images)

# Mostrar gráfico
plt.figure(figsize=(12,6))
plt.bar(classes, counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Clases")
plt.ylabel("Número de muestras")
plt.title("Número de imágenes por clase en el dataset de entrenamiento")
plt.tight_layout()
plt.show()
