# inspect_model.py
from tensorflow.keras.models import load_model

# 1. Cargar el modelo
model_path = "MyCNN.h5"
model = load_model(model_path)

# 2. Información básica
print("=== INPUT SHAPE ===")
print(model.input_shape)  # e.g., (None, 224, 224, 3)

print("\n=== OUTPUT SHAPE ===")
print(model.output_shape)  # e.g., (None, 10) -> 10 clases

print("\n=== SUMMARY ===")
model.summary()  # Detalle de todas las capas

# 3. Extra: número de clases
num_classes = model.output_shape[-1]
print(f"\nNúmero de clases que predice el modelo: {num_classes}")
