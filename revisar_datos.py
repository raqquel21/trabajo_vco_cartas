# Este archivo se ha hecho para revisar si se han etiquetado bien las cartas despues de etiquetar_cartas.py

import numpy as np
from trabajo_vco import Card, Motif

# Cargamos el archivo que acabas de etiquetar
datos = np.load('trainCards.npz', allow_pickle=True)
cartas = datos['Cartas'] # esto es diferente (dentro de etiquetar_cartas hemos nombrado a la clase Cartas para distinguitlo del primer script)

print(f"Revisando {len(cartas)} cartas:\n")
for c in cartas:
    print(f"ID: {c.cardId} | Palo Real: {c.realSuit} | Figura Real: {c.realFigure}")
    for i, m in enumerate(c.motifs):
        print(f"  Motivo {i}: {m.label}")
