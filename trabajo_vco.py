#!/usr/bin/env python3
"""
extraer_cartas.py

Script académico para EXTRAER cartas y MOTIVOS, rotar las ROI,
y guardar todas las cartas con sus motivos en un fichero .npz
(para posterior etiquetado manual con 'etiquetar_cartas.py').

No realiza clasificación automática. Sigue la especificación del trabajo.
"""
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
from tkinter import filedialog, Tk

# -------------------------
# Ventanas OpenCV
# -------------------------
window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_labels = 'Labels_image'
window_roi = 'ROI_image'
window_rotated = 'ROI_rotated'
window_motivos = "Motifs_on_card"



# Umbral usado en el trabajo
GLOBAL_THRESHOLD = 184
# -------------------------
# Configuración de visualización
# -------------------------
SHOW_WINDOWS = True

# -------------------------
# Estructuras de datos
# -------------------------
class Card:
    # Suits. Palos de las cartas de póker
    DIAMONDS = 'Diamonds'   # Rombos
    SPADES   = 'Spades'     # Picas
    HEARTS   = 'Hearts'     # Corazones
    CLUBS    = 'Clubs'      # Tréboles

    # Figuras de las cartas
    FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K')

    def __init__(self):
        # Identificador
        self.cardId = 0

        # Etiquetas reales (verdad fundamental → a rellenar con etiquetar_cartas.py)
        self.realSuit = 'i'
        self.realFigure = 'i'

        # Etiquetas predichas (se rellenarán en la fase de clasificación)
        self.predictedSuit = ''
        self.predictedFigure = ''

        # Bounding Box de la carta en la imagen original
        bboxType = [('x', np.intc),('y', np.intc),
                    ('width', np.intc),('height', np.intc)]
        self.boundingBox = np.zeros(1, dtype=bboxType).view(np.recarray)

        # Ángulo de rotación (grados)
        self.angle = 0.0

        # ROI rotada (gris y color)
        self.grayImage = np.empty([0,0], dtype=np.uint8)
        self.colorImage = np.empty([0,0,3], dtype=np.uint8)

        # Lista de motivos extraídos dentro de la carta
        self.motifs = []

    def __repr__(self):
        rep = f"Card number: {self.cardId} -- Real Suit/Figure: {self.realSuit}/{self.realFigure} -- "
        rep += f"Predicted Suit/Figure: {self.predictedSuit}/{self.predictedFigure}"
        bb = f"Bounding Box: {self.boundingBox} Rect angle: {self.angle}"
        ims = f"Gray image: {self.grayImage.shape} Color image: {self.colorImage.shape}"
        return rep + "\n" + bb + "\n" + ims

class Motif:
    """
    Representa un motivo detectado dentro de una carta.
    Guardamos: area, boundingBox local en la ROI, contour (relativo a la ROI),
    label (por defecto 'Others' hasta etiquetado), image (recorte binario).
    """
    def __init__(self):
        self.area = 0
        self.boundingBox = None  # (x,y,w,h) relativo a la ROI
        self.contour = None      # coordenadas relativas a la ROI
        self.label = "Others"    # etiqueta a rellenar manualmente
        self.image = None        # recorte binario del motivo

    def __repr__(self):
        return f"Motif(area={self.area}, bbox={self.boundingBox}, label={self.label})"
    
# ---------------------------------------------------------
#         CLASIFICACIÓN DE PALOS
# ---------------------------------------------------------
def clasificar_motivo(motif, gray_img, color_img):

    x, y, w, h = motif.boundingBox
    mask = motif.image

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return "Others"

    hu = cv2.HuMoments(M).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu)+1e-9)
    hu1 = hu[0]

    crop_color = color_img[y:y+h, x:x+w]
    if crop_color.size == 0:
        return "Others"

    B,G,R,_ = cv2.mean(crop_color, mask=mask)
    is_red = (R > G+40 and R > B+40)

    # Diamantes y corazones → rojos
    if is_red:
        if hu1 < 4.2:
            return "Diamonds"
        else:
            return "Hearts"
    else:
        # Picas y tréboles
        if hu1 < 4.5:
            return "Spades"
        else:
            return "Clubs"
# ---------------------------------------------------------
#         CLASIFICACIÓN DE FIGURAS (A,2..K)
# ---------------------------------------------------------
def clasificar_figura(motif):

    x, y, w, h = motif.boundingBox
    aspect = w/h if h>0 else 1

    M = cv2.moments(motif.image)
    if M["m00"] == 0:
        return "Others"

    hu = cv2.HuMoments(M).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu)+1e-9)
    h1 = hu[0]

    # Rangos aproximados por prueba real
    if h1 < 3.8:  return "0"
    if h1 < 4.1:  return "A"
    if h1 < 4.2:  return "2"
    if h1 < 4.32: return "3"
    if h1 < 4.42: return "4"
    if h1 < 4.55: return "5"
    if h1 < 4.72: return "6"
    if h1 < 4.85: return "7"
    if h1 < 5.0:  return "8"
    return "9"



# -------------------------
# Funciones utilitarias
# -------------------------
def label2rgb(label_img):
    if label_img.size == 0 or np.max(label_img) == 0:
        return np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
    label_hue = np.uint8(179 * (label_img) / np.max(label_img))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    colored = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    colored[label_img == 0] = 0
    return colored



def segmentar_objetos_carta(card):
    """
    Dado un objeto Card con card.grayImage (ROI rotada en gris),
    extrae los motivos como componentes conectadas de la
    imagen binaria obtenida por umbralización invertida.
    Guarda los Motif en card.motifs. NO clasifica motivos aquí.
    """
    roi = card.grayImage
    if roi is None or roi.size == 0:
        return

    # Umbralizado invertido: objetos (motivos) en blanco sobre negro
    _, BW = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)

    # Componentes conectadas
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(BW, 4, cv2.CV_32S)

    for i in range(1, nlabels):  # ignoramos 0 que es el fondo
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filtrado opcional: ignorar componentes muy pequeñas (ruido)
        # (Se deja pequeño umbral para no perder detalles de figuras)
        if area < 20:
            continue

        m = Motif()
        m.area = area
        m.boundingBox = (x, y, w, h)

        # Recorte del motivo (binario)
        try:
            m.image = BW[y:y+h, x:x+w].copy()
        except Exception:
            m.image = None

        # Máscara para obtener contorno relativo a la ROI (local)
        mask_local = (labels[y:y+h, x:x+w] == i).astype('uint8') * 255
        contours, _ = cv2.findContours(mask_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            m.contour = contours[0]

        # Clasificar palo o figura
        palo = clasificar_motivo(m, card.grayImage, card.colorImage)
        if palo in ("Diamonds","Spades","Hearts","Clubs"):
            m.label = palo
        else:
            m.label = clasificar_figura(m)

        card.motifs.append(m)
    # Información por consola
    print(f" → {len(card.motifs)} motivos detectados en carta {card.cardId}")

def dibujar_motivos(card):
    """
    Visualización sencilla de motivos sobre la ROI (contornos y cuadros).
    """
    if card.grayImage is None or card.grayImage.size == 0:
        return
    vis = cv2.cvtColor(card.grayImage, cv2.COLOR_GRAY2BGR)
    for m in card.motifs:
        if m.contour is not None:
            cv2.drawContours(vis, [m.contour], -1, (0,255,0), 1)
        if m.boundingBox is not None:
            x,y,w,h = m.boundingBox
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)
            # Mostrar etiqueta provisional (Others)
            cv2.putText(vis, m.label, (x, max(y-4,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    cv2.imshow(window_motivos, vis)

# -------------------------
# Main: proceso de imágenes
# -------------------------
def main():
    if SHOW_WINDOWS:
        cv2.namedWindow(window_original, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_threshold, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_labels, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_roi, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_rotated, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(window_motivos, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Pedimos carpeta al usuario (diálogo gráfico)
    root = Tk()
    root.withdraw()  # ocultar la ventana principal de Tk
    print("Seleccione la carpeta que contiene las imágenes (training o test).")
    base_path = filedialog.askdirectory(title="Seleccione carpeta con imágenes")
    root.destroy()

    if not base_path:
        print("No se seleccionó carpeta. Saliendo.")
        return

    # Pedimos si es training o test para nombrado de salida
    tipo = ''
    while tipo.lower() not in ('train', 'test'):
        tipo = input("¿Este conjunto es 'train' o 'test'? (escribe 'train' o 'test'): ").strip()
        if tipo == '':
            tipo = 'train'  # valor por defecto si el usuario presiona Enter

    cards = []
    icard = 0

    # Recorremos archivos de la carpeta (no recursivo)
    # Si quieres recursividad, cambia os.walk por os.listdir o similar.
    for root_dir, dirs, files in os.walk(base_path, topdown=True):
        # Procesar solo la carpeta principal (no subcarpetas por defecto):
        # Si prefieres procesar recursivamente, elimina este break.
        # break  # <-- comenta si quieres solo la carpeta principal
        for name in files:
            if not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue
            filename = os.path.join(root_dir, name)
            img = cv2.imread(filename)
            if img is None:
                print(f"No se pudo leer {filename}, saltando.")
                continue

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if SHOW_WINDOWS:
                cv2.imshow(window_original, img)

            # Umbralización global para segmentar cartas del fondo
            _, thresh_inv = cv2.threshold(img_gray, GLOBAL_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

            # Invertimos para que las cartas aparezcan en blanco (componentes)
            inv = 255 - thresh_inv

            # Componentes conectadas para encontrar objetos grandes (cartas)
            totalLabels, label_ids, stats, centroids = cv2.connectedComponentsWithStats(inv, 4, cv2.CV_32S)

            # Imagen para visualizar máscaras de cartas detectadas
            output_all = np.zeros(img_gray.shape, dtype='uint8')

            for i in range(1, totalLabels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                # Filtrar por área (umbral grande para coger sólo cartas)
                if area < 300000:
                    continue

                x1 = int(stats[i, cv2.CC_STAT_LEFT])
                y1 = int(stats[i, cv2.CC_STAT_TOP])
                w  = int(stats[i, cv2.CC_STAT_WIDTH])
                h  = int(stats[i, cv2.CC_STAT_HEIGHT])

                componentMask = (label_ids == i).astype('uint8') * 255
                output_all = cv2.bitwise_or(output_all, componentMask)

                # Recortamos ROI en gris y color (local)
                componentMask_cropped = componentMask[y1:y1+h, x1:x1+w].copy()
                roi_gray  = img_gray[y1:y1+h, x1:x1+w].copy()
                roi_color = img[y1:y1+h, x1:x1+w].copy()

                # Contornos en la máscara recortada
                contours, _ = cv2.findContours(componentMask_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    print(f"No se encontraron contornos en componente {i}.")
                    continue

                cnt = contours[0]

                # Rectángulo mínimo rotado
                minRect = cv2.minAreaRect(cnt)   # ((cx,cy),(w,h), angle)
                ((cx, cy), (rw, rh), rect_angle) = minRect

                # Corregir el ángulo (convención OpenCV)
                angle = rect_angle
                if angle < -45:
                    angle += 90.0

                # Rotación local para poner la carta "vertical"
                (h_roi, w_roi) = roi_color.shape[:2]
                rot_center = (w_roi // 2, h_roi // 2)
                M = cv2.getRotationMatrix2D(rot_center, angle, 1.0)

                rotated_color = cv2.warpAffine(roi_color, M, (w_roi, h_roi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                rotated_gray  = cv2.warpAffine(roi_gray,  M, (w_roi, h_roi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

                # Mostrar ROI y ROI rotada para comprobación
                cv2.imshow(window_roi, roi_color)
                cv2.imshow(window_rotated, rotated_color)

                # Crear objeto Card y almacenar datos básicos
                c = Card()
                c.cardId = icard
                c.boundingBox.x = x1
                c.boundingBox.y = y1
                c.boundingBox.width = w
                c.boundingBox.height = h
                c.angle = float(angle)
                c.grayImage = rotated_gray
                c.colorImage = rotated_color

                # Extraer motivos dentro de la carta (no clasifica -> solo guarda)
                segmentar_objetos_carta(c)

                # Mostrar motivos sobre la ROI rotada
                dibujar_motivos(c)

                # Guardar la carta en la lista
                cards.append(c)
                icard += 1

                print(f"[{icard}] Carta guardada: id={c.cardId}, area={area}, angle={c.angle:.2f} deg, motivos={len(c.motifs)}")

            # Mostrar etiquetas coloreadas y máscara de cartas
            cv2.imshow(window_threshold, output_all)
            if 'label_ids' in locals():
                cv2.imshow(window_labels, label2rgb(label_ids))
            
            if SHOW_WINDOWS:
            # Espera a tecla para continuar (no bloqueante: espera a que se pulse una tecla)
                print("Pulsa 'n' para siguiente imagen, 'q' o ESC para salir.")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    # n -> siguiente imagen; q/ESC -> salir
                    if key == ord('n'):
                        break
                    if key == ord('q') or key == 27:
                        print("Salida solicitada por el usuario.")
                        # Guardar lo que llevemos y salir de bucle superior
                        root_save = base_path
                        save_name = 'trainCards.npz' if tipo.lower()=='train' else 'testCards.npz'
                        save_path = os.path.join(os.getcwd(), save_name)
                        np.savez_compressed(save_path, Cards=cards)
                        print(f"Guardado parcial en: {save_path} (cartas={len(cards)})")
                        cv2.destroyAllWindows()
                        return
            else:
                pass

        # Si quieres NO recorrer subcarpetas (solo la principal), descomenta la siguiente línea:
        break

    # Guardar fichero final
    save_name = 'trainCards.npz' if tipo.lower()=='train' else 'testCards.npz'
    save_path = os.path.join(os.getcwd(), save_name)
    np.savez_compressed(save_path, Cards=cards)
    print(f"\nGuardado final: {save_path}  (total cartas = {len(cards)})")
    
    if SHOW_WINDOWS:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
