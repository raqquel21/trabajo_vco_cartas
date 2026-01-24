import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from clases_cartas import Card, Motif


FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K')
SUITS = ('Rombos','Picas','Corazones','Treboles')
MOTIF_LABELS = (
    'Rombos','Picas','Corazones','Treboles',
    '0','2','3','4','5','6','7','8','9','A','J','Q','K','Others'
)

############################
# ====== TRAINING =========
############################

npzfile = np.load('trainCards.npz', allow_pickle=True)
cards = npzfile['Cartas']

samples = []
responses = []

for card in cards:
    for mot in card.motifs:
        lbl = mot.motifLabel
        if lbl == 'i':
            continue
        samples.append(mot.features)
        responses.append(MOTIF_LABELS.index(lbl))

sampl = np.asarray(samples, dtype=np.float32)
resp = np.asarray(responses, dtype=np.int32)

############################
# ==== NORMALIZACIÓN ======
############################

mean = sampl.mean(axis=0)
std = sampl.std(axis=0) + 1e-8

sampl_norm = (sampl - mean) / std

############################
# ========= kNN ===========
############################

knn = cv.ml.KNearest_create()
knn.setDefaultK(3)
knn.setIsClassifier(True)
knn.train(sampl_norm, cv.ml.ROW_SAMPLE, resp)

############################
# ========= SVM ===========
############################

svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setC(1.0)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
svm.train(sampl_norm, cv.ml.ROW_SAMPLE, resp)

############################
# ========= TEST ==========
############################

npzfileT = np.load('testCards.npz', allow_pickle=True)
cardsTest = npzfileT['Cartas']

samplesTest = []
responsesTest = []

for card in cardsTest:
    for mot in card.motifs:
        lbl = mot.motifLabel
        if lbl == 'i':
            lbl = 'Others'
        samplesTest.append(mot.features)
        responsesTest.append(MOTIF_LABELS.index(lbl))

samplTest = np.asarray(samplesTest, dtype=np.float32)
respTest = np.asarray(responsesTest, dtype=np.int32)

samplTest_norm = (samplTest - mean) / std

############################
# ===== PREDICCIÓN ========
############################

# kNN
_, res_knn, _, _ = knn.findNearest(samplTest_norm, k=3)
pred_knn = res_knn.flatten()

# SVM
_, res_svm = svm.predict(samplTest_norm)
pred_svm = res_svm.flatten()

############################
# ===== EVALUACIÓN ========
############################

def evaluar(nombre, real, pred):
    print(f"\n===== {nombre} =====")
    print("Accuracy:", np.mean(real == pred))
    print("MCC:", matthews_corrcoef(real, pred))
    print(classification_report(real, pred, target_names=MOTIF_LABELS))

    cm = confusion_matrix(real, pred)
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=MOTIF_LABELS)
    disp.plot(xticks_rotation='vertical')
    plt.title(nombre)
    plt.show()


evaluar("kNN ", respTest, pred_knn)
evaluar("SVM LINEAL", respTest, pred_svm)

############################
# ===== GUARDAR ===========
############################

knn.save('modelo_knn.yml')
svm.save('modelo_svm.yml')

