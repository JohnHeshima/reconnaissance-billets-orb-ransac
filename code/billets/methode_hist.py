"""Variante B (implémentée pour la comparaison) : histogrammes de couleur
HSV + descripteurs de texture LBP + k plus proches voisins.

Chaque image est résumée par un vecteur global :
  * histogramme 2D teinte x saturation (16 x 8 cases) + histogramme de la
    valeur (8 cases), en HSV pour séparer la couleur de l'éclairage ;
  * histogrammes LBP uniformes (P=8, R=1) et (P=16, R=2) sur les niveaux de
    gris, qui décrivent la micro-texture (guillochis, impression).
Chaque bloc est normalisé (somme = 1) puis passé à la racine carrée : la
distance euclidienne entre vecteurs devient alors la distance de Hellinger,
proche de celle de Bhattacharyya, adaptée à la comparaison d'histogrammes.
La décision est un vote des k plus proches voisins de la base
d'apprentissage ; si le plus proche voisin est trop loin (distance > delta),
l'image est rejetée (« inconnu »).
"""

import time

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier

from .pretraitement import redimensionner

INCONNU = "inconnu"


def descripteur_histogrammes(img, poids_couleur=1.0, poids_texture=1.0, cote_max=320):
    petit, _ = redimensionner(img, cote_max)
    hsv = cv2.cvtColor(petit, cv2.COLOR_BGR2HSV)
    h_s = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    gris = cv2.cvtColor(petit, cv2.COLOR_BGR2GRAY)
    blocs_texture = []
    for p, r in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gris, p, r, method="uniform")
        hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
        blocs_texture.append(hist.astype(np.float64))

    def norme(x):
        return np.sqrt(x / (x.sum() + 1e-12))

    couleur = np.concatenate([norme(h_s), norme(v)]) * poids_couleur
    texture = np.concatenate([norme(b) for b in blocs_texture]) * poids_texture
    return np.concatenate([couleur, texture]).astype(np.float32)


class ReconnaisseurHistogrammes:
    def __init__(self, k=3, delta=None, poids_couleur=1.0, poids_texture=1.0):
        self.k = k
        self.delta = delta
        self.poids_couleur = poids_couleur
        self.poids_texture = poids_texture
        self.knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")

    def decrire(self, img):
        return descripteur_histogrammes(img, self.poids_couleur, self.poids_texture)

    def entrainer(self, images, etiquettes):
        X = np.stack([self.decrire(im) for im in images])
        self.knn.fit(X, np.asarray(etiquettes))
        return self

    def entrainer_vecteurs(self, X, etiquettes):
        self.knn.fit(X, np.asarray(etiquettes))
        return self

    def predire_vecteur(self, x):
        dist, _ = self.knn.kneighbors(x.reshape(1, -1), n_neighbors=1)
        d = float(dist[0, 0])
        classe = str(self.knn.predict(x.reshape(1, -1))[0])
        if self.delta is not None and d > self.delta:
            classe = INCONNU
        return classe, d

    def predire(self, img, details=False):
        t0 = time.perf_counter()
        x = self.decrire(img)
        classe, d = self.predire_vecteur(x)
        return {"classe": classe, "distance": d, "temps_ms": 1000 * (time.perf_counter() - t0)}
