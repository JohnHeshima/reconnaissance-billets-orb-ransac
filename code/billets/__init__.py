"""Reconnaissance de billets de banque par méthodes classiques.

Sujet 39 - Variante A : ORB + appariement de Hamming + RANSAC (homographie)
par rapport à des images de référence.
Variante B (comparaison) : histogrammes de couleur HSV + texture LBP + k-NN.
"""

from .pretraitement import charger_image, preparer_gris, redimensionner
from .methode_orb import ReconnaisseurORB
from .methode_hist import ReconnaisseurHistogrammes

INCONNU = "inconnu"

__all__ = [
    "charger_image",
    "preparer_gris",
    "redimensionner",
    "ReconnaisseurORB",
    "ReconnaisseurHistogrammes",
    "INCONNU",
]
