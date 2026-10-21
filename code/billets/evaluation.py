"""Métriques d'évaluation en monde ouvert (classes connues + rejet « inconnu »)."""

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

INCONNU = "inconnu"


def iou_quadrilateres(p, q):
    """IoU entre deux quadrilatères convexes (localisation du billet)."""
    if p is None or q is None:
        return 0.0
    p = np.asarray(p, np.float32).reshape(-1, 2)
    q = np.asarray(q, np.float32).reshape(-1, 2)
    if not (cv2.isContourConvex(p) and cv2.isContourConvex(q)):
        p, q = cv2.convexHull(p).reshape(-1, 2), cv2.convexHull(q).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(p, q)
    union = cv2.contourArea(p) + cv2.contourArea(q) - inter
    return float(inter / union) if union > 0 else 0.0


def metriques(y_vrai, y_pred, classes_connues):
    """Exactitude, P/R/F1 macro sur les classes connues, taux de fausse acceptation."""
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    pos = y_vrai != INCONNU
    neg = ~pos
    p, r, f, _ = precision_recall_fscore_support(y_vrai, y_pred, labels=list(classes_connues),
                                                 average="macro", zero_division=0)
    return {
        "exactitude": float((y_vrai == y_pred).mean()),
        "precision_macro": float(p),
        "rappel_macro": float(r),
        "f1_macro": float(f),
        # positifs : billet connu bien identifié
        "taux_identification": float((y_pred[pos] == y_vrai[pos]).mean()) if pos.any() else np.nan,
        # positifs confondus avec une autre coupure (erreur la plus grave)
        "taux_confusion": float(((y_pred[pos] != y_vrai[pos]) & (y_pred[pos] != INCONNU)).mean()) if pos.any() else np.nan,
        # positifs rejetés à tort
        "taux_faux_rejet": float((y_pred[pos] == INCONNU).mean()) if pos.any() else np.nan,
        # négatifs acceptés comme un billet connu
        "taux_fausse_acceptation": float((y_pred[neg] != INCONNU).mean()) if neg.any() else np.nan,
    }


def matrice_confusion(y_vrai, y_pred, classes_connues):
    etiquettes = list(classes_connues) + [INCONNU]
    return confusion_matrix(y_vrai, y_pred, labels=etiquettes), etiquettes


def decision_seuil(classes_brutes, scores, tau, plus_grand_mieux=True):
    """Applique a posteriori un seuil de rejet sur des scores déjà calculés."""
    classes_brutes, scores = np.asarray(classes_brutes, dtype=object), np.asarray(scores, float)
    ok = scores >= tau if plus_grand_mieux else scores <= tau
    return np.where(ok, classes_brutes, INCONNU)
