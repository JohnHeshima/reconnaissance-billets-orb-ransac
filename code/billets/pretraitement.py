"""Prétraitements communs aux deux méthodes."""

import cv2
import numpy as np


def charger_image(chemin):
    """Lit une image couleur (BGR), y compris si le chemin contient des accents."""
    donnees = np.fromfile(str(chemin), dtype=np.uint8)
    img = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Image illisible : {chemin}")
    return img


def redimensionner(img, cote_max):
    """Réduit l'image pour que son plus grand côté vaille au plus `cote_max` pixels.

    Renvoie l'image et le facteur d'échelle appliqué (utile pour remonter
    une homographie vers les coordonnées d'origine).
    """
    h, w = img.shape[:2]
    f = min(1.0, cote_max / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    return img, f


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preparer_gris(img, clahe=True):
    """Conversion en niveaux de gris + égalisation locale d'histogramme (CLAHE).

    La CLAHE compense en partie les éclairages faibles ou non uniformes,
    ce qui stabilise la détection des points FAST utilisés par ORB.
    """
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gris) if clahe else gris


def recadrer_billet(img, marge=0.0):
    """Détecte le plus grand quadrilatère de l'image et le redresse.

    Sert à préparer une photo de référence prise au téléphone (billet posé
    à plat sur un fond uni) : on obtient une image du billet seul, vu de face.
    Renvoie None si aucun quadrilatère plausible n'est trouvé.
    """
    petit, f = redimensionner(img, 1000)
    gris = cv2.GaussianBlur(cv2.cvtColor(petit, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bords = cv2.Canny(gris, 30, 100)
    bords = cv2.dilate(bords, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(bords, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * petit.shape[0] * petit.shape[1]:
        return None
    rect = cv2.minAreaRect(c)
    coins = cv2.boxPoints(rect) / f
    coins = _ordonner_coins(coins)
    largeur = int(max(np.linalg.norm(coins[0] - coins[1]), np.linalg.norm(coins[2] - coins[3])))
    hauteur = int(max(np.linalg.norm(coins[1] - coins[2]), np.linalg.norm(coins[3] - coins[0])))
    if hauteur > largeur:  # billet en portrait : on le remet en paysage
        coins = np.roll(coins, -1, axis=0)
        largeur, hauteur = hauteur, largeur
    dst = np.float32([[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]])
    H = cv2.getPerspectiveTransform(coins.astype(np.float32), dst)
    out = cv2.warpPerspective(img, H, (largeur, hauteur))
    if marge > 0:
        dy, dx = int(hauteur * marge), int(largeur * marge)
        out = out[dy:hauteur - dy, dx:largeur - dx]
    return out


def _ordonner_coins(p):
    """Ordonne 4 points : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    p = np.asarray(p, dtype=np.float32)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)
