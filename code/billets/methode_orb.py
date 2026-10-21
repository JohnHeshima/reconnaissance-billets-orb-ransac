"""Variante A : ORB + appariement de Hamming + RANSAC (homographie).

Principe
--------
1. Pour chaque image de référence (une par coupure et par face), on extrait
   des points d'intérêt ORB (détecteur FAST orienté + descripteur BRIEF
   binaire de 256 bits, invariant en rotation, multi-échelle par pyramide).
2. Pour une photo à reconnaître, on extrait ses points ORB puis, pour chaque
   référence, on apparie les descripteurs par distance de Hamming
   (k = 2 plus proches voisins) et on filtre avec le test de ratio de Lowe.
3. Les appariements restants alimentent RANSAC, qui estime l'homographie
   référence -> photo et sépare les appariements cohérents (inliers) des
   faux appariements (outliers).
4. Le score d'une référence est son nombre d'inliers. La coupure retenue est
   celle de la meilleure référence ; si ce score est inférieur au seuil tau,
   la photo est rejetée (« inconnu » : pas un billet connu).
5. L'homographie projette le contour du billet de référence dans la photo,
   ce qui donne aussi sa localisation (évaluée par IoU).
"""

import time

import cv2
import numpy as np

from .pretraitement import preparer_gris, redimensionner

INCONNU = "inconnu"


class ReconnaisseurORB:
    def __init__(self, n_points=2000, ratio=0.75, seuil_ransac=5.0, tau=15,
                 clahe=True, cote_max_ref=800, cote_max_requete=1024,
                 controle_geometrique=True, n_niveaux=8, relance_zoom=False):
        self.n_points = n_points
        self.ratio = ratio
        self.seuil_ransac = seuil_ransac
        self.tau = tau
        self.clahe = clahe
        self.cote_max_ref = cote_max_ref
        self.cote_max_requete = cote_max_requete
        self.controle_geometrique = controle_geometrique
        self.relance_zoom = relance_zoom
        self.orb = cv2.ORB_create(nfeatures=n_points, scaleFactor=1.2, nlevels=n_niveaux,
                                  edgeThreshold=15, patchSize=31, fastThreshold=10)
        self.apparieur = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.references = []

    # ------------------------------------------------------------------ apprentissage
    def ajouter_reference(self, classe, img, face=""):
        """Enregistre une image de référence (billet seul, vu de face)."""
        petit, _ = redimensionner(img, self.cote_max_ref)
        gris = preparer_gris(petit, self.clahe)
        kp, des = self.orb.detectAndCompute(gris, None)
        if des is None or len(kp) < 10:
            raise ValueError(f"Trop peu de points ORB sur la référence {classe} {face}")
        h, w = gris.shape
        self.references.append({
            "classe": classe, "face": face, "kp": kp, "des": des,
            "coins": np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
            "image": petit,
        })
        return len(kp)

    @property
    def classes(self):
        return sorted({r["classe"] for r in self.references})

    # ------------------------------------------------------------------ reconnaissance
    def extraire(self, img, zoom=1.0):
        petit, f = redimensionner(img, self.cote_max_requete)
        if zoom != 1.0:
            petit = cv2.resize(petit, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            f *= zoom
        gris = preparer_gris(petit, self.clahe)
        kp, des = self.orb.detectAndCompute(gris, None)
        return kp, des, f, gris.shape

    def comparer(self, ref, kp, des, forme):
        """Appariement + RANSAC contre une référence. Renvoie (inliers, H, détails)."""
        if des is None or len(kp) < 4:
            return 0, None, {"bons": 0}
        paires = self.apparieur.knnMatch(ref["des"], des, k=2)
        bons = [m for m, *n in paires if n and m.distance < self.ratio * n[0].distance]
        if len(bons) < 4:
            return 0, None, {"bons": len(bons)}
        src = np.float32([ref["kp"][m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
        H, masque = cv2.findHomography(src, dst, cv2.RANSAC, self.seuil_ransac,
                                       maxIters=2000, confidence=0.995)
        if H is None:
            return 0, None, {"bons": len(bons)}
        n_in = int(masque.sum())
        if self.controle_geometrique and not homographie_plausible(H, ref["coins"], forme):
            return 0, None, {"bons": len(bons), "rejet_geom": True, "inliers_bruts": n_in}
        return n_in, H, {"bons": len(bons), "masque": masque.ravel().astype(bool), "appariements": bons}

    def predire(self, img, details=False):
        t0 = time.perf_counter()
        res = self._predire(img, 1.0, details)
        if self.relance_zoom and res["classe"] == INCONNU:
            # Billet peut-être trop petit dans la photo : on relance sur l'image
            # agrandie x2, ce qui fait apparaître des points ORB à une échelle
            # que la pyramide de la référence n'atteint pas.
            res2 = self._predire(img, 2.0, details)
            if res2["score"] > res["score"]:
                res = res2
        res["temps_ms"] = 1000 * (time.perf_counter() - t0)
        return res

    def _predire(self, img, zoom, details):
        kp, des, f, forme = self.extraire(img, zoom)
        scores = {}
        meilleur = (0, None, None, None)
        for ref in self.references:
            n_in, H, info = self.comparer(ref, kp, des, forme)
            cle = ref["classe"]
            if n_in > scores.get(cle, -1):
                scores[cle] = n_in
            if n_in > meilleur[0]:
                meilleur = (n_in, H, ref, info)
        n_in, H, ref, info = meilleur
        classe = ref["classe"] if (ref is not None and n_in >= self.tau) else INCONNU
        polygone = None
        if ref is not None and H is not None:
            polygone = cv2.perspectiveTransform(ref["coins"], H).reshape(-1, 2) / f
        tri = sorted(scores.values(), reverse=True) + [0, 0]
        res = {
            "classe": classe,
            "classe_brute": ref["classe"] if ref is not None else INCONNU,
            "face": ref["face"] if ref is not None else "",
            "score": n_in,
            "second_score": tri[1],
            "scores": scores,
            "polygone": polygone if classe != INCONNU else None,
            "polygone_brut": polygone,
            "n_points": len(kp),
            "zoom": zoom,
        }
        if details:
            res.update({"kp": kp, "facteur": f, "ref": ref, "H": H, "info": info})
        return res


def homographie_plausible(H, coins, forme):
    """Rejette les homographies dégénérées (billet retourné, écrasé, aberrant).

    Un vrai billet vu par une caméra donne un quadrilatère convexe, non
    retourné (déterminant positif), d'aire raisonnable et sans côtés
    dégénérés. Ces contrôles éliminent une partie des faux positifs où
    RANSAC trouve par hasard quelques inliers alignés.
    """
    if abs(np.linalg.det(H)) < 1e-6:
        return False
    if np.linalg.det(H[:2, :2]) <= 0:
        return False
    quad = cv2.perspectiveTransform(coins, H).reshape(-1, 2)
    if not np.all(np.isfinite(quad)):
        return False
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False
    aire = cv2.contourArea(quad.astype(np.float32))
    aire_image = forme[0] * forme[1]
    if aire < 0.005 * aire_image or aire > 4.0 * aire_image:
        return False
    cotes = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    if cotes.min() < 0.1 * cotes.max():
        return False
    # Rapport largeur/hauteur projeté compatible avec celui de la référence
    w0, h0 = coins[2, 0]
    r0 = w0 / h0
    r = (cotes[0] + cotes[2]) / (cotes[1] + cotes[3] + 1e-9)
    if not (r0 / 3.5 < r < r0 * 3.5):
        return False
    return True
