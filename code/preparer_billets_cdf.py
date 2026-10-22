"""Prépare les images officielles des billets en francs congolais (site de la BCC).

Source : Banque Centrale du Congo, www.bcc.cd/la-banque/billets-et-monnaie/
(pages « Reconnaître les billets » et « Connaître le franc congolais »), images
hébergées sur resources.bcc.cd/images/billets/. Reproduction autorisée à des fins
d'étude et de recherche avec citation de la source ; usage commercial interdit.

Le site donne, pour chaque coupure, une image « propre » d'une seule face et deux
images (recto et verso) annotées de pastilles bleues numérotées qui signalent les
éléments de sécurité. Ce script :
  1. recadre chaque image sur le billet (marges blanches, liseré d'une autre image,
     cadre noir du 20 000 FC) ;
  2. efface les pastilles des images annotées (détection par couleur et forme,
     puis reconstruction par inpainting de Telea) ;
  3. écrit deux dossiers :
       references/<coupure>_<face>.jpg : l'image de référence de chaque face, la
                 version propre quand le site en donne une, sinon la version nettoyée ;
       sources/<coupure>_<face>.jpg    : l'image à partir de laquelle on simule les
                 photos, la version nettoyée quand elle existe, pour que la photo ne
                 vienne pas du même fichier que la référence.

Usage : python preparer_billets_cdf.py --brut ../donnees/bcc_brut --sortie ../donnees/billets_cdf
(--brut contient connus/ et autres/ avec les noms 500FC_recto.jpg, 500FC_verso_annote.jpg, ...)
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from billets.pretraitement import charger_image


def plus_longue_suite(vrai):
    meilleur, debut, cur = (0, 0), None, 0
    for i, v in enumerate(list(vrai) + [False]):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            if i - debut > meilleur[1] - meilleur[0]:
                meilleur = (debut, i)
            debut = None
    return meilleur


def recadrer(img):
    """Garde le plus grand bloc de lignes puis de colonnes non blanches, puis retire un cadre sombre."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    encre = (gris < 245) | (sat > 25)
    # un maximum glissant sur 15 px enjambe les fins filets blancs internes au billet
    lisse = lambda p: cv2.dilate(p.astype(np.float32).reshape(1, -1), np.ones((1, 15), np.uint8)).ravel()
    l0, l1 = plus_longue_suite(lisse(encre.mean(axis=1)) > 0.03)
    c0, c1 = plus_longue_suite(lisse(encre[l0:l1].mean(axis=0)) > 0.03)
    h, w = encre.shape  # on retire la marge ajoutée par le maximum glissant, sauf au bord de l'image
    l0, l1 = l0 + 7 * (l0 > 0), l1 - 7 * (l1 < h)
    c0, c1 = c0 + 7 * (c0 > 0), c1 - 7 * (c1 < w)
    img = img[l0:l1, c0:c1]
    for _ in range(8):  # cadre noir (20 000 FC) : bords nettement plus sombres que l'intérieur
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        bords = [g[0].mean(), g[-1].mean(), g[:, 0].mean(), g[:, -1].mean()]
        if min(bords) > 90:
            break
        img = img[1 if bords[0] <= 90 else 0: g.shape[0] - (1 if bords[1] <= 90 else 0),
                  1 if bords[2] <= 90 else 0: g.shape[1] - (1 if bords[3] <= 90 else 0)]
    return img


def masque_pastilles(img):
    """Pastilles bleu vif d'environ 43 px de diamètre (pour une image de 1 200 px de large)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bleu = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(bleu)
    masque = np.zeros(img.shape[:2], np.uint8)
    k = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 34 <= w <= 52 and 34 <= h <= 52 and a / (w * h) > 0.55:
            cv2.circle(masque, tuple(int(v) for v in cen[i]), int(max(w, h) / 2) + 4, 255, -1)
            k += 1
    return masque, k


def nettoyer(img):
    masque, k = masque_pastilles(img)
    return cv2.inpaint(img, masque, 7, cv2.INPAINT_TELEA), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brut", default="../donnees/bcc_brut")
    ap.add_argument("--sortie", default="../donnees/billets_cdf")
    a = ap.parse_args()
    brut, sortie = Path(a.brut), Path(a.sortie)
    (sortie / "references").mkdir(parents=True, exist_ok=True)
    (sortie / "sources").mkdir(parents=True, exist_ok=True)
    faces = {}
    for chemin in sorted(brut.glob("*/*.jpg")):
        coupure, face, *annote = chemin.stem.split("_")
        faces.setdefault((coupure, face), {})["annote" if annote else "propre"] = chemin
    for (coupure, face), versions in sorted(faces.items()):
        nom = f"{coupure}_{face}.jpg"
        propre = recadrer(charger_image(versions["propre"])) if "propre" in versions else None
        nettoyee = None
        if "annote" in versions:
            nettoyee, k = nettoyer(charger_image(versions["annote"]))
            nettoyee = recadrer(nettoyee)
            print(f"{versions['annote'].name}: {k} pastilles effacées")
        ref = propre if propre is not None else nettoyee
        src = nettoyee if nettoyee is not None else propre
        cv2.imwrite(str(sortie / "references" / nom), ref, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(sortie / "sources" / nom), src, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{nom}: référence {'propre' if propre is not None else 'nettoyée'} {ref.shape[1]}x{ref.shape[0]}, "
              f"source {'nettoyée' if nettoyee is not None else 'propre'} {src.shape[1]}x{src.shape[0]}")


if __name__ == "__main__":
    main()
