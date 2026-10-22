"""Prépare et évalue les photos réelles de billets congolais (photos prises au téléphone).

Organisation attendue des photos brutes (noms libres, un dossier par coupure) :
    photos/
      1000FC/  1000FC_recto_ref.jpg  1000FC_verso_ref.jpg  img_001.jpg ...
      5000FC/  ...
      negatifs/  dollar.jpg  recu.jpg ...
Une photo dont le nom contient « _ref » est une photo de référence (billet à
plat, fond uni) ; elle est recadrée et redressée automatiquement puis copiée
dans le dossier des références sous le nom <coupure>_<recto|verso>.jpg.
Toutes les autres photos servent de test.

Usage
  python preparer_photos_reelles.py preparer --photos ../donnees/photos --refs ../donnees/references
  python preparer_photos_reelles.py evaluer  --photos ../donnees/photos --refs ../donnees/references --tau 15
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from billets import INCONNU, ReconnaisseurORB, charger_image
from billets.evaluation import matrice_confusion, metriques
from billets.pretraitement import recadrer_billet

EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def preparer(photos, refs):
    refs.mkdir(parents=True, exist_ok=True)
    for dossier in sorted(p for p in photos.iterdir() if p.is_dir() and p.name != "negatifs"):
        for chemin in sorted(dossier.iterdir()):
            if chemin.suffix not in EXT or "_ref" not in chemin.stem:
                continue
            face = "verso" if "verso" in chemin.stem.lower() else "recto"
            img = charger_image(chemin)
            billet = recadrer_billet(img)
            if billet is None:
                print(f"!! recadrage impossible, image gardée telle quelle : {chemin.name}")
                billet = img
            cible = refs / f"{dossier.name}_{face}.jpg"
            cv2.imwrite(str(cible), billet, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{chemin.name} -> {cible.name} ({billet.shape[1]}x{billet.shape[0]})")


def evaluer(photos, refs, tau, rho, sortie):
    if not list(refs.glob("*.jpg")):
        refs = Path("../donnees/billets_cdf/references")
        print(f"Pas de références photographiées : on utilise les images de la BCC ({refs})")
    reco = ReconnaisseurORB(tau=tau, rho=rho, n_niveaux=12)
    for chemin in sorted(refs.glob("*.jpg")):
        classe, _, face = chemin.stem.rpartition("_")
        reco.ajouter_reference(classe, charger_image(chemin), face)
    classes = reco.classes
    lignes = []
    for dossier in sorted(p for p in photos.iterdir() if p.is_dir()):
        vrai = INCONNU if dossier.name == "negatifs" else dossier.name
        for chemin in sorted(dossier.iterdir()):
            if chemin.suffix not in EXT or "_ref" in chemin.stem:
                continue
            r = reco.predire(charger_image(chemin))
            lignes.append({"fichier": f"{dossier.name}/{chemin.name}", "vrai": vrai, "predit": r["classe"],
                           "face": r["face"], "inliers": r["score"], "second": r["second_score"],
                           "temps_ms": round(r["temps_ms"], 1)})
    y = [l["vrai"] for l in lignes]
    p = [l["predit"] for l in lignes]
    m = metriques(y, p, classes)
    cm, etiq = matrice_confusion(y, p, classes)
    sortie.mkdir(parents=True, exist_ok=True)
    with open(sortie / "photos_reelles_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)
    json.dump({"tau": tau, "rho": rho, "n_photos": len(lignes), "metriques": m, "etiquettes": etiq, "matrice": cm.tolist()},
              open(sortie / "photos_reelles_synthese.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["preparer", "evaluer"])
    ap.add_argument("--photos", default="../donnees/photos")
    ap.add_argument("--refs", default="../donnees/references")
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.26)
    ap.add_argument("--sortie", default="../resultats")
    a = ap.parse_args()
    if a.action == "preparer":
        preparer(Path(a.photos), Path(a.refs))
    else:
        evaluer(Path(a.photos), Path(a.refs), a.tau, a.rho, Path(a.sortie))
