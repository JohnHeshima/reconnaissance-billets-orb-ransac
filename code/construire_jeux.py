"""Construit les jeux d'images simulées (validation, test, apprentissage de la variante B).

Usage : python construire_jeux.py --refs ../donnees/billets_publics --sortie ../donnees/synthetique

Le découpage des billets est fixé dans `config_billets.py` :
  * CONNUS          : coupures que le système doit reconnaître (une référence chacune) ;
  * NEG_CALIBRATION : autres billets, vus seulement pendant le réglage des seuils ;
  * NEG_TEST        : autres billets jamais vus, réservés au test final (monde ouvert).
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from billets.pretraitement import charger_image
from billets.synthese import CATEGORIES, generer_photo
from config_billets import CONNUS, NEG_CALIBRATION, NEG_TEST

# nombre de photos par (billet connu, catégorie) / par (billet négatif, catégorie) / scènes vides
PLAN = {
    "validation": {"graine": 101, "pos": 2, "neg": 1, "vides": 30, "negatifs": NEG_CALIBRATION},
    "test": {"graine": 202, "pos": 4, "neg": 1, "vides": 50, "negatifs": NEG_TEST},
    "apprentissage_b": {"graine": 303, "pos": 6, "neg": 2, "vides": 30, "negatifs": NEG_CALIBRATION},
}


def construire(refs_dir, sortie, jeux=None):
    refs_dir, sortie = Path(refs_dir), Path(sortie)
    cache = {}

    def billet(nom):
        if nom not in cache:
            cache[nom] = charger_image(refs_dir / f"{nom}.jpg")
        return cache[nom]

    for jeu, p in PLAN.items():
        if jeux and jeu not in jeux:
            continue
        rng = np.random.default_rng(p["graine"])
        dossier = sortie / jeu
        dossier.mkdir(parents=True, exist_ok=True)
        lignes = []

        def ecrire(img, classe, categorie, quad, source, distracteur):
            nom = f"{len(lignes):04d}_{classe}_{categorie}.jpg"
            cv2.imwrite(str(dossier / nom), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lignes.append({"fichier": nom, "classe": classe, "categorie": categorie,
                           "source": source, "distracteur": distracteur or "",
                           "quad": json.dumps(quad.round(1).tolist()) if quad is not None else ""})

        for classe in CONNUS:
            for cat in CATEGORIES:
                for _ in range(p["pos"]):
                    d = str(rng.choice(p["negatifs"])) if rng.random() < 0.25 else None
                    img, quad = generer_photo(rng, billet(classe), cat, billet(d) if d else None)
                    ecrire(img, classe, cat, quad, classe, d)
        for nom in p["negatifs"]:
            for cat in CATEGORIES:
                for _ in range(p["neg"]):
                    img, quad = generer_photo(rng, billet(nom), cat)
                    ecrire(img, "inconnu", cat, quad, nom, None)
        for _ in range(p["vides"]):
            cat = str(rng.choice(["facile", "eclairage", "flou", "compression"]))
            img, _ = generer_photo(rng, None, cat)
            ecrire(img, "inconnu", cat, None, "scene_vide", None)

        with open(dossier / "manifeste.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
            w.writeheader()
            w.writerows(lignes)
        print(f"{jeu}: {len(lignes)} images -> {dossier}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="../donnees/billets_publics")
    ap.add_argument("--sortie", default="../donnees/synthetique")
    ap.add_argument("--jeux", nargs="*")
    a = ap.parse_args()
    construire(a.refs, a.sortie, a.jeux)
