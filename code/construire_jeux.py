"""Construit les jeux d'images simulées (validation, test, apprentissage de la variante B).

Usage : python construire_jeux.py --donnees ../donnees --sortie ../donnees/synthetique

Le découpage des billets est fixé dans `config_billets.py` :
  * CONNUS          : coupures en francs congolais à reconnaître, recto et verso ;
  * NEG_CALIBRATION : billets inconnus vus seulement pendant le réglage des seuils
                      (50 FC, de même maquette, et 10 billets étrangers) ;
  * NEG_TEST        : billets inconnus jamais vus, réservés au test final
                      (100 FC et 200 FC, de même maquette, et 34 billets étrangers).
Les photos des billets congolais sont simulées à partir des images « sources »,
distinctes des images de référence chaque fois que la BCC en publie deux versions.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from billets.pretraitement import charger_image
from billets.synthese import CATEGORIES, generer_photo
from config_billets import CONNUS, FACES, NEG_CALIBRATION, NEG_TEST, chemin_source, est_congolais

# nombre de photos par (face connue, catégorie) / par (billet étranger, catégorie) /
# par (face d'une coupure congolaise inconnue, catégorie) / scènes vides
PLAN = {
    "validation": {"graine": 101, "pos": 3, "neg": 1, "cousins": 3, "vides": 30, "negatifs": NEG_CALIBRATION},
    "test": {"graine": 202, "pos": 5, "neg": 1, "cousins": 3, "vides": 50, "negatifs": NEG_TEST},
    "apprentissage_b": {"graine": 303, "pos": 8, "neg": 2, "cousins": 4, "vides": 30, "negatifs": NEG_CALIBRATION},
}


def construire(donnees, sortie, jeux=None):
    sortie = Path(sortie)
    cache = {}

    def billet(nom):
        if nom not in cache:
            cache[nom] = charger_image(chemin_source(donnees, nom))
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
            for face in FACES:
                source = f"{classe}_{face}"
                for cat in CATEGORIES:
                    for _ in range(p["pos"]):
                        d = str(rng.choice(p["negatifs"])) if rng.random() < 0.25 else None
                        img, quad = generer_photo(rng, billet(source), cat, billet(d) if d else None)
                        ecrire(img, classe, cat, quad, source, d)
        for nom in p["negatifs"]:
            n = p["cousins"] if est_congolais(nom) else p["neg"]
            for cat in CATEGORIES:
                for _ in range(n):
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
    ap.add_argument("--donnees", default="../donnees")
    ap.add_argument("--sortie", default="../donnees/synthetique")
    ap.add_argument("--jeux", nargs="*")
    a = ap.parse_args()
    construire(a.donnees, a.sortie, a.jeux)
