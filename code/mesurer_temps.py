"""Mesure les temps de calcul sur un seul cœur (les expériences tournent en parallèle sur 4 processus).

Usage : python mesurer_temps.py --donnees ../donnees --jeux ../donnees/synthetique --resultats ../resultats
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from billets import ReconnaisseurORB, charger_image
from billets.methode_hist import descripteur_histogrammes
from config_billets import charger_references
from experiences import lire_manifeste


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="../donnees")
    ap.add_argument("--jeux", default="../donnees/synthetique")
    ap.add_argument("--resultats", default="../resultats")
    ap.add_argument("-n", type=int, default=150)
    a = ap.parse_args()
    res = Path(a.resultats)
    syn = json.load(open(res / "synthese.json", encoding="utf-8"))
    params = dict(syn["A"]["parametres"])
    test = lire_manifeste(Path(a.jeux) / "test")
    rng = np.random.default_rng(0)
    images = [charger_image(Path(a.jeux) / "test" / test[i]["fichier"])
              for i in rng.choice(len(test), a.n, replace=False)]

    reco = charger_references(ReconnaisseurORB(**params), a.donnees, charger_image)
    sans_zoom = charger_references(ReconnaisseurORB(**{**params, "relance_zoom": False}), a.donnees, charger_image)
    reco.predire(images[0])  # échauffement
    t_a = [reco.predire(img)["temps_ms"] for img in images]
    t_sz = [sans_zoom.predire(img)["temps_ms"] for img in images]
    t_orb, t_b = [], []
    pb = syn["B"]
    pc, pt = {"couleur + texture": (1, 1), "couleur seule": (1, 0), "texture seule": (0, 1)}[pb["descripteurs"]]
    for img in images[:60]:
        t0 = time.perf_counter()
        reco.extraire(img)
        t_orb.append(1000 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        descripteur_histogrammes(img, pc, pt)
        t_b.append(1000 * (time.perf_counter() - t0))
    out = {"A_moyen": float(np.mean(t_a)), "A_median": float(np.median(t_a)), "A_p95": float(np.percentile(t_a, 95)),
           "A_sans_zoom_moyen": float(np.mean(t_sz)), "A_extraction_orb": float(np.mean(t_orb)),
           "B_descripteur": float(np.mean(t_b)), "n": a.n}
    json.dump(out, open(res / "temps_un_coeur.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
