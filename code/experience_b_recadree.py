"""Expérience complémentaire : variante B avec un recadrage idéal du billet.

La variante B décrit l'image entière ; sur une photo où le billet n'occupe qu'une
partie du cadre, le fond domine ses histogrammes. Pour séparer les deux causes
d'erreur (descripteur peu discriminant ou absence de localisation), on lui donne
ici le billet parfaitement recadré et redressé grâce à la vérité terrain : c'est
la meilleure performance qu'elle pourrait atteindre si la localisation était
résolue par ailleurs. Les scènes vides gardent l'image entière.
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from billets import ReconnaisseurHistogrammes, charger_image
from billets.evaluation import decision_seuil, matrice_confusion, metriques
from billets.methode_hist import descripteur_histogrammes
from config_billets import CONNUS
from experiences import lire_manifeste, meilleur_seuil


def recadrer(img, quad, largeur=600, hauteur=300):
    dst = np.float32([[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]])
    H = cv2.getPerspectiveTransform(np.float32(quad), dst)
    return cv2.warpPerspective(img, H, (largeur, hauteur))


def _desc(args):
    chemin, quad, pc, pt = args
    img = charger_image(chemin)
    if quad:
        img = recadrer(img, json.loads(quad))
    return descripteur_histogrammes(img, pc, pt)


def descripteurs(dossier, lignes, pc, pt, n_proc):
    taches = [(str(Path(dossier) / l["fichier"]), l["quad"], pc, pt) for l in lignes]
    with Pool(n_proc) as pool:
        return np.stack(pool.map(_desc, taches, chunksize=16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeux", default="../donnees/synthetique")
    ap.add_argument("--resultats", default="../resultats")
    ap.add_argument("--proc", type=int, default=4)
    a = ap.parse_args()
    jeux, res = Path(a.jeux), Path(a.resultats)
    val, test, app = (lire_manifeste(jeux / j) for j in ("validation", "test", "apprentissage_b"))
    y_val, y_test, y_app = (np.array([l["classe"] for l in x]) for x in (val, test, app))

    lignes, meilleur = [], None
    for nom, pc, pt in (("couleur + texture", 1.0, 1.0), ("couleur seule", 1.0, 0.0), ("texture seule", 0.0, 1.0)):
        X_app = descripteurs(jeux / "apprentissage_b", app, pc, pt, a.proc)
        X_val = descripteurs(jeux / "validation", val, pc, pt, a.proc)
        for k in (1, 3, 5, 7):
            rb = ReconnaisseurHistogrammes(k=k).entrainer_vecteurs(X_app, y_app)
            s = [rb.predire_vecteur(x) for x in X_val]
            brutes, dist = np.array([x[0] for x in s], dtype=object), np.array([x[1] for x in s])
            delta, m = meilleur_seuil(y_val, brutes, dist, np.quantile(dist, np.linspace(0.3, 1, 71)), plus_grand_mieux=False)
            lignes.append({"descripteurs": nom, "k": k, "delta": float(delta), **m})
            print(f"[B recadrée/val] {nom:18s} k={k} F1={m['f1_macro']:.3f}", flush=True)
            if meilleur is None or m["f1_macro"] > meilleur[0]:
                meilleur = (m["f1_macro"], nom, pc, pt, k, float(delta))
    with open(res / "B_recadree_ablation_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    _, nom, pc, pt, k, delta = meilleur
    X_app = descripteurs(jeux / "apprentissage_b", app, pc, pt, a.proc)
    X_test = descripteurs(jeux / "test", test, pc, pt, a.proc)
    rb = ReconnaisseurHistogrammes(k=k, delta=delta).entrainer_vecteurs(X_app, y_app)
    pred = np.array([rb.predire_vecteur(x)[0] for x in X_test], dtype=object)
    m = metriques(y_test, pred, CONNUS)
    cats = np.array([l["categorie"] for l in test])
    par_cat = {c: float((pred[(cats == c) & (y_test != "inconnu")] == y_test[(cats == c) & (y_test != "inconnu")]).mean())
               for c in dict.fromkeys(cats)}
    cm, etiq = matrice_confusion(y_test, pred, CONNUS)
    out = {"descripteurs": nom, "k": k, "delta": delta, "test": m, "par_categorie": par_cat, "matrice": cm.tolist()}
    json.dump(out, open(res / "B_recadree_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out["test"], indent=1), par_cat)


if __name__ == "__main__":
    main()
