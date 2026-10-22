"""Expérience complémentaire : enregistrer aussi les autres coupures en circulation.

Sur le test, les fausses acceptations de la variante A viennent des coupures de
100 et 200 FC, de la même série graphique que le 500 FC. Dans un système
déployé, on enregistrerait toutes les coupures en circulation (50 à 20 000 FC) :
une photo de 100 FC serait alors reconnue comme un 100 FC, que le système
signale comme « pas une des cinq coupures demandées ». On mesure ici cet effet
sur le même jeu de test, avec les mêmes paramètres et les mêmes seuils (aucun
nouveau réglage) ; les prédictions 50, 100 et 200 FC comptent comme « inconnu ».

Usage : python experience_toutes_coupures.py --donnees ../donnees --jeux ../donnees/synthetique --resultats ../resultats
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from billets import INCONNU, ReconnaisseurORB, charger_image
from billets.evaluation import matrice_confusion, metriques
from config_billets import CONNUS, FACES, chemin_reference
from experiences import lire_manifeste

AUTRES = ["50FC", "100FC", "200FC"]
_RECO = None


def _init(donnees, params):
    global _RECO
    _RECO = ReconnaisseurORB(**params)
    for c in CONNUS + AUTRES:
        for f in FACES:
            _RECO.ajouter_reference(c, charger_image(chemin_reference(donnees, c, f)), f)


def _predire(chemin):
    r = _RECO.predire(charger_image(chemin))
    return r["classe"], r["classe_brute"], r["score"], r["correlation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="../donnees")
    ap.add_argument("--jeux", default="../donnees/synthetique")
    ap.add_argument("--resultats", default="../resultats")
    ap.add_argument("--proc", type=int, default=4)
    a = ap.parse_args()
    res = Path(a.resultats)
    syn = json.load(open(res / "synthese.json", encoding="utf-8"))
    test = lire_manifeste(Path(a.jeux) / "test")
    chemins = [str(Path(a.jeux) / "test" / l["fichier"]) for l in test]
    with Pool(a.proc, initializer=_init, initargs=(a.donnees, syn["A"]["parametres"])) as pool:
        sorties = pool.map(_predire, chemins, chunksize=8)
    brut = np.array([s[0] for s in sorties], dtype=object)
    pred = np.array([INCONNU if c in AUTRES else c for c in brut], dtype=object)
    y = np.array([l["classe"] for l in test], dtype=object)
    sources = [l["source"] for l in test]
    m = metriques(y, pred, CONNUS)
    cousins = np.array([s.split("_")[0] in AUTRES for s in sources])
    autres_neg = (y == INCONNU) & ~cousins
    coupure_vraie = np.array([s.split("_")[0] for s in sources], dtype=object)
    out = {
        "test": m,
        "fausse_acceptation_cousins": float((pred[cousins] != INCONNU).mean()),
        "fausse_acceptation_autres": float((pred[autres_neg] != INCONNU).mean()),
        "cousins_reconnus_comme_eux_memes": float((brut[cousins] == coupure_vraie[cousins]).mean()),
        "matrice": matrice_confusion(y, pred, CONNUS)[0].tolist(),
    }
    json.dump(out, open(res / "A_toutes_coupures_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "matrice"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
