"""Télécharge les images des billets étrangers utilisés comme billets « inconnus ».

Les images proviennent de la collection de David St-Onge publiée par le projet
« Hyperinflation Archive » (https://github.com/Rogzy-DB/hyperinflation-bank-note).
Elles ne sont pas couvertes par la licence de ce dépôt et ne sont donc pas
redistribuées ici : ce script les récupère à la version exacte utilisée pour les
résultats.

Usage : python telecharger_billets.py --sortie ../donnees/billets_publics
"""

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

from config_billets import ETRANGERS_CALIBRATION, ETRANGERS_TEST

VERSION = "73ecc31826e23efeef9edf34f6d38dfc4ca8202a"
BASE = f"https://raw.githubusercontent.com/Rogzy-DB/hyperinflation-bank-note/{VERSION}/assets/bills/web/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default="../donnees/billets_publics")
    a = ap.parse_args()
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for nom in ETRANGERS_CALIBRATION + ETRANGERS_TEST:
        cible = sortie / f"{nom}.jpg"
        if cible.exists():
            continue
        url = BASE + urllib.parse.quote(f"{nom}.jpg")
        with urllib.request.urlopen(url, timeout=60) as r:
            cible.write_bytes(r.read())
        print("téléchargé", cible.name)
    print(len(list(sortie.glob("*.jpg"))), "images dans", sortie)


if __name__ == "__main__":
    main()
