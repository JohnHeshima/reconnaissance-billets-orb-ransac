"""Télécharge les images officielles des billets en francs congolais depuis le site de la BCC.

Les images sont déjà fournies dans donnees/bcc_brut/ ; ce script permet de les
récupérer à nouveau à la source (adresses listées dans donnees/bcc_brut/sources.csv).
Source : Banque Centrale du Congo, www.bcc.cd. Reproduction autorisée à des fins
d'étude et de recherche avec citation de la source ; usage commercial interdit.

Usage : python telecharger_billets_cdf.py --sortie ../donnees/bcc_brut
puis   python preparer_billets_cdf.py
"""

import argparse
import csv
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="../donnees/bcc_brut/sources.csv")
    ap.add_argument("--sortie", default="../donnees/bcc_brut")
    a = ap.parse_args()
    sortie = Path(a.sortie)
    with open(a.sources, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            cible = sortie / ligne["fichier"]
            if cible.exists():
                continue
            cible.parent.mkdir(parents=True, exist_ok=True)
            requete = urllib.request.Request(ligne["url_image"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(requete, timeout=60) as r:
                cible.write_bytes(r.read())
            print("téléchargé", ligne["fichier"])


if __name__ == "__main__":
    main()
