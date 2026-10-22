"""Billets utilisés : coupures en francs congolais à reconnaître et billets « inconnus ».

Billets congolais : images officielles de la Banque Centrale du Congo (www.bcc.cd),
préparées par `preparer_billets_cdf.py` dans donnees/billets_cdf/ :
  references/<coupure>_<face>.jpg : image de référence de chaque face ;
  sources/<coupure>_<face>.jpg    : image servant à simuler les photos.
Billets étrangers : collection de David St-Onge publiée par le projet « Hyperinflation
Archive » (github.com/Rogzy-DB/hyperinflation-bank-note), récupérés par
`telecharger_billets.py` dans donnees/billets_publics/.
"""

from pathlib import Path

# les 5 coupures demandées par le sujet, chacune avec ses deux faces
CONNUS = ["500FC", "1000FC", "5000FC", "10000FC", "20000FC"]
FACES = ("recto", "verso")

# « cousins » : coupures congolaises de même maquette (même banque, même mise en page,
# mêmes motifs de sécurité) qui ne font pas partie des billets à reconnaître.
# C'est le cas le plus difficile pour le rejet en monde ouvert.
COUSINS_CALIBRATION = ["50FC_recto", "50FC_verso"]
COUSINS_TEST = ["100FC_recto", "100FC_verso", "200FC_recto", "200FC_verso"]

# billets étrangers vus pendant le réglage des seuils (et l'apprentissage de la variante B)
ETRANGERS_CALIBRATION = [
    "1000_mark", "10000_Kronen", "100k_mark", "10_centavos_Japan_Occupation",
    "50k_Rouble", "500k_pesos", "F2-6 2_ZWD",
    "100_millions_MilPengo", "1_million_georgia", "10_millards_mark",
]

# billets étrangers jamais vus avant le test final
ETRANGERS_TEST = [
    "1000_Kronen", "100_Trillion_Zimbabwe", "100_billion_mark",
    "500k_Zaire", "10k_Mark", "1_billion_Notgeld", "1_millard_MilPengo",
    "2000_millions_drachma", "500_millions_drachma",
    "500_millions_zimbabwe", "50_millards_dinar", "5_milliard_dinar",
    "F2-10 1_ZWR", "F2-12 50_Trillions_ZWR", "F2-4 500_millions_mark",
    "F2-5 1_million_B-Pengo", "F2-7 20k_cheque_ZWD", "F2-8 50_cents_ZWN",
    "F2-9 5_Billions_ZWN", "TizMillio_B_Pengo", "_million_mark", "million_mark",
    "100k_Zaire", "100_Billion_Zimbabwe", "1000_Rouble", "500_Angola",
    "10k_Liban", "1_million_bolivar", "200k_nicaragua", "10k_Zlotych",
    "1000_Cruzados", "1_million_Intis", "1_million_Lira", "100_milliards_Drachma",
]

NEG_CALIBRATION = COUSINS_CALIBRATION + ETRANGERS_CALIBRATION
NEG_TEST = COUSINS_TEST + ETRANGERS_TEST


def est_congolais(nom):
    return nom.split("_")[0].endswith("FC")


def chemin_reference(donnees, classe, face):
    return Path(donnees) / "billets_cdf" / "references" / f"{classe}_{face}.jpg"


def chemin_source(donnees, nom):
    """Image d'où l'on simule les photos : `nom` vaut <coupure>_<face> ou le nom d'un billet étranger."""
    if est_congolais(nom):
        return Path(donnees) / "billets_cdf" / "sources" / f"{nom}.jpg"
    return Path(donnees) / "billets_publics" / f"{nom}.jpg"


def charger_references(reco, donnees, charger):
    """Ajoute au reconnaisseur ORB les deux faces de chaque coupure connue."""
    for c in CONNUS:
        for f in FACES:
            reco.ajouter_reference(c, charger(chemin_reference(donnees, c, f)), f)
    return reco
