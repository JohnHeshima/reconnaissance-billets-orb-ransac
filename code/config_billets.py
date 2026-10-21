"""Répartition des billets publics utilisés pour valider la chaîne de traitement.

Ces images (billets historiques de la collection de David St-Onge, publiées
par le projet « Hyperinflation Archive », github.com/Rogzy-DB/hyperinflation-bank-note ;
l'image du billet de 10 000 australes, vide dans la source, est écartée) remplacent provisoirement les
billets en francs congolais : elles permettent de mettre au point et de
mesurer la méthode avec exactement le même protocole. Avec de vraies photos de
francs congolais, il suffit de remplacer ces listes par les coupures congolaises.
"""

# 12 billets « connus » : le système dispose d'une image de référence de chacun
CONNUS = [
    "100k_Zaire", "100_Billion_Zimbabwe", "1000_Rouble", "500_Angola",
    "10k_Liban", "1_million_bolivar", "200k_nicaragua", "10k_Zlotych",
    "1000_Cruzados", "1_million_Intis", "1_million_Lira", "100_milliards_Drachma",
]

# billets inconnus vus pendant le réglage des seuils (et l'apprentissage de la variante B)
NEG_CALIBRATION = [
    "1000_mark", "10000_Kronen", "100k_mark", "10_centavos_Japan_Occupation",
    "50k_Rouble", "500k_pesos", "F2-6 2_ZWD",
    "100_millions_MilPengo", "1_million_georgia", "10_millards_mark",
]

# billets inconnus jamais vus avant le test final (dont des « cousins » de billets connus :
# 500k_Zaire pour 100k_Zaire, deux autres billets du Zimbabwe 2008, etc.)
NEG_TEST = [
    "1000_Kronen", "100_Trillion_Zimbabwe", "100_billion_mark",
    "500k_Zaire", "10k_Mark", "1_billion_Notgeld", "1_millard_MilPengo",
    "2000_millions_drachma", "500_millions_drachma",
    "500_millions_zimbabwe", "50_millards_dinar", "5_milliard_dinar",
    "F2-10 1_ZWR", "F2-12 50_Trillions_ZWR", "F2-4 500_millions_mark",
    "F2-5 1_million_B-Pengo", "F2-7 20k_cheque_ZWD", "F2-8 50_cents_ZWN",
    "F2-9 5_Billions_ZWN", "TizMillio_B_Pengo", "_million_mark", "million_mark",
]
