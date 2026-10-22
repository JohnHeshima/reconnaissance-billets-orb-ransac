# Reconnaissance de billets en francs congolais par ORB + RANSAC

Reconnaissance de la coupure d'un billet en francs congolais (500, 1 000, 5 000, 10 000 et 20 000 FC) sur une photo prise librement, avec des méthodes classiques de reconnaissance des formes, sans apprentissage profond.
Projet du cours de Reconnaissance des formes, Master 1 Intelligence artificielle et sciences des données, Université de Kinshasa (sujet 39, variante A).

**Méthode (variante A)** : points d'intérêt et descripteurs ORB, appariement par distance de Hamming avec le test de ratio de Lowe, estimation d'une homographie par RANSAC par rapport à **une seule image de référence** par coupure et par face. Le billet est accepté si le nombre d'inliers atteint τ = 8 **et** si une vérification photométrique (corrélation des gradients entre la référence et la photo redressée par l'homographie) atteint κ = 0,26 ; sinon la photo est rejetée (« inconnu »). L'homographie localise aussi le billet dans la photo.

**Comparaison (variante B)** : histogrammes de couleur HSV + textures LBP + k plus proches voisins, sur les mêmes données.

## Résultats (914 photos de test)

| | A : ORB + RANSAC | A, 8 coupures en référence | B : HSV + LBP + k-NN | B, billet recadré (oracle) |
|---|---|---|---|---|
| F1 macro | **0,932** | **0,962** | 0,263 | 0,853 |
| Billets connus bien identifiés | **93,6 %** | 92,7 % | 24,4 % | 99,3 % |
| Confusions entre coupures | **0,2 %** | 0 % | 19,3 % | 0 % |
| Billets inconnus acceptés à tort | 8,0 % | **0 %** | 41,8 % | 34,1 % |
| Localisation (IoU moyen) | 0,955 | – | – | – |
| Temps par photo (un cœur) | 155 ms | – | 49 ms | – |

- Aucun billet étranger ni aucune scène vide n'est accepté par A. Ses fausses acceptations viennent toutes des 100 et 200 FC, de la même série graphique que le 500 FC (34 % d'entre eux sont pris pour des 500 FC).
- Enregistrer aussi les 50, 100 et 200 FC en référence supprime ces fausses acceptations : F1 0,962, aucune fausse acceptation, aucune confusion.
- La vérification photométrique divise par deux les fausses acceptations sur la validation (5,7 % → 2,9 %).
- Points faibles de A : billet photographié de loin (76 %) et billet froissé (84 %).
- B dépend du cadrage : sur une photo libre, le fond domine son histogramme ; même recadrée, son rejet des billets inconnus ne se généralise pas.

![Robustesse par type de difficulté](figures/10_robustesse_categories.png)

## Données

Aucune base publique de billets en francs congolais n'existe : la base est construite à partir des images officielles du site de la **Banque Centrale du Congo** (`donnees/bcc_brut/`, adresses dans `sources.csv`). Pour chaque coupure, une face est propre et l'autre porte des pastilles bleues numérotées d'annotation ; `code/preparer_billets_cdf.py` détecte et efface ces pastilles (inpainting), puis recadre le billet (`donnees/billets_cdf/`).

> Images : © Banque Centrale du Congo. La BCC autorise leur reproduction à des fins d'étude et de recherche, avec la mention de la source ; tout usage commercial est interdit. Ces images ne sont pas couvertes par la licence MIT du code.

Billets « inconnus » : les 50, 100 et 200 FC (même série graphique que les coupures connues ; 50 FC pour le réglage, 100 et 200 FC pour le test seulement) et 44 billets étrangers numérisés (collection de David St-Onge, projet [Hyperinflation Archive](https://github.com/Rogzy-DB/hyperinflation-bank-note)), téléchargés par `code/telecharger_billets.py` et non copiés ici.

Un générateur (`code/billets/synthese.py`) produit des photos réalistes à vérité terrain exacte, réparties en neuf types de difficulté : facile, rotation, perspective, petite échelle, éclairage, flou, occlusion par la main, billet froissé, compression. Il produit 444 photos de validation, 914 de test et 1 002 d'apprentissage pour B.

Pour évaluer de vraies photos prises au smartphone, voir `donnees/photos/LISEZMOI.md`.

## Reproduire

```bash
pip install -r requirements.txt
cd code
python preparer_billets_cdf.py           # images BCC : pastilles effacées, billets recadrés
python telecharger_billets.py            # billets étrangers (9 Mo)
python construire_jeux.py                # jeux de validation, test et apprentissage B
python experiences.py                    # ablation, réglage des seuils, test final
python experience_b_recadree.py          # variante B avec recadrage idéal
python experience_toutes_coupures.py     # A avec les 8 coupures en référence
python mesurer_temps.py                  # temps sur un seul cœur
python figures.py                        # figures
python demo.py --appariements photo.jpg  # une photo (ou --camera 0 pour la webcam)
```

Le notebook `Reconnaissance_billets_ORB_RANSAC.ipynb` (déjà exécuté, compatible Google Colab) suit la chaîne pas à pas sur une photo, puis présente les résultats.

## Organisation

| Chemin | Contenu |
|---|---|
| `code/billets/methode_orb.py` | Variante A : `ReconnaisseurORB` (références, appariement, RANSAC, contrôle géométrique, vérification photométrique, rejet) |
| `code/billets/methode_hist.py` | Variante B : descripteurs HSV + LBP, k-NN |
| `code/billets/synthese.py` | Générateur de photos |
| `code/billets/evaluation.py` | Métriques en monde ouvert, IoU, matrices de confusion |
| `code/billets/pretraitement.py` | Niveaux de gris + CLAHE, recadrage automatique d'une photo de référence |
| `code/config_billets.py` | Coupures connues et inconnues, chemins des images |
| `donnees/bcc_brut/`, `donnees/billets_cdf/` | Images de la BCC, brutes puis nettoyées |
| `resultats/` | Résultats chiffrés (CSV, JSON) |
| `figures/` | Figures |

## Licence

Code sous licence MIT (voir `LICENSE`). Les images de billets restent la propriété de leurs ayants droit (voir la section Données).
