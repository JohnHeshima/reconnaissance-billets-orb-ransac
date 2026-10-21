# Photos de billets congolais

Un dossier par coupure, plus un dossier `negatifs` :

    photos/
      1000FC/   1000FC_recto_ref.jpg  1000FC_verso_ref.jpg  img_001.jpg  img_002.jpg ...
      5000FC/   ...
      negatifs/ dollar_01.jpg  recu_01.jpg ...

- `_ref` dans le nom : photo de référence (billet à plat, fond uni, vu du dessus).
- Toutes les autres photos servent de test.

Puis, depuis `code/` :

    python preparer_photos_reelles.py preparer   # références recadrées -> donnees/references/
    python preparer_photos_reelles.py evaluer --tau 7
