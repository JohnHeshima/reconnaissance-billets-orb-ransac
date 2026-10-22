# Photos de billets congolais prises au smartphone

Un dossier par coupure, plus un dossier `negatifs` :

    photos/
      500FC/    img_001.jpg  img_002.jpg ...
      1000FC/   ...
      5000FC/   ...
      10000FC/  ...
      20000FC/  ...
      negatifs/ dollar_01.jpg  recu_01.jpg  scene_vide_01.jpg ...

Photos de test conseillées, pour chaque coupure et chaque face : billet tenu en main, incliné, plié ou froissé,
partiellement caché, mal éclairé, sur un fond encombré, photographié de près et de loin, dans différentes orientations.
Les 50, 100 et 200 FC peuvent aller dans leur propre dossier (`100FC/`...) : ils sont reconnus avec les références BCC.

Puis, depuis `code/` :

    python preparer_photos_reelles.py evaluer          # références BCC, tau = 8, kappa = 0,26

Facultatif : remplacer les images de la BCC par vos propres photos de référence (billet à plat, fond uni, vu du
dessus, nom `<coupure>_recto_ref.jpg` ou `<coupure>_verso_ref.jpg` dans le dossier de la coupure) :

    python preparer_photos_reelles.py preparer         # références recadrées -> donnees/references/
    python preparer_photos_reelles.py evaluer
