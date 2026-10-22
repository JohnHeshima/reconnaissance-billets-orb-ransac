"""Démonstration : reconnaître un billet sur une photo (ou en direct à la webcam).

Exemples
  python demo.py photo.jpg
  python demo.py --appariements photo.jpg   # montre les inliers RANSAC
  python demo.py --camera 0                 # webcam (q pour quitter)
  python demo.py --refs ../donnees/references photo.jpg   # références préparées à partir de vos photos

Le dossier de références contient une image par coupure et par face, nommée
<coupure>_<face>.jpg (par ex. 1000FC_recto.jpg, 1000FC_verso.jpg) ou <coupure>.jpg.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from billets import INCONNU, ReconnaisseurORB, charger_image


def charger_references(dossier, reco):
    for chemin in sorted(Path(dossier).glob("*.jpg")) + sorted(Path(dossier).glob("*.png")):
        nom = chemin.stem
        classe, _, face = nom.rpartition("_") if nom.endswith(("_recto", "_verso")) else (nom, "", "")
        n = reco.ajouter_reference(classe, charger_image(chemin), face)
        print(f"référence {classe:>22s} {face:6s} : {n} points ORB")


def annoter(img, res):
    out = img.copy()
    epais = max(2, img.shape[1] // 300)
    if res["classe"] != INCONNU and res["polygone"] is not None:
        cv2.polylines(out, [res["polygone"].astype(np.int32)], True, (0, 220, 0), epais)
        texte = f"{res['classe']} {res['face']}  ({res['score']} inliers)"
        couleur = (0, 160, 0)
    else:
        texte = f"Billet non reconnu (meilleur score {res['score']})"
        couleur = (0, 0, 220)
    echelle = img.shape[1] / 900
    (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, echelle, epais)
    cv2.rectangle(out, (0, 0), (tw + 20, th + 24), (255, 255, 255), -1)
    cv2.putText(out, texte, (10, th + 12), cv2.FONT_HERSHEY_SIMPLEX, echelle, couleur, epais)
    return out


def dessiner_appariements(img, res, max_traits=80):
    """Image référence | photo, avec les appariements retenus par RANSAC."""
    ref, info = res.get("ref"), res.get("info", {})
    if ref is None or "masque" not in info:
        return None
    f = res["facteur"]
    petit = cv2.resize(img, None, fx=f, fy=f) if f != 1 else img
    inliers = [m for m, ok in zip(info["appariements"], info["masque"]) if ok][:max_traits]
    return cv2.drawMatches(ref["image"], ref["kp"], petit, res["kp"], inliers, None,
                           matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--refs", default="../donnees/billets_cdf/references")
    ap.add_argument("--tau", type=int, default=8, help="nombre minimal d'inliers pour accepter")
    ap.add_argument("--rho", type=float, default=0.26, help="corrélation minimale des gradients (vérification photométrique)")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--appariements", action="store_true")
    ap.add_argument("--sortie", default="sorties_demo")
    a = ap.parse_args()

    reco = ReconnaisseurORB(tau=a.tau, rho=a.rho, n_niveaux=12)
    charger_references(a.refs, reco)

    if a.camera is not None:
        cap = cv2.VideoCapture(a.camera)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            cv2.imshow("Reconnaissance de billets (q pour quitter)", annoter(img, reco.predire(img)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    sortie = Path(a.sortie)
    sortie.mkdir(exist_ok=True)
    for chemin in a.images:
        img = charger_image(chemin)
        res = reco.predire(img, details=True)
        print(f"{chemin}: {res['classe']} {res['face']} | inliers={res['score']} "
              f"(2e meilleur : {res['second_score']}) | {res['temps_ms']:.0f} ms")
        cv2.imwrite(str(sortie / f"{Path(chemin).stem}_resultat.jpg"), annoter(img, res))
        if a.appariements:
            vis = dessiner_appariements(img, res)
            if vis is not None:
                cv2.imwrite(str(sortie / f"{Path(chemin).stem}_appariements.jpg"), vis)


if __name__ == "__main__":
    main()
