"""Figures du rapport et de la présentation (à lancer après experiences.py)."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from skimage.feature import local_binary_pattern  # noqa: E402

from billets import ReconnaisseurORB, charger_image  # noqa: E402
from billets.pretraitement import preparer_gris, redimensionner  # noqa: E402
from billets.synthese import CATEGORIES  # noqa: E402
from config_billets import CONNUS, FACES, chemin_reference, charger_references  # noqa: E402
from demo import dessiner_appariements  # noqa: E402

COUL_A, COUL_B = "#2a78d6", "#eb6834"
GRIS, ENCRE, ENCRE2 = "#d6d5d0", "#0b0b0b", "#52514e"
BLEUS = LinearSegmentedColormap.from_list("bleus", ["#fcfcfb", "#cfe0f6", "#7fb0ea", "#2a78d6", "#0d3f80"])
NOMS_CAT = {"facile": "facile", "rotation": "rotation", "perspective": "perspective", "echelle": "petite échelle",
            "eclairage": "éclairage", "flou": "flou", "occlusion": "occlusion", "froisse": "froissé",
            "compression": "compression"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": ENCRE2, "xtick.color": ENCRE2,
    "ytick.color": ENCRE2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11, "text.color": ENCRE,
})


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def lire_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default="../donnees")
    ap.add_argument("--jeux", default="../donnees/synthetique")
    ap.add_argument("--resultats", default="../resultats")
    ap.add_argument("--figures", default="../figures")
    a = ap.parse_args()
    donnees, jeux, res, fig = Path(a.donnees), Path(a.jeux), Path(a.resultats), Path(a.figures)
    fig.mkdir(parents=True, exist_ok=True)
    syn = json.load(open(res / "synthese.json", encoding="utf-8"))
    preds = lire_csv(res / "test_predictions.csv")
    manif = {l["fichier"]: l for l in lire_csv(jeux / "test" / "manifeste.csv")}
    tau = syn["A"]["tau"]

    # 1. billets de référence
    f, axes = plt.subplots(len(CONNUS), 2, figsize=(7.4, 8.6))
    for ligne, c in zip(axes, CONNUS):
        for ax, face in zip(ligne, FACES):
            ax.imshow(rgb(charger_image(chemin_reference(donnees, c, face))))
            ax.set_title(f"{c[:-2]} FC, {face}", fontsize=8, fontweight="normal")
            ax.axis("off")
    f.tight_layout()
    f.savefig(fig / "01_references.png")
    plt.close(f)

    # 2. exemples de photos simulées par catégorie
    f, axes = plt.subplots(3, 3, figsize=(10, 7.6))
    vus = {}
    for l in manif.values():
        if l["classe"] != "inconnu" and l["categorie"] not in vus:
            vus[l["categorie"]] = l["fichier"]
    for ax, c in zip(axes.ravel(), CATEGORIES):
        ax.imshow(rgb(charger_image(jeux / "test" / vus[c])))
        ax.set_title(NOMS_CAT[c], fontsize=10)
        ax.axis("off")
    f.savefig(fig / "02_exemples_categories.png")
    plt.close(f)

    # 3. points ORB sur une référence + descripteurs de la variante B
    ref = charger_image(chemin_reference(donnees, "1000FC", "recto"))
    petit, _ = redimensionner(ref, 800)
    gris = preparer_gris(petit)
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, fastThreshold=10)
    kp = orb.detect(gris, None)
    vis = cv2.drawKeypoints(petit, kp[:400], None, color=(214, 120, 42), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    f, ax = plt.subplots(figsize=(8, 4.2))
    ax.imshow(rgb(vis))
    ax.set_title(f"Points ORB détectés ({len(kp)} au total, 400 affichés avec échelle et orientation)")
    ax.axis("off")
    f.savefig(fig / "03_points_orb.png")
    plt.close(f)

    f, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    hsv = cv2.cvtColor(petit, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    axes[0].imshow(hh.T, origin="lower", aspect="auto", cmap=BLEUS, extent=[0, 180, 0, 256])
    axes[0].set_xlabel("teinte H")
    axes[0].set_ylabel("saturation S")
    axes[0].set_title("Histogramme 2D teinte × saturation")
    axes[0].grid(False)
    lbp = local_binary_pattern(cv2.cvtColor(petit, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
    axes[1].imshow(lbp, cmap="gray")
    axes[1].set_title("Codes LBP uniformes (P=8, R=1)")
    axes[1].axis("off")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    axes[2].bar(range(10), hist / hist.sum(), color=COUL_B, width=0.7)
    axes[2].set_xlabel("code LBP (9 = non uniforme)")
    axes[2].set_title("Histogramme LBP")
    f.tight_layout()
    f.savefig(fig / "04_descripteurs_variante_b.png")
    plt.close(f)

    # 5. appariements RANSAC sur deux photos de test difficiles
    reco = charger_references(ReconnaisseurORB(**{k: v for k, v in syn["A"]["parametres"].items()}), donnees, charger_image)
    exemples = [p for p in preds if p["vrai"] == p["A_pred"] != "inconnu" and p["categorie"] in ("perspective", "occlusion")]
    f, axes = plt.subplots(2, 1, figsize=(10, 8.4))
    for ax, ex in zip(axes, [exemples[0], next(e for e in exemples if e["categorie"] == "occlusion")]):
        img = charger_image(jeux / "test" / ex["fichier"])
        r = reco.predire(img, details=True)
        ax.imshow(rgb(dessiner_appariements(img, r, 60)))
        ax.set_title(f"{NOMS_CAT[ex['categorie']]} : {r['classe']} reconnu, {r['score']} inliers RANSAC "
                     f"(2e meilleure référence : {r['second_score']})", fontsize=10)
        ax.axis("off")
    f.tight_layout()
    f.savefig(fig / "05_appariements_ransac.png")
    plt.close(f)

    # 6. localisation : polygone estimé vs vérité terrain
    f, axes = plt.subplots(2, 3, figsize=(11, 5.8))
    choix = []
    for c in ("rotation", "perspective", "echelle", "eclairage", "froisse", "occlusion"):
        choix.append(next(p for p in preds if p["categorie"] == c and p["A_iou"] not in ("",) and p["vrai"] == p["A_pred"]))
    for ax, p in zip(axes.ravel(), choix):
        img = charger_image(jeux / "test" / p["fichier"])
        r = reco.predire(img)
        gt = np.array(json.loads(manif[p["fichier"]]["quad"]), np.int32)
        cv2.polylines(img, [gt], True, (255, 160, 60), 6)
        if r["polygone_brut"] is not None:
            cv2.polylines(img, [r["polygone_brut"].astype(np.int32)], True, (0, 200, 0), 3)
        ax.imshow(rgb(img))
        ax.set_title(f"{NOMS_CAT[p['categorie']]} : IoU = {float(p['A_iou']):.2f}", fontsize=10)
        ax.axis("off")
    f.suptitle("Localisation par homographie : vérité terrain (bleu clair) et estimation (vert)", fontsize=11)
    f.tight_layout()
    f.savefig(fig / "06_localisation_iou.png")
    plt.close(f)

    # 7. distribution des scores (inliers) : positifs vs négatifs
    sc_pos = np.array([float(p["A_inliers"]) for p in preds if p["vrai"] != "inconnu" and p["A_brute"] == p["vrai"]])
    sc_neg = np.array([float(p["A_inliers"]) for p in preds if p["vrai"] == "inconnu"])
    f, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.r_[np.arange(0, 60, 2), np.geomspace(60, max(61, sc_pos.max() + 1), 12)]
    ax.hist(np.clip(sc_pos, 0, 60), bins=np.arange(0, 62, 2), color=COUL_A, alpha=0.6, label="billet connu (bonne référence)")
    ax.hist(np.clip(sc_neg, 0, 60), bins=np.arange(0, 62, 2), color=COUL_B, alpha=0.9, histtype="stepfilled",
            edgecolor="white", linewidth=0.8, label="photos sans billet connu")
    ax.axvline(tau, color=ENCRE, lw=1.2, ls="--")
    ax.text(tau + 0.8, ax.get_ylim()[1] * 0.9, f"seuil τ = {tau}", color=ENCRE)
    ax.set_xlabel("nombre d'inliers RANSAC de la meilleure référence (valeurs > 60 regroupées à 60)")
    ax.set_ylabel("nombre de photos")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Séparation des scores sur le jeu de test")
    f.savefig(fig / "07_distribution_scores.png")
    plt.close(f)

    # 8. courbes en fonction du seuil tau (validation et test)
    cv = json.load(open(res / "A_courbe_seuil_validation.json"))
    ct = syn["courbe_seuil_test_A"]
    f, ax = plt.subplots(figsize=(8, 3.6))
    for courbe, ls, nom in ((cv, ":", "validation"), (ct, "-", "test")):
        t = [c["seuil"] for c in courbe]
        ax.plot(t, [c["f1_macro"] for c in courbe], color=COUL_A, ls=ls, lw=2, label=f"F1 macro ({nom})")
        ax.plot(t, [c["taux_fausse_acceptation"] for c in courbe], color=COUL_B, ls=ls, lw=2, label=f"fausse acceptation ({nom})")
    ax.axvline(tau, color=ENCRE, lw=1, ls="--")
    ax.set_xlabel("seuil de rejet τ (inliers)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="center right")
    ax.set_title("Choix du seuil de rejet τ")
    f.savefig(fig / "08_courbe_seuil.png")
    plt.close(f)

    # 9. matrices de confusion
    etiq = [e.replace("_", " ") for e in syn["matrices"]["etiquettes"]]
    f, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, cle, titre in ((axes[0], "A", "Variante A : ORB + RANSAC"), (axes[1], "B", "Variante B : HSV + LBP + k-NN")):
        cm = np.array(syn["matrices"][cle])
        ax.imshow(np.log1p(cm), cmap=BLEUS)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if np.log1p(cm[i, j]) > np.log1p(cm.max()) * 0.6 else ENCRE)
        ax.set_xticks(range(len(etiq)), etiq, rotation=70, fontsize=7)
        ax.set_yticks(range(len(etiq)), etiq, fontsize=7)
        ax.set_xlabel("classe prédite")
        ax.set_ylabel("classe réelle")
        ax.set_title(titre)
        ax.grid(False)
    f.tight_layout()
    f.savefig(fig / "09_matrices_confusion.png")
    plt.close(f)

    # 10. robustesse par catégorie
    pc = lire_csv(res / "test_par_categorie.csv")
    rec = json.load(open(res / "B_recadree_test.json", encoding="utf-8")) if (res / "B_recadree_test.json").exists() else None
    series = [("A : ORB + RANSAC", [float(p["A_identification"]) for p in pc], COUL_A),
              ("B : HSV + LBP + k-NN", [float(p["B_identification"]) for p in pc], COUL_B)]
    if rec:
        series.append(("B avec recadrage idéal du billet", [rec["par_categorie"][p["categorie"]] for p in pc], "#1baf7a"))
    f, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(pc))
    larg = 0.8 / len(series)
    for i, (nom, vals, coul) in enumerate(series):
        xs = x - 0.4 + larg * (i + 0.5)
        ax.bar(xs, vals, larg * 0.92, color=coul, label=nom)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{100 * v:.0f}", ha="center", fontsize=6.5, color=ENCRE2)
    ax.set_xticks(x, [NOMS_CAT[p["categorie"]] for p in pc], rotation=20)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("taux d'identification")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("Billets connus correctement identifiés (%), par type de difficulté", pad=30)
    f.savefig(fig / "10_robustesse_categories.png")
    plt.close(f)

    # 11. ablation variante A
    ab = lire_csv(res / "A_ablation_validation.csv")
    f, ax = plt.subplots(figsize=(8, 4.6))
    noms = [r["configuration"] for r in ab]
    v = [float(r["f1_macro"]) for r in ab]
    y = np.arange(len(ab))[::-1]
    ax.barh(y, v, color=[COUL_A if n == "référence" else "#7fb0ea" for n in noms], height=0.6)
    for yi, vi, r in zip(y, v, ab):
        ax.text(vi + 0.003, yi, f"{vi:.3f}  (τ={r['tau']}, {float(r['temps_ms']) / float(ab[0]['temps_ms']):.1f}× temps)", va="center", fontsize=7, color=ENCRE2)
    ax.set_yticks(y, noms, fontsize=8)
    ax.set_xlim(min(v) - 0.05, 1.04)
    ax.set_xlabel("F1 macro sur la validation (seuil τ optimisé pour chaque configuration)")
    ax.set_title("Étude d'ablation de la variante A")
    f.savefig(fig / "11_ablation_a.png")
    plt.close(f)

    # 12. cas d'échec de A
    echecs = [p for p in preds if p["vrai"] != p["A_pred"]][:6]
    if echecs:
        f, axes = plt.subplots(2, 3, figsize=(11, 5.8))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, p in zip(axes.ravel(), echecs):
            ax.imshow(rgb(charger_image(jeux / "test" / p["fichier"])))
            ax.set_title(f"{NOMS_CAT[p['categorie']]} | vrai : {p['vrai'][:16]}\nprédit : {p['A_pred'][:16]} "
                         f"({p['A_inliers']} inliers)", fontsize=8)
        f.tight_layout()
        f.savefig(fig / "12_echecs_a.png")
        plt.close(f)
    # 13. vérification photométrique : inliers et corrélation des gradients sur le test
    rho = syn["A"].get("rho")
    groupes = [("billet connu (bonne coupure)", lambda p: p["vrai"] != "inconnu" and p["A_brute"] == p["vrai"], COUL_A),
               ("coupure congolaise inconnue (100, 200 FC)", lambda p: p["vrai"] == "inconnu" and p["source"].split("_")[0].endswith("FC"), COUL_B),
               ("billet étranger ou scène vide", lambda p: p["vrai"] == "inconnu" and not p["source"].split("_")[0].endswith("FC"), "#8a8984")]
    f, ax = plt.subplots(figsize=(8, 4.4))
    for nom, cond, coul in groupes:
        sel = [p for p in preds if cond(p)]
        ax.scatter([max(1, float(p["A_inliers"])) for p in sel], [float(p["A_correlation"]) for p in sel],
                   s=10, alpha=0.6, color=coul, label=f"{nom} ({len(sel)})", edgecolors="none")
    ax.set_xscale("log")
    ax.axvline(tau, color=ENCRE, lw=1, ls="--")
    if rho is not None:
        ax.axhline(rho, color=ENCRE, lw=1, ls=":")
        ax.text(1.05, rho + 0.015, f"κ = {rho}".replace(".", ","), color=ENCRE, fontsize=8)
    ax.text(tau * 1.05, 0.95, f"τ = {tau}", color=ENCRE, fontsize=8)
    ax.set_xlabel("inliers RANSAC de la meilleure référence (échelle log ; 0 ramené à 1)")
    ax.set_ylabel("corrélation des gradients après redressement")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Décision de la variante A sur le jeu de test")
    f.savefig(fig / "13_verification_photometrique.png")
    plt.close(f)

    # 14. pourquoi les coupures de même maquette trompent les inliers
    brut = ReconnaisseurORB(**{**syn["A"]["parametres"], "tau": 0, "rho": None, "relance_zoom": False})
    charger_references(brut, donnees, charger_image)
    cousin = max((p for p in preds if p["source"] in ("100FC_recto", "200FC_recto") and p["categorie"] in ("facile", "rotation")),
                 key=lambda p: float(p["A_inliers"]))
    vrai = next(p for p in preds if p["source"] == "500FC_recto" and p["categorie"] == "facile" and not p["distracteur"])
    f, axes = plt.subplots(2, 1, figsize=(10, 8.4))
    for ax, ex, titre in ((axes[0], vrai, "billet de 500 FC"), (axes[1], cousin, f"billet de {cousin['source'].split('FC')[0]} FC (inconnu, même maquette)")):
        img = charger_image(jeux / "test" / ex["fichier"])
        r = brut.predire(img, details=True)
        ax.imshow(rgb(dessiner_appariements(img, r, 60)))
        ax.set_title(f"{titre} : meilleure référence {r['classe_brute']} {r['face']}, {r['score']} inliers, "
                     f"corrélation {r['correlation']:.2f}".replace(".", ","), fontsize=10)
        ax.axis("off")
    f.tight_layout()
    f.savefig(fig / "14_coupures_meme_maquette.png")
    plt.close(f)
    print("figures écrites dans", fig)


if __name__ == "__main__":
    main()
