"""Protocole expérimental complet : réglage sur la validation, mesure finale sur le test.

Usage : python experiences.py --refs ../donnees/billets_publics --jeux ../donnees/synthetique \
                              --resultats ../resultats --figures ../figures

Étapes
  1. Variante A (ORB + RANSAC) : étude d'ablation sur la validation (ratio de Lowe,
     nombre de points, seuil RANSAC, CLAHE, contrôle géométrique, pyramide), choix du
     seuil de rejet tau qui maximise le F1 macro de validation.
  2. Variante B (HSV + LBP + k-NN) : choix de k, des descripteurs et du seuil de
     distance delta sur la même validation.
  3. Évaluation unique sur le jeu de test (billets inconnus jamais vus).
"""

import argparse
import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from billets import INCONNU, ReconnaisseurHistogrammes, ReconnaisseurORB, charger_image
from billets.evaluation import decision_seuil, iou_quadrilateres, matrice_confusion, metriques
from billets.methode_hist import descripteur_histogrammes
from config_billets import CONNUS

ARGS = None
_RECO = None


def lire_manifeste(dossier):
    with open(Path(dossier) / "manifeste.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ variante A
def _init_orb(refs, params):
    global _RECO
    _RECO = ReconnaisseurORB(**params)
    for c in CONNUS:
        _RECO.ajouter_reference(c, charger_image(Path(refs) / f"{c}.jpg"))


def _predire_orb(chemin):
    p = _RECO.predire(charger_image(chemin))
    return {"classe_brute": p["classe_brute"], "score": p["score"], "second": p["second_score"],
            "polygone": None if p["polygone_brut"] is None else p["polygone_brut"].tolist(),
            "temps_ms": p["temps_ms"], "zoom": p.get("zoom", 1.0)}


def executer_orb(refs, dossier, lignes, params, n_proc):
    chemins = [str(Path(dossier) / l["fichier"]) for l in lignes]
    with Pool(n_proc, initializer=_init_orb, initargs=(refs, params)) as pool:
        return pool.map(_predire_orb, chemins, chunksize=8)


def meilleur_seuil(y, brutes, scores, candidats, plus_grand_mieux=True):
    """Seuil qui maximise le F1 macro ; à égalité on garde le plus prudent."""
    best = None
    for t in candidats:
        m = metriques(y, decision_seuil(brutes, scores, t, plus_grand_mieux), CONNUS)
        cle = (round(m["f1_macro"], 6), t if plus_grand_mieux else -t)
        if best is None or cle > best[0]:
            best = (cle, t, m)
    return best[1], best[2]


def courbe_seuil(y, brutes, scores, candidats, plus_grand_mieux=True):
    out = []
    for t in candidats:
        m = metriques(y, decision_seuil(brutes, scores, t, plus_grand_mieux), CONNUS)
        m["seuil"] = float(t)
        out.append(m)
    return out


# ------------------------------------------------------------------ variante B
def _desc(args):
    chemin, pc, pt = args
    return descripteur_histogrammes(charger_image(chemin), pc, pt)


def descripteurs(dossier, lignes, pc, pt, n_proc):
    taches = [(str(Path(dossier) / l["fichier"]), pc, pt) for l in lignes]
    with Pool(n_proc) as pool:
        return np.stack(pool.map(_desc, taches, chunksize=16))


# ------------------------------------------------------------------ programme principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="../donnees/billets_publics")
    ap.add_argument("--jeux", default="../donnees/synthetique")
    ap.add_argument("--resultats", default="../resultats")
    ap.add_argument("--proc", type=int, default=4)
    a = ap.parse_args()
    jeux, res = Path(a.jeux), Path(a.resultats)
    res.mkdir(parents=True, exist_ok=True)

    val = lire_manifeste(jeux / "validation")
    test = lire_manifeste(jeux / "test")
    app = lire_manifeste(jeux / "apprentissage_b")
    y_val = np.array([l["classe"] for l in val])
    y_test = np.array([l["classe"] for l in test])
    taus = list(range(4, 61))

    # ---------------- 1. ablation variante A sur la validation
    base = dict(n_points=2000, ratio=0.75, seuil_ransac=5.0, clahe=True, controle_geometrique=True,
                n_niveaux=8, tau=0)
    variantes = [("référence", {})]
    variantes += [(f"ratio={r}", {"ratio": r}) for r in (0.65, 0.70, 0.80, 0.85)]
    variantes += [(f"points={n}", {"n_points": n}) for n in (500, 1000, 4000)]
    variantes += [(f"RANSAC={s}px", {"seuil_ransac": s}) for s in (3.0, 8.0, 12.0)]
    variantes += [("sans CLAHE", {"clahe": False}), ("sans contrôle géométrique", {"controle_geometrique": False}),
                  ("pyramide 12 niveaux", {"n_niveaux": 12})]
    ablation = []
    preds_val = {}
    for nom, delta in variantes:
        params = {**base, **delta}
        t0 = time.time()
        p = executer_orb(a.refs, jeux / "validation", val, params, a.proc)
        brutes = np.array([q["classe_brute"] for q in p], dtype=object)
        scores = np.array([q["score"] for q in p], float)
        tau, m = meilleur_seuil(y_val, brutes, scores, taus)
        ablation.append({"configuration": nom, "tau": tau, **m,
                         "temps_ms": float(np.mean([q["temps_ms"] for q in p]))})
        preds_val[nom] = (p, brutes, scores)
        print(f"[A/val] {nom:28s} tau={tau:3d} F1={m['f1_macro']:.3f} FA={m['taux_fausse_acceptation']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # configuration retenue : on garde la référence sauf si une variante gagne plus de
    # 0,01 de F1 (environ deux photos de validation), écart en dessous duquel la
    # différence n'est pas significative sur 336 photos
    choix = ablation[0]
    for r in ablation[1:]:
        if r["f1_macro"] > choix["f1_macro"] + 0.01 and r["taux_fausse_acceptation"] <= choix["taux_fausse_acceptation"]:
            choix = r
    nom_choix = choix["configuration"]
    params_choix = {**base, **dict(variantes)[nom_choix]}
    tau_a = choix["tau"]

    # relance zoom x2 (dépend de tau, évaluée à part)
    params_zoom = {**params_choix, "tau": tau_a, "relance_zoom": True}
    p = executer_orb(a.refs, jeux / "validation", val, params_zoom, a.proc)
    brutes = np.array([q["classe_brute"] for q in p], dtype=object)
    scores = np.array([q["score"] for q in p], float)
    m = metriques(y_val, decision_seuil(brutes, scores, tau_a), CONNUS)
    ablation.append({"configuration": f"{nom_choix} + relance zoom x2", "tau": tau_a, **m,
                     "temps_ms": float(np.mean([q["temps_ms"] for q in p]))})
    print(f"[A/val] relance zoom          F1={m['f1_macro']:.3f}", flush=True)
    zoom_utile = m["f1_macro"] > choix["f1_macro"] + 0.01
    if zoom_utile:
        params_choix = params_zoom
    params_choix["tau"] = tau_a

    with open(res / "A_ablation_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation[0].keys()))
        w.writeheader()
        w.writerows(ablation)
    p_ref, b_ref, s_ref = preds_val["référence"]
    json.dump(courbe_seuil(y_val, b_ref, s_ref, taus), open(res / "A_courbe_seuil_validation.json", "w"), indent=1)

    # ---------------- 2. variante B : réglage sur la validation
    y_app = np.array([l["classe"] for l in app])
    ablation_b = []
    meilleur_b = None
    for nom, pc, pt in (("couleur + texture", 1.0, 1.0), ("couleur seule", 1.0, 0.0), ("texture seule", 0.0, 1.0)):
        X_app = descripteurs(jeux / "apprentissage_b", app, pc, pt, a.proc)
        X_val = descripteurs(jeux / "validation", val, pc, pt, a.proc)
        for k in (1, 3, 5, 7):
            rb = ReconnaisseurHistogrammes(k=k, poids_couleur=pc, poids_texture=pt).entrainer_vecteurs(X_app, y_app)
            sorties = [rb.predire_vecteur(x) for x in X_val]
            brutes = np.array([s[0] for s in sorties], dtype=object)
            dist = np.array([s[1] for s in sorties])
            deltas = np.quantile(dist, np.linspace(0.3, 1.0, 71))
            delta, mb = meilleur_seuil(y_val, brutes, dist, deltas, plus_grand_mieux=False)
            ablation_b.append({"descripteurs": nom, "k": k, "delta": float(delta), **mb})
            print(f"[B/val] {nom:18s} k={k} delta={delta:.3f} F1={mb['f1_macro']:.3f}", flush=True)
            if meilleur_b is None or mb["f1_macro"] > meilleur_b[0]:
                meilleur_b = (mb["f1_macro"], nom, pc, pt, k, float(delta))
    with open(res / "B_ablation_validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_b[0].keys()))
        w.writeheader()
        w.writerows(ablation_b)
    _, nom_b, pc, pt, k_b, delta_b = meilleur_b

    # ---------------- 3. test final
    p_test = executer_orb(a.refs, jeux / "test", test, params_choix, a.proc)
    brutes_a = np.array([q["classe_brute"] for q in p_test], dtype=object)
    scores_a = np.array([q["score"] for q in p_test], float)
    pred_a = decision_seuil(brutes_a, scores_a, tau_a)

    X_app = descripteurs(jeux / "apprentissage_b", app, pc, pt, a.proc)
    X_test = descripteurs(jeux / "test", test, pc, pt, a.proc)
    rb = ReconnaisseurHistogrammes(k=k_b, delta=delta_b, poids_couleur=pc, poids_texture=pt).entrainer_vecteurs(X_app, y_app)
    t0 = time.perf_counter()
    sorties_b = [rb.predire_vecteur(x) for x in X_test]
    t_knn = 1000 * (time.perf_counter() - t0) / len(X_test)
    # temps complet de B (descripteur + k-NN) mesuré sur 40 images
    t_desc = []
    for l in test[:40]:
        img = charger_image(jeux / "test" / l["fichier"])
        t1 = time.perf_counter()
        descripteur_histogrammes(img, pc, pt)
        t_desc.append(1000 * (time.perf_counter() - t1))
    pred_b = np.array([s[0] for s in sorties_b], dtype=object)

    lignes = []
    for l, q, pb, sb in zip(test, p_test, pred_b, sorties_b):
        iou = ""
        if l["classe"] != INCONNU and q["classe_brute"] == l["classe"] and q["polygone"] is not None:
            iou = round(iou_quadrilateres(q["polygone"], json.loads(l["quad"])), 4)
        lignes.append({"fichier": l["fichier"], "vrai": l["classe"], "categorie": l["categorie"],
                       "source": l["source"], "distracteur": l["distracteur"],
                       "A_pred": decision_seuil([q["classe_brute"]], [q["score"]], tau_a)[0],
                       "A_brute": q["classe_brute"], "A_inliers": q["score"], "A_second": q["second"],
                       "A_iou": iou, "A_temps_ms": round(q["temps_ms"], 1), "A_zoom": q["zoom"],
                       "B_pred": pb, "B_distance": round(sb[1], 4)})
    with open(res / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    m_a = metriques(y_test, pred_a, CONNUS)
    m_b = metriques(y_test, pred_b, CONNUS)
    ious = np.array([r["A_iou"] for r in lignes if r["A_iou"] != ""], float)
    m_a.update({"iou_moyen": float(ious.mean()), "iou_median": float(np.median(ious)),
                "part_iou_sup_0_5": float((ious > 0.5).mean()),
                "temps_ms_moyen": float(np.mean([q["temps_ms"] for q in p_test]))})
    m_b.update({"temps_ms_moyen": float(np.mean(t_desc) + t_knn)})

    cats = np.array([l["categorie"] for l in test])
    sources = np.array([l["source"] for l in test])
    par_cat = []
    for c in dict.fromkeys(l["categorie"] for l in test):
        mpos = (cats == c) & (y_test != INCONNU)
        mneg = (cats == c) & (y_test == INCONNU)
        par_cat.append({"categorie": c,
                        "A_identification": float((pred_a[mpos] == y_test[mpos]).mean()),
                        "B_identification": float((pred_b[mpos] == y_test[mpos]).mean()),
                        "A_fausse_acceptation": float((pred_a[mneg] != INCONNU).mean()) if mneg.any() else "",
                        "B_fausse_acceptation": float((pred_b[mneg] != INCONNU).mean()) if mneg.any() else ""})
    with open(res / "test_par_categorie.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(par_cat[0].keys()))
        w.writeheader()
        w.writerows(par_cat)

    par_classe = []
    for c in CONNUS:
        tp_a = int(((pred_a == c) & (y_test == c)).sum()); tp_b = int(((pred_b == c) & (y_test == c)).sum())
        par_classe.append({"classe": c, "support": int((y_test == c).sum()),
                           "A_precision": tp_a / max(1, int((pred_a == c).sum())), "A_rappel": tp_a / max(1, int((y_test == c).sum())),
                           "B_precision": tp_b / max(1, int((pred_b == c).sum())), "B_rappel": tp_b / max(1, int((y_test == c).sum()))})
    with open(res / "test_par_classe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(par_classe[0].keys()))
        w.writeheader()
        w.writerows(par_classe)

    # négatifs acceptés à tort : par billet inconnu (cousins des billets connus ?)
    fa = {}
    for s, yp_a, yp_b, yv in zip(sources, pred_a, pred_b, y_test):
        if yv == INCONNU:
            d = fa.setdefault(s, {"source": s, "n": 0, "A_acceptes": 0, "B_acceptes": 0, "A_classes": set(), "B_classes": set()})
            d["n"] += 1
            if yp_a != INCONNU:
                d["A_acceptes"] += 1; d["A_classes"].add(yp_a)
            if yp_b != INCONNU:
                d["B_acceptes"] += 1; d["B_classes"].add(yp_b)
    with open(res / "test_negatifs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "n", "A_acceptes", "B_acceptes", "A_classes", "B_classes"])
        w.writeheader()
        for d in fa.values():
            w.writerow({**d, "A_classes": " ".join(sorted(d["A_classes"])), "B_classes": " ".join(sorted(d["B_classes"]))})

    cm_a, etiq = matrice_confusion(y_test, pred_a, CONNUS)
    cm_b, _ = matrice_confusion(y_test, pred_b, CONNUS)
    synthese = {
        "classes_connues": CONNUS,
        "tailles": {"validation": len(val), "test": len(test), "apprentissage_b": len(app),
                    "test_positifs": int((y_test != INCONNU).sum()), "test_negatifs": int((y_test == INCONNU).sum())},
        "A": {"configuration": nom_choix, "relance_zoom": bool(zoom_utile),
              "parametres": {k: v for k, v in params_choix.items()}, "tau": tau_a, "test": m_a},
        "B": {"descripteurs": nom_b, "k": k_b, "delta": delta_b, "test": m_b},
        "matrices": {"etiquettes": etiq, "A": cm_a.tolist(), "B": cm_b.tolist()},
        "courbe_seuil_test_A": courbe_seuil(y_test, brutes_a, scores_a, taus),
    }
    json.dump(synthese, open(res / "synthese.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({"A": synthese["A"], "B": synthese["B"]}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
