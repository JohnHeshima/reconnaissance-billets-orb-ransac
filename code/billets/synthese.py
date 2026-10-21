"""Génération contrôlée de « photos » de billets pour l'évaluation.

Chaque photo simulée place un billet (image de référence) dans une scène :
fond (table, tissu, fond encombré), transformation projective (position,
échelle, rotation, inclinaison de la caméra), puis une dégradation dont on
contrôle le type. On connaît donc exactement la vérité terrain : la coupure
et le quadrilatère occupé par le billet (pour l'IoU).

Catégories de difficulté (une par photo) :
  facile, rotation, perspective, echelle, eclairage, flou, occlusion,
  froisse, compression
"""

import cv2
import numpy as np

CATEGORIES = ["facile", "rotation", "perspective", "echelle", "eclairage",
              "flou", "occlusion", "froisse", "compression"]

TAILLE = (768, 1024)  # hauteur, largeur des photos simulées


# ---------------------------------------------------------------- fonds
def _bruit_lisse(rng, h, w, echelle):
    petit = rng.random((max(2, h // echelle), max(2, w // echelle))).astype(np.float32)
    return cv2.resize(petit, (w, h), interpolation=cv2.INTER_CUBIC)


def fond_aleatoire(rng, h=TAILLE[0], w=TAILLE[1]):
    genre = rng.choice(["uni", "bois", "tissu", "encombre"])
    if genre == "bois":
        y = np.arange(h, dtype=np.float32)[:, None]
        n = _bruit_lisse(rng, h, w, 40) * 25
        veines = np.sin((y + n) / rng.uniform(4, 9)) * 0.5 + 0.5
        base = np.array([rng.uniform(40, 90), rng.uniform(80, 130), rng.uniform(120, 180)], np.float32)
        img = base[None, None, :] * (0.75 + 0.35 * veines[..., None])
    elif genre == "tissu":
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        p = rng.uniform(5, 14)
        trame = 0.5 + 0.25 * np.sin(x / p) + 0.25 * np.sin(y / p)
        base = rng.uniform(40, 220, 3).astype(np.float32)
        img = base[None, None, :] * (0.7 + 0.4 * trame[..., None])
    else:
        base = rng.uniform(30, 235, 3).astype(np.float32)
        grad = _bruit_lisse(rng, h, w, 200)[..., None]
        img = base[None, None, :] * (0.8 + 0.4 * grad)
    img = img + rng.normal(0, 4, (h, w, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    if genre == "encombre":
        _encombrer(rng, img)
    return img


def _encombrer(rng, img):
    """Ajoute des objets parasites : formes, traits, texte imprimé."""
    h, w = img.shape[:2]
    for _ in range(rng.integers(6, 18)):
        couleur = tuple(int(c) for c in rng.integers(0, 256, 3))
        t = rng.integers(0, 4)
        if t == 0:
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (p1[0] + int(rng.integers(-250, 250)), p1[1] + int(rng.integers(-250, 250)))
            cv2.rectangle(img, p1, p2, couleur, -1)
        elif t == 1:
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(10, 120)), couleur, -1)
        elif t == 2:
            cv2.line(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                     (int(rng.integers(0, w)), int(rng.integers(0, h))), couleur, int(rng.integers(1, 8)))
        else:
            mots = ["RECU", "TOTAL 25000", "KINSHASA", "FACTURE", "MERCI", "N 004512", "TVA 16%"]
            cv2.putText(img, str(rng.choice(mots)), (int(rng.integers(0, w - 200)), int(rng.integers(30, h))),
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 2.0)), couleur, int(rng.integers(1, 4)))


# ---------------------------------------------------------------- géométrie
def _quadrilatere(rng, w_ref, h_ref, categorie, H, W):
    """Tire le quadrilatère image où le billet sera projeté."""
    if categorie == "echelle":
        frac = rng.uniform(0.16, 0.26)
    elif categorie == "facile":
        frac = rng.uniform(0.5, 0.75)
    else:
        frac = rng.uniform(0.35, 0.7)
    larg = frac * W
    haut = larg * h_ref / w_ref
    if categorie == "rotation":
        angle = rng.uniform(0, 360)
    elif categorie == "facile":
        angle = rng.uniform(-12, 12)
    else:
        angle = rng.uniform(-35, 35) + (180 if rng.random() < 0.25 else 0)
    coins = np.float32([[-larg / 2, -haut / 2], [larg / 2, -haut / 2], [larg / 2, haut / 2], [-larg / 2, haut / 2]])
    # inclinaison de la caméra : déplacement des coins (perspective)
    if categorie == "perspective":
        force = rng.uniform(0.22, 0.38)
    elif categorie == "facile":
        force = rng.uniform(0.0, 0.05)
    else:
        force = rng.uniform(0.02, 0.12)
    cote = rng.integers(0, 4)  # côté qui « s'éloigne » de la caméra
    retrecit = np.ones(4, np.float32)
    retrecit[[cote, (cote + 1) % 4]] = 1 - force
    coins = coins * retrecit[:, None]
    coins += rng.normal(0, 0.02 * larg, coins.shape)
    a = np.deg2rad(angle)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    coins = coins @ R.T
    # translation : le billet reste en grande partie dans l'image
    ext = coins.max(axis=0) - coins.min(axis=0)
    marge_x = max(0.0, (W - ext[0]) / 2) + 0.06 * ext[0]
    marge_y = max(0.0, (H - ext[1]) / 2) + 0.06 * ext[1]
    centre = np.float32([W / 2 + rng.uniform(-marge_x, marge_x), H / 2 + rng.uniform(-marge_y, marge_y)])
    return (coins + centre).astype(np.float32)


def _froisser(rng, billet):
    """Déformation élastique lisse + ombrage de plis (billet froissé / plié)."""
    h, w = billet.shape[:2]
    amp = rng.uniform(4, 10)
    dx = (_bruit_lisse(rng, h, w, 60) - 0.5) * 2 * amp
    dy = (_bruit_lisse(rng, h, w, 60) - 0.5) * 2 * amp
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(billet, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ombre = 1 + 0.9 * (cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=5) / 100.0)
    for _ in range(rng.integers(1, 3)):  # plis nets
        x0 = rng.uniform(0.2, 0.8) * w
        bande = np.exp(-((x - x0) ** 2) / (2 * rng.uniform(2, 6) ** 2))
        ombre = ombre - 0.25 * bande
    out = np.clip(out.astype(np.float32) * np.clip(ombre, 0.55, 1.25)[..., None], 0, 255)
    return out.astype(np.uint8)


def _usure(rng, billet):
    """Billet un peu usé : désaturation, jaunissement, salissures."""
    hsv = cv2.cvtColor(billet, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.75, 1.0)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    tache = _bruit_lisse(rng, *billet.shape[:2], 30)
    out *= (1 - 0.12 * tache[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- photométrie
def _eclairage(rng, img):
    img = img.astype(np.float32) / 255
    choix = rng.integers(0, 3)
    if choix == 0:  # sous-exposition (pièce sombre)
        img = img ** rng.uniform(1.8, 2.6) * rng.uniform(0.55, 0.8)
    elif choix == 1:  # ombre portée en dégradé
        h, w = img.shape[:2]
        a = rng.uniform(0, 2 * np.pi)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (np.cos(a) * (x / w - 0.5) + np.sin(a) * (y / h - 0.5)) * rng.uniform(2, 4)
        img = img * (0.25 + 0.75 / (1 + np.exp(-t)))[..., None]
    else:  # lumière chaude d'ampoule + faible contraste
        img = img * np.float32([0.65, 0.85, 1.1]) * 0.8 + 0.12
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _flou(rng, img):
    if rng.random() < 0.5:
        s = rng.uniform(2.0, 3.5)
        return cv2.GaussianBlur(img, (0, 0), s)
    n = int(rng.integers(11, 19))
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1
    M = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1)
    k = cv2.warpAffine(k, M, (n, n))
    return cv2.filter2D(img, -1, k / k.sum())


def _occlusion(rng, img, quad):
    """Doigts / main posés sur le billet : 20 à 40 % de sa surface cachée."""
    masque_billet = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(masque_billet, quad.astype(np.int32), 255)
    aire = masque_billet.sum() / 255
    cible = rng.uniform(0.2, 0.4) * aire
    cache = np.zeros_like(masque_billet)
    peau = np.array([rng.uniform(40, 110), rng.uniform(70, 140), rng.uniform(110, 200)])
    essais = 0
    while (cv2.bitwise_and(cache, masque_billet).sum() / 255) < cible and essais < 12:
        c = quad[rng.integers(0, 4)] * 0.4 + quad.mean(axis=0) * 0.6 + rng.normal(0, 40, 2)
        axes = (int(rng.uniform(40, 90)), int(rng.uniform(120, 260)))
        cv2.ellipse(cache, (int(c[0]), int(c[1])), axes, rng.uniform(0, 180), 0, 360, 255, -1)
        essais += 1
    m = cache.astype(bool)
    teinte = np.clip(peau + rng.normal(0, 6, (m.sum(), 3)), 0, 255)
    img[m] = teinte.astype(np.uint8)
    return img


def _compression(rng, img):
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, rng.uniform(6, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(12, 25))])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.resize(img, (TAILLE[1], TAILLE[0]), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- scène complète
def generer_photo(rng, billet, categorie, distracteur=None):
    """Compose une photo simulée.

    billet : image BGR du billet à placer (None pour une scène sans billet).
    distracteur : image d'un autre billet (non reconnu) à poser dans la scène.
    Renvoie (photo, quadrilatère du billet ou None).
    """
    H, W = TAILLE
    photo = fond_aleatoire(rng)
    if distracteur is not None:
        q = _quadrilatere(rng, distracteur.shape[1], distracteur.shape[0], "facile", H, W)
        _coller(photo, distracteur, q * rng.uniform(0.5, 0.8) + rng.uniform(0, 200, 2).astype(np.float32))
    quad = None
    if billet is not None:
        b = billet
        if rng.random() < 0.4:
            b = _usure(rng, b)
        if categorie == "froisse":
            b = _froisser(rng, b)
        quad = _quadrilatere(rng, b.shape[1], b.shape[0], categorie, H, W)
        _coller(photo, b, quad)
    if categorie == "eclairage":
        photo = _eclairage(rng, photo)
    elif categorie == "flou":
        photo = _flou(rng, photo)
    elif categorie == "occlusion" and quad is not None:
        photo = _occlusion(rng, photo, quad)
    elif categorie == "compression":
        photo = _compression(rng, photo)
    # bruit de capteur et compression JPEG légers, toujours présents
    photo = np.clip(photo + rng.normal(0, 2.5, photo.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(80, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), quad


def _coller(photo, billet, quad):
    h, w = billet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    H, W = photo.shape[:2]
    deforme = cv2.warpPerspective(billet, M, (W, H), flags=cv2.INTER_LINEAR)
    masque = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    masque = cv2.GaussianBlur(masque, (3, 3), 0).astype(np.float32)[..., None] / 255
    photo[:] = (photo * (1 - masque) + deforme * masque).astype(np.uint8)
