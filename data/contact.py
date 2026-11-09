import numpy as np
import pandas as pd

from config import Config

PICOT_SIZE = Config.CORPS_PERCEVANT // 4
RAYON_CORPS_PERCU = Config.CORPS_PERCU

PICOT_OFFSETS = np.array(
    [[dx, dy]
     for dy in (-1.5, -0.5, 0.5, 1.5)
     for dx in (-1.5, -0.5, 0.5, 1.5)],
    dtype=np.float32,
) * PICOT_SIZE


def distance_point_carre(px, py, cx, cy, demi_cote=40):
    """
    Calcule la distance point-carré pour la distance entre l'avatar corps perçu et la matrice corps percevant 
    """
    x_min, x_max = cx - demi_cote, cx + demi_cote
    y_min, y_max = cy - demi_cote, cy + demi_cote
    px_proche = np.clip(px, x_min, x_max)
    py_proche = np.clip(py, y_min, y_max)
    return np.sqrt((px - px_proche) ** 2 + (py - py_proche) ** 2)


def contact_binaire(tt_sujet, tt_partenaire, rayon=RAYON_CORPS_PERCU):
    """
    Recalcule les contacts de manière géométrique pour éviter l'artefact de contact brut
    Renvoie un résultat binaire : contact ou pas contact, correspondant à si un ou des picots sont activés
    """
    d = distance_point_carre(tt_partenaire[:, 0], tt_partenaire[:, 1],
                             tt_sujet[:, 0], tt_sujet[:, 1])
    return (d <= rayon).astype(np.int8)


def contact_AB(df, rayon=RAYON_CORPS_PERCU):
    tt_A = df[list(Config.COL_AT)].to_numpy(np.float32)
    tt_B = df[list(Config.COL_BT)].to_numpy(np.float32)
    A_percoit_B = contact_binaire(tt_A, tt_B, rayon)
    B_percoit_A = contact_binaire(tt_B, tt_A, rayon)
    return (A_percoit_B | B_percoit_A).astype(np.int8)


def picots_activation(curseur_xy, partenaire_xy, offsets=PICOT_OFFSETS, rayon=RAYON_CORPS_PERCU):
    """
    Recalcule l'activation de chaque picot suivant les positions
    """
    centres = curseur_xy[:, None, :] + offsets[None, :, :]
    dx = np.abs(partenaire_xy[:, None, 0] - centres[..., 0]) - PICOT_SIZE / 2
    dy = np.abs(partenaire_xy[:, None, 1] - centres[..., 1]) - PICOT_SIZE / 2
    dx = np.clip(dx, 0, None)
    dy = np.clip(dy, 0, None)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return (dist <= rayon).astype(np.float32)

def detecter_ruptures_contact(df: pd.DataFrame, col_contact: str,
                               col_temps: str = "temps_relatif_ms",
                               duree_min_ms: float = 0) -> tuple[np.ndarray, np.ndarray]:
    diff = df[col_contact].diff()
    idx_ruptures = df.index[diff == -1].tolist()
    idx_reprises = df.index[diff == 1].tolist()

    temps = pd.to_numeric(df[col_temps], errors='coerce').to_numpy()

    ruptures_valides = []
    reprises_valides = []

    for idx_r in idx_ruptures:
        suivant = next((i for i in idx_reprises if i > idx_r), None)
        fin_idx = suivant if suivant is not None else len(df) - 1
        duree_ms = temps[fin_idx] - temps[idx_r]

        if duree_ms >= duree_min_ms:
            ruptures_valides.append(idx_r)
            if suivant is not None:
                reprises_valides.append(suivant)

    return np.array(ruptures_valides), np.array(reprises_valides)