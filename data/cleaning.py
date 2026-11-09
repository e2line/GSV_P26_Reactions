import numpy as np
import pandas as pd

from config import Config


def tronquer_debut(df, seuil_position=5):
    """
    Tronque le début de l'essai, c'est-à-dire avant que les positions curseurs soit bien au milieu 
    de l'espace à seuil_position prêt, afin d'éviter les débuts avec données non encore chargées
    """
    centre = Config.ESPACE_X / 2
    demarrage = (
        ((df[Config.COL_AR[0]] - centre).abs() < seuil_position)
        & ((df[Config.COL_AR[1]] - centre).abs() < seuil_position)
        & ((df[Config.COL_BR[0]] - centre).abs() < seuil_position)
        & ((df[Config.COL_BR[1]] - centre).abs() < seuil_position)
    )
    if not demarrage.any():
        return 0
    return int(df[demarrage].index[0])


def nettoyer_essai(df, bornes=None):
    """
    Retire les colonnes positions NaN et les positions curseurs en dehors de l'espace
    """
    bornes = bornes or (0, Config.ESPACE_X, 0, Config.ESPACE_Y)
    x_min, x_max, y_min, y_max = bornes

    df = df.copy()

    cols_pos = [c for c in df.columns
                if c.endswith('-rX') or c.endswith('-rY')
                or c.endswith('-tX') or c.endswith('-tY')]
    df = df.dropna(subset=cols_pos).reset_index(drop=True)

    m = np.ones(len(df), bool)
    for cx, cy in [Config.COL_AT, Config.COL_BT]:
        m &= df[cx].between(x_min, x_max).to_numpy()
        m &= df[cy].between(y_min, y_max).to_numpy()
    df = df[m].reset_index(drop=True)

    df['temps_relatif_ms'] = df[Config.COL_T] - df[Config.COL_T].iloc[0]
    return df


def interpoler_paliers(df, cols_xy, col_t='temps_relatif_ms'):
    """
    Interpole les positions entre deux changements afin d'éviter les changements bruts entre les enregistrements
    """
    t = df[col_t].to_numpy(float)
    for cx, cy in cols_xy:
        maj = ((df[cx].diff().fillna(1) != 0)
               | (df[cy].diff().fillna(1) != 0)).to_numpy()
        for c in (cx, cy):
            df[c] = np.interp(t, t[maj], df[c].to_numpy(float)[maj])
    return df