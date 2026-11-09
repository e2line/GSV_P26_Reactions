import numpy as np
import pandas as pd

from config import Config


def cible_atteinte(df, meta):
    """
    Renvoie l'indice du premier contact avec la cible si atteinte suivant les règles de réussite :
    Les deux doivent toucher la cible à la fois à moins de rayon de leur corps perçu px
    """
    cxA = meta.get('cible_A_x')
    cyA = meta.get('cible_A_y')
    if Config.TACHE != 'GSV' and (cxA is None or cyA is None):
        return None
    tt_guide = df[list(Config.COL_GT)].to_numpy(np.float32)
    tt_suiveur = df[list(Config.COL_ST)].to_numpy(np.float32)
    dAG = np.linalg.norm(tt_guide - np.array([cxA, cyA], np.float32), axis=1)
    dAS = np.linalg.norm(tt_suiveur - np.array([cxA, cyA], np.float32), axis=1)
    res = np.where((dAG <= Config.CORPS_PERCU) & (dAS <= Config.CORPS_PERCU))[0]
    # Si on est en guide-guide, on teste aussi la cible B
    if meta['condition'] in ['GB', 'GH']:
        cxB = meta.get('cible_B_x')
        cyB = meta.get('cible_B_y')
        if cxB is None or cyB is None:
            return None
        dBG = np.linalg.norm(tt_guide - np.array([cxB, cyB], np.float32), axis=1)
        dBS = np.linalg.norm(tt_suiveur - np.array([cxB, cyB], np.float32), axis=1)
        if meta['condition'] in ['BB', 'BH', 'HB', 'HH']:
            res = res or np.where((dBG <= Config.CORPS_PERCU) & (dBS <= Config.CORPS_PERCU))[0]
    return int(res[0]) if len(res) else None

def ding(df, meta):
    """
    Ajoute l'indice de réussite aux métadonnées
    """
    meta['ding_idx'] = cible_atteinte(df, meta)
    if meta['ding_idx'] is not None:
        meta['t_ding_s'] = meta['ding_idx'] * Config.DT / 1000
    return df, meta

def verif_ding(dfs, metas):
    rows = []
    for f, meta in metas.items():
        df = dfs[f]
        tt = df[list(Config.COL_ST)].to_numpy(np.float32)
        cx = meta.get('cible_x')
        cy = meta.get('cible_y')
        if cx is not None and cy is not None:
            d_fin = float(np.linalg.norm(tt[-1] - np.array([cx, cy], np.float32)))
        else:
            d_fin = np.nan
        rows.append({
            'fichier': f,
            'binome': meta.get('binome'),
            'condition': meta.get('condition'),
            'cible': (cx, cy) if cx is not None else None,
            'ding_idx': meta.get('ding_idx'),
            't_ding_s': meta.get('t_ding_s'),
            'duree_s': meta.get('duree_secondes'),
            'dist_finale_suiveur': d_fin,
        })
    return pd.DataFrame(rows)