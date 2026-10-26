from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import json


# CONFIGURATION

class Config:
    CACHE_ROOT = Path("GSV/caches")
    OUTPUT_DIR = Path("caches")
    CONDITIONS = ['BB', 'BH', 'HB', 'HH']
    POSITIONS = ['tt', 'rt', 'tr', 'rr']
    PAS_DIRECTION = 20  # 200 ms
    DT_MS = 10



# CALCUL DES MÉTRIQUES

def vitesse_signee(df, col_pos_source, col_nom, role, pas_direction=20, dt_ms=10):
    """
    Calcule la vitesse signée d'un sujet vers une cible.
    
    Paramètres
    ----------
    df : DataFrame
    col_pos_source : tuple (x, y) de la position du sujet
    col_pos_cible : tuple (x, y) de la position de la cible
    col_nom : str, nom de la colonne à créer
    pas_direction : int, nombre de frames pour le delta
    dt_ms : float, durée d'une frame
    
    Retour
    ------
    df avec la nouvelle colonne
    """
    x_src, y_src = col_pos_source
    
    # Delta de position
    dx = df[x_src].shift(-pas_direction//2) - df[x_src].shift(pas_direction//2)
    dy = df[y_src].shift(-pas_direction//2) - df[y_src].shift(pas_direction//2)
    
    # Direction vers la cible
    if role == 'A':
        # Axe : A -> B
        ux = df['B-tX'] - df['A-tX']
        uy = df['B-tY'] - df['A-tY']
    else: 
        # Axe : B -> A
        ux = df['A-tX'] - df['B-tX']
        uy = df['A-tY'] - df['B-tY']

    norme_u = np.sqrt(ux**2 + uy**2)
    
    # Projection
    proj = (dx * ux + dy * uy) / (norme_u + 1e-9)
    
    # Vitesse signée
    dt = pas_direction * dt_ms
    df[col_nom] = proj / dt

    # Mettre NaN sur les premières et dernières lignes (pas d'historique)
    df.loc[:pas_direction//2-1, col_nom] = np.nan
    df.loc[len(df)-pas_direction//2:, col_nom] = np.nan
    
    return df


def acceleration(df, col_vitesse, col_nom, pas_direction=20, dt_ms=10):
    """
    Calcule l'accélération à partir d'une vitesse signée.
    """
    dv = df[col_vitesse].shift(-pas_direction//2) - df[col_vitesse].shift(pas_direction//2)
    dt = pas_direction * dt_ms
    df[col_nom] = dv / dt

    # Mettre NaN sur les premières et dernières lignes (pas d'historique)
    df.loc[:pas_direction//2-1, col_nom] = np.nan
    df.loc[len(df)-pas_direction//2:, col_nom] = np.nan
    return df


def interpoler_paliers(df, cols_xy, col_t='temps_relatif_ms'):
    t = df[col_t].to_numpy(float)
    for cx, cy in cols_xy:
        maj = ((df[cx].diff().fillna(1) != 0) | (df[cy].diff().fillna(1) != 0)).to_numpy()
        for c in (cx, cy):
            df[c] = np.interp(t, t[maj], df[c].to_numpy(float)[maj])
    return df


def enrichir_dataframe(df, pos='tt', pas_direction=20):
    """
    Enrichit un DataFrame avec les nouvelles métriques pour une position.
    """
    # Mapping des colonnes selon la position
    mapping = {
        'tt': {
            'A': ('A-tX', 'A-tY'),
            'B': ('B-tX', 'B-tY'),
        },
        'rt': {
            'A': ('A-rX', 'A-rY'),
            'B': ('B-tX', 'B-tY'),
        },
        'tr': {
            'A': ('A-tX', 'A-tY'),
            'B': ('B-rX', 'B-rY'),
        },
        'rr': {
            'A': ('A-rX', 'A-rY'),
            'B': ('B-rX', 'B-rY'),
        },
    }
    
    pos_A = mapping[pos]['A']
    pos_B = mapping[pos]['B']

    df = interpoler_paliers(df, [pos_A, pos_B])
    
    # Vitesses signées
    df = vitesse_signee(df, pos_A, f'v_signed_A_vers_B_{pos}', 'A', pas_direction)
    df = vitesse_signee(df, pos_B, f'v_signed_B_vers_A_{pos}', 'B', pas_direction)
    
    # Accélérations
    df = acceleration(df, f'v_signed_A_vers_B_{pos}', f'acc_A_vers_B_{pos}', pas_direction)
    df = acceleration(df, f'v_signed_B_vers_A_{pos}', f'acc_B_vers_A_{pos}', pas_direction)
    
    return df


# PIPELINE

def enrichir_caches(pas_direction=20):
    """Enrichit tous les caches avec les nouvelles métriques."""
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for cond in Config.CONDITIONS:
        cache_dir = Config.CACHE_ROOT / f"cache_dataframes_{cond}"
        print(cache_dir)
        if not cache_dir.exists():
            continue
        print("dossier trouvé")
        # Charger les métadonnées
        with open(cache_dir / "metadatas.json") as f:
            metas = json.load(f)
        
        for nom_fichier in metas.keys():
            print(f"traitement {nom_fichier}")
            nom_safe = nom_fichier.replace('/', '_').replace('\\', '_')
            pq_file = cache_dir / f"{nom_safe}.parquet"
            if not pq_file.exists():
                continue
            
            df = pd.read_parquet(pq_file)
            
            # Enrichir pour les 4 positions
            for pos in Config.POSITIONS:
                df = enrichir_dataframe(df, pos, pas_direction)
            
            # Sauvegarder dans le nouveau dossier
            output_file = Config.OUTPUT_DIR / f"{cond}_{nom_safe}_{pas_direction}.parquet"
            df.to_parquet(output_file, index=False)
            
            print(f"✅ {cond} / {nom_fichier}")


if __name__ == "__main__":
    enrichir_caches()
