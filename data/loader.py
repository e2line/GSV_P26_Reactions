import re
from pathlib import Path

import openpyxl
import pandas as pd

from config import Config
from data.cleaning import tronquer_debut


def extraire_binome(chaine:str):
    """
    Récupère le numéro de binôme dans le nom du fichier
    """
    binome = re.search(r'(B\d{1,3})', str(chaine))
    return binome.group(1) if binome else None


def extraire_num_condition(chaine:str):
    """
    Récupère le numéro de condition dans le nom du fichier
    """
    cond = re.search(r'(C\d{1,3})', str(chaine))
    return cond.group(1) if cond else None


def parser_cible(cellule:str):
    """
    Récupère les positions de la cible avec la cellule d'info complément
    """
    if Config.TACHE != 'GSV' or not cellule:
        return None, None
    mx = re.search(r'(\d+)\s*[xX]', str(cellule))
    my = re.search(r'[xX]\s*(\d+)', str(cellule))
    cx = int(mx.group(1)) if mx else None
    cy = int(my.group(1)) if my else None
    return cx, cy


def charger_essai(path, roles_inverses=False, tronquer_debut_essai=True, seuil_position=5):
    """
    Récupère les cellules utiles depuis les fichiers essais excels
    """
    path = Path(path)
    fichier = path.stem

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        cellule_D15 = wb.active['D15'].value
        wb.close()
    except Exception as e:
        print(f"⚠️  {fichier} - D15 (cellule cible A) illisible : {e}")
        cellule_D15 = None

    try:
        wb = openpyxl.load_workbook(path, data_only=True)
        cellule_N15 = wb.active['N15'].value
        wb.close()
    except Exception as e:
        print(f"⚠️  {fichier} - N15 (cellule cible B) illisible : {e}")
        cellule_N15 = None

    df = pd.read_excel(
        path, skiprows=19, nrows=3001,
        usecols='B, D, E, F, G, J, K, L, M, N, O, P, Q, T, U, V, W',
    )

    if roles_inverses:
        _inverser_roles(df)

    df['temps_relatif_ms'] = df[Config.COL_T] - df[Config.COL_T].iloc[0]

    premier_index = 0
    if tronquer_debut_essai:
        premier_index = tronquer_debut(df, seuil_position=seuil_position)
        if premier_index > 0:
            df = df.iloc[premier_index:].reset_index(drop=True)
            df['temps_relatif_ms'] = df[Config.COL_T] - df[Config.COL_T].iloc[0]

    cible_A_x, cible_A_y = parser_cible(cellule_D15)
    cible_B_x, cible_B_y = parser_cible(cellule_N15)
    binome = extraire_binome(path.parent.name) or extraire_binome(fichier)
    num_cond = extraire_num_condition(path.parent.name) or extraire_num_condition(fichier)
    condition = Config.DICT_COND.get(num_cond)
    meta = {
        'fichier': fichier,
        'chemin': str(path),
        'binome': binome,
        'num_cond': num_cond,
        'condition': condition,
        'nb_lignes': len(df),
        'index_coupe': premier_index,
        'duree_secondes': len(df) * Config.DT / 1000.0,
        'cible_A_x': cible_A_x,
        'cible_A_y': cible_A_y,
        'cible_B_x': cible_B_x,
        'cible_B_y': cible_B_y,
        'roles_inverses': roles_inverses,
        'tronquer_debut': tronquer_debut_essai,
    }
    return df, meta


def _inverser_roles(df):
    """
    Inverse les roles quand A est suiveur et B guide
    """
    cols_A = ['A-tX', 'A-tY', 'A-rX', 'A-rY', 'A-Red']
    cols_B = ['B-tX', 'B-tY', 'B-rX', 'B-rY', 'B-Blue']
    for ca, cb in zip(cols_A, cols_B):
        tmp = df[ca].copy()
        df[ca] = df[cb].values
        df[cb] = tmp.values