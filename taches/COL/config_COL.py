from pathlib import Path

class Config_COL:
    CORPS_TANGIBLE = 50     # rayon corps tangible
    I = 3                   # intensité du pseudo-haptique
    O = 0.4                 # ocilations
    XLSX_DIR = "XLSX"     # dossier d'entrée, contenant les excels
    CACHE_DIR = "cache/COL"     # dossier des caches
    CONDITIONS = ['HH']   # Conditions de corps possibles

    # Dicitonnaire avec pour clés les numéros des différentes conditions et pour valeurs les conditions de corps correspondantes
    DICT_COND = {
            'C01': 'HH',
            'C02': 'HH',
            'C03': 'HH',
            'C04': 'HH'
        }

    # Conditions où les rôles sont inversés, ici aucun car A et B ont les mêmes buts et dispositions
    COND_INVERSES = set()