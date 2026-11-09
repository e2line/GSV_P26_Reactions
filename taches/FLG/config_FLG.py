from pathlib import Path

class Config_COL:
    CORPS_TANGIBLE = 50     # rayon corps tangible
    I = 6                   # intensité du pseudo-haptique
    O = 0.4                 # ocilations
    XLSX_DIR = "XLSX"     # dossier d'entrée, contenant les excels
    CACHE_DIR = "cache/COL"     # dossier des caches
    CONDITIONS = ['RR', 'Rr', 'rR', 'rr']   # Conditions d'intensités de corps possibles, ici toujours Bosses

    # Dicitonnaire avec pour clés les numéros des différentes conditions et pour valeurs les intensités des conditions de corps correspondantes
    DICT_COND = {
                'C01': 'Rr',
                'C02': 'rr',
                'C03': 'rR',
                'C04': 'RR',
                'C05': 'Rr',
                'C06': 'rr',
                'C07': 'rR',
                'C08': 'RR'
            }

    # Conditions où les rôles sont inversés, ici aucun car A et B ont les mêmes buts et dispositions
    COND_INVERSES = set()