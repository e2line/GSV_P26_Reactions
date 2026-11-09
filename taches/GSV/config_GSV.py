from pathlib import Path

class Config_GSV:
    CORPS_TANGIBLE = 50     # rayon corps tangible
    I = 3                   # intensité du pseudo-haptique
    O = 0.4                 # ocilations
    XLSX_DIR = "XLSX"     # dossier d'entrée, contenant les excels
    CACHE_DIR = "cache/GSV"     # dossier des caches
    CONDITIONS = ['BB', 'BH', 'HB', 'HH', 'GB', 'GH']   # Conditions de corps possibles

    # Colonnes de positions rennomées pour avoir guide et suiveur sans mélanger sujet A et B
    COL_GR = ("A-rX", "A-rY")
    COL_GT = ("A-tX", "A-tY")
    COL_SR = ("B-rX", "B-rY")
    COL_ST = ("B-tX", "B-tY")

    # Dicitonnaire avec pour clés les numéros des différentes conditions et pour valeurs les conditions de corps correspondantes
    DICT_COND = {
            'C01': 'HB',
            'C02': 'BB',
            'C03': 'GB',
            'C04': 'HH',
            'C05': 'HB',
            'C06': 'GH',
            'C07': 'BB',
            'C08': 'BH',
            'C09': 'HH',
            'C10': 'BH'
        }

    # Conditions où B est guide et A suiveur
    COND_INVERSES = {'C01', 'C04', 'C07', 'C10'}