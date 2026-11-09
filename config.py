from pathlib import Path
import importlib

class Config:
    DT = 10                 # pas brut en ms
    ESPACE_X = 800          # nombres de pixels de l'espace virtuel en x
    ESPACE_Y = 800          # nombres de pixels de l'espace virtuel en y
    CORPS_PERCEVANT = 4*20  # taille de la matrice (chaps récepteur)
    CORPS_PERCU = 20        # rayon du corps perçu
    GRAINE = 42             # graine pour répéter l'aléatoire
    DECOUPAGE = 5           # nombres de plis de binomes
    PAS_DIRECTION = 20      # Pas en ms par défaut pour les fenêtres glissantes
    WINDOW_SMOOTH = 30
    POSITIONS = ['tt', 'rt', 'tr', 'rr']                # Positions utilisées (curseur, souris)

    DT_MODEL = 50
    BIN = DT_MODEL // DT

    DING_LARGEUR_MS = 200

    RUPTURE_AVANT_MS = 500
    RUPTURE_APRES_MS = 500

    RESULTATS_DIR = Path("resultats")
    FOLDS_PATH = Path("folds.json")
    FOLDS_SHA256 = "32ff95467c0f9470afc0952791a08327c04f14952e7afff095edfcb1f957b4fb"

    HISTORIQUE_S = [0.5, 1.0, 2.0]
    ALPHAS_RIDGE = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    COL_T = "Tps(ms)"

    # Colonnes positions des sujets A et B
    COL_AR = ("A-rX", "A-rY")
    COL_AT = ("A-tX", "A-tY")
    COL_BR = ("B-rX", "B-rY")
    COL_BT = ("B-tX", "B-tY")

    @classmethod
    def charger_tache(cls, tache):
        """Charge les paramètres spécifiques à la tâche depuis
        config_taches/config_<nom>.py (classe Config_<Nom>).
        """
        if not tache:
            raise ValueError("tache vide")
        nom_module = tache.lower()
        nom_classe = f"Config_{tache}"
        chemin = f"taches.{tache}.config_{tache}"

        try:
            module = importlib.import_module(chemin)
        except ModuleNotFoundError as e:
            raise ValueError(
                f"Tâche {tache!r} : module {chemin!r} introuvable. "
                f"Créez config_taches/config_{nom_module}.py."
            ) from e

        if not hasattr(module, nom_classe):
            raise ValueError(
                f"Tâche {tache!r} : classe {nom_classe!r} absente de "
                f"{chemin}. Définissez-la."
            )

        classe = getattr(module, nom_classe)
        for attr in dir(classe):
            if attr.isupper():
                setattr(cls, attr, getattr(classe, attr))
        cls.TACHE = tache
        return cls