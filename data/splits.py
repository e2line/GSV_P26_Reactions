import hashlib
import json
from pathlib import Path

import numpy as np

from config import Config


def _hash_fichier(path):
    """
    Récupération du hash
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_folds(metadatas, seed=None, n_splits=None, out=None):
    """
    Construction des plis de binômes de manière aléatoire avec une graine pour avoir toujours les mêmes plis
    Affiche une erreur si le fichier est déjà créé afin de ne pas changer les plis en cours de route pour
    éviter le p-hacking
    Enregistre les plis dans un fichier json
    """
    seed = seed if seed is not None else Config.GRAINE
    n_splits = n_splits or Config.DECOUPAGE
    out = Path(out or Config.FOLDS_PATH)

    if out.exists():
        raise RuntimeError(
            f"{out} existe déjà. Découpage pré-enregistré : ne pas écraser."
        )

    binome_of_file = {f: m.get('binome') for f, m in metadatas.items()}
    manquants = [f for f, b in binome_of_file.items() if b is None]
    if manquants:
        raise ValueError(
            f"{len(manquants)} fichiers sans binôme (Bxx). "
            f"Exemples : {manquants[:3]}"
        )

    binomes = sorted(set(binome_of_file.values()))
    if len(binomes) < n_splits:
        raise ValueError(
            f"{len(binomes)} binômes pour {n_splits} plis : impossible."
        )

    rng = np.random.RandomState(seed)
    ordre = rng.permutation(len(binomes))
    fold_of_binome = {
        binomes[i]: int(ordre[k] % n_splits)
        for k, i in enumerate(ordre)
    }

    payload = {
        'seed': seed,
        'n_splits': n_splits,
        'unite': 'BINOME (Bxx)',
        'fold_of_binome': fold_of_binome,
        'date': '2026-10-28',
        'note': 'Pré-enregistré — ne pas modifier.',
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    h = _hash_fichier(out)
    print(f"folds.json écrit. SHA-256 : {h}")
    print("Copiez ce hash dans Config.FOLDS_SHA256 et dans REGISTRE.md.")
    return payload


def load_folds(path=None, expected_sha256=None):
    """
    Lis le fichier json de plis
    Vérifie que le hash du fichier correspond bien, c'est-à-dire que le fichier n'a pas été reconstruit
    """
    path = Path(path or Config.FOLDS_PATH)
    if expected_sha256 is None:
        expected_sha256 = Config.FOLDS_SHA256
    if expected_sha256:
        h = _hash_fichier(path)
        if h != expected_sha256:
            raise RuntimeError(
                f"folds.json modifié ! Attendu {expected_sha256[:12]}…, "
                f"obtenu {h[:12]}…"
            )
    return json.loads(path.read_text())


def split_outer(folds, k):
    files = folds['fold_of_file']
    test = [f for f, kk in files.items() if kk == k]
    train = [f for f, kk in files.items() if kk != k]
    return train, test


def split_inner(folds, k, seed=None):
    seed = seed if seed is not None else Config.GRAINE
    rng = np.random.RandomState(seed + k)
    folds_train = [j for j in range(folds['n_splits']) if j != k]
    val = int(rng.choice(folds_train))
    inner = [j for j in folds_train if j != val]
    return inner, val