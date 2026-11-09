import json
from pathlib import Path

import pandas as pd

from config import Config
from data.loader import charger_essai, extraire_num_condition
from data.cleaning import nettoyer_essai
from data.contact import contact_AB
from data.ding import ding


def traitement_condition(dossier, num_cond, condition, tronquer_debut=True):
    """
    Charge tous les essais d'un dossier plat dont :
    - le nom contient la tâche courante (Config.TACHE)
    - le numéro de condition est `num_cond`

    Retourne (dataframes, metadatas). Chaque essai est nettoyé, enrichi
    de `contact_AB` et du Ding.
    """
    dossier = Path(dossier)
    dataframes, metadatas = {}, {}

    tache = Config.TACHE
    if not tache:
        raise ValueError("Config.TACHE non défini. Appelez Config.charger_tache(...) d'abord.")

    roles_inverses = num_cond in getattr(Config, 'COND_INVERSES', set())

    for f in sorted(dossier.glob("*.xlsx")):
        nom = f.stem
        if tache not in nom:
            continue
        if extraire_num_condition(nom) != num_cond:
            continue
        try:
            df, meta = charger_essai(
                f,
                roles_inverses=roles_inverses,
                tronquer_debut_essai=tronquer_debut,
            )
            df = nettoyer_essai(df)
            df['contact_AB'] = contact_AB(df)
            df, meta = ding(df, meta)
            meta['condition'] = condition
            dataframes[meta['fichier']] = df
            metadatas[meta['fichier']] = meta
        except Exception as e:
            print(f"❌ {f.name} : {e}")

    print(f"  {num_cond} ({condition}, inversé={roles_inverses}) "
          f": {len(dataframes)} essais")
    return dataframes, metadatas


def cache_dataframes(dataframes, metadatas, cache_dir=None):
    cache_dir = Path(cache_dir or Config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for old in cache_dir.glob("*.parquet"):
        old.unlink()

    for nom, df in dataframes.items():
        nom_safe = nom.replace('/', '_').replace('\\', '_')
        df.to_parquet(cache_dir / f"{nom_safe}.parquet", index=False)

    meta_serializable = {}
    for nom, meta in metadatas.items():
        meta_copy = {}
        for k, v in meta.items():
            if isinstance(v, (list, tuple)):
                meta_copy[k] = list(v)
            elif hasattr(v, 'item'):
                meta_copy[k] = v.item()
            else:
                meta_copy[k] = v
        meta_serializable[nom] = meta_copy

    with open(cache_dir / "metadatas.json", "w", encoding="utf-8") as f:
        json.dump(meta_serializable, f, ensure_ascii=False, indent=2)

    print(f"✅ Cache sauvegardé dans {cache_dir} ({len(dataframes)} fichiers)")


def charger_cache(cache_dir=None):
    cache_dir = Path(cache_dir or Config.CACHE_DIR)
    meta_file = cache_dir / "metadatas.json"

    if not cache_dir.exists() or not meta_file.exists():
        return None, None

    try:
        with open(meta_file, "r", encoding="utf-8") as fh:
            metadatas = json.load(fh)
    except Exception as e:
        print(f"⚠️ Cache métadonnées illisible ({e}).")
        return None, None

    dataframes = {}
    for nom in metadatas.keys():
        nom_safe = nom.replace('/', '_').replace('\\', '_')
        pq_file = cache_dir / f"{nom_safe}.parquet"
        if not pq_file.exists():
            print(f"⚠️ {pq_file.name} manquant.")
            return None, None
        dataframes[nom] = pd.read_parquet(pq_file)

    if not dataframes:
        return None, None

    return dataframes, metadatas


def _nums_condition_pour(condition):
    return [num for num, code in Config.DICT_COND.items() if code == condition]


def construire_charger_cache(force_reload=False, tronquer_debut=True):
    """
    Construit ou recharge le cache pour la tâche courante (Config.TACHE),condition par condition.

    Structure :
        Config.CACHE_DIR / <condition> / *.parquet
        Config.CACHE_DIR / <condition> / metadatas.json

    Retourne un dictionnaire {condition: (dataframes, metadatas)}.
    """
    if not Config.TACHE:
        raise ValueError("Config.TACHE non défini.")
    
    cache_racine = Path(Config.CACHE_DIR)
    resultats = {}
    
    for cond in Config.CONDITIONS:
        cache_cond = cache_racine / cond
        
        # Si force_reload=False ET le cache existe → on le charge
        if not force_reload:
            dfs, metas = charger_cache(cache_cond)
            if dfs is not None:
                resultats[cond] = (dfs, metas)
                print(f"✅ {cond} rechargé depuis le cache ({len(dfs)} essais)")
                continue  # on passe à la condition suivante
        
        # Sinon (force_reload=True, ou cache manquant) → on reconstruit
        print(f"Construction de {Config.TACHE}/{cond}...")
        dfs, metas = {}, {}
        for num_cond in _nums_condition_pour(cond):
            d, m = traitement_condition(Config.XLSX_DIR, num_cond, cond, tronquer_debut=tronquer_debut)
            dfs.update(d)
            metas.update(m)
        if not dfs:
            print(f"⚠️ Condition {cond} : aucun essai chargé.")
            continue
        cache_dataframes(dfs, metas, cache_cond)
        resultats[cond] = (dfs, metas)
    
    if not resultats:
        print("❌ Aucun fichier chargé.")
        return {}
    
    return resultats