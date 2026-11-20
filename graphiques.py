import sys
from pathlib import Path

# --- Ajouter la racine au PYTHONPATH ---
RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np
import io
import tempfile
import os
import importlib

from config import Config
from data.build_cache import construire_charger_cache
from data.contact import detecter_ruptures_contact
from traitement_fichier import enrichir_dataframe

# --- Configuration ---
CACHE_DIR = RACINE / "cache"
CONDITIONS = ['BB', 'BH', 'HB', 'HH', 'GB', 'GH']
POSITIONS = ['tt', 'rt', 'tr', 'rr']
TYPES_EVENTS = ['rupture', 'reprise']
ROLES = ['A', 'B']
TACHES = ['GSV', 'BST', 'COL']
# Autres variables définies dans la Config

st.title("Analyse temporelle des réactions")

# ============================================================
# CHARGEMENT
# ============================================================
@st.cache_data(max_entries=2, ttl=3600, show_spinner="Calcul des moyennes…")
def calculer_moyennes(taches_lst, conditions_par_tache, positions_lst, projections,
                      type_events_tuple, roles_tuple, periode, essai_filtre,
                      evenement_filtre, fenetre_avant_ms, fenetre_apres_ms, pas_direction):
    # ========== 1) CHARGEMENT + ENRICHISSEMENT + EXTRACTION ==========
    n_avant = int(fenetre_avant_ms / 10)
    n_apres = int(fenetre_apres_ms / 10)
    type_events_list = type_events_tuple[0]
    roles_list = roles_tuple[0]

    # Accumulateur : { cle: { 'temps': [], 'acc': [], 'v': [], ... } }
    bruts = {}

    for tache in taches_lst:
        proj_intersujets = (projections[tache] == "Vers l'autre")
        Config.charger_tache(tache)
        cache_tache = Path(Config.CACHE_DIR)
        if not cache_tache.exists():
            construire_charger_cache(force_reload=False, tronquer_debut=True)

        for cond in conditions_par_tache[tache]:
            chemin_cond = Path(Config.CACHE_DIR) / cond
            if not chemin_cond.exists():
                continue

            # Métadonnées (une fois par condition)
            chemin_meta = chemin_cond / "metadatas.json"
            metas = {}
            if chemin_meta.exists() and Config.TACHE == 'GSV':
                with open(chemin_meta, encoding="utf-8") as f:
                    metas = json.load(f)

            for fichier in chemin_cond.glob("*.parquet"):
                essai = fichier.stem
                if essai_filtre != 'Tous' and essai != essai_filtre:
                    continue

                df = pd.read_parquet(fichier)
                if 'contact_AB' not in df.columns:
                    continue

                # Cibles pour GSV
                cible_A = cible_B = None
                if Config.TACHE == 'GSV':
                    meta = metas.get(essai, {})
                    ca = (meta.get('cible_A_x'), meta.get('cible_A_y'))
                    cb = (meta.get('cible_B_x'), meta.get('cible_B_y'))
                    cible_A = ca if ca[0] is not None and ca[1] is not None else None
                    cible_B = cb if cb[0] is not None and cb[1] is not None else None

                # Enrichir
                for pos in positions_lst:
                    df = enrichir_dataframe_graphique(
                        df, pos, Config.TACHE, proj_intersujets,
                        cible_A, cible_B, pas_direction)

                # Détecter les évènements
                idx_ruptures, idx_reprises = detecter_ruptures_contact(
                    df, 'contact_AB', col_temps='temps_relatif_ms')
                tous_les_evenements = (
                    [(idx, 'rupture') for idx in idx_ruptures] +
                    [(idx, 'reprise') for idx in idx_reprises])
                tous_les_evenements.sort()

                # ===== MODE ÉVÈNEMENT =====
                if periode == 'Evenement':
                    for pos in positions_lst:
                        for te in type_events_list:
                            idx_events = idx_ruptures if te == 'rupture' else idx_reprises
                            if evenement_filtre != 'Tous':
                                if evenement_filtre < len(idx_events):
                                    idx_events = [idx_events[evenement_filtre]]
                                else:
                                    idx_events = []

                            for r in roles_list:
                                col_acc = f'acceleration_{r}_{pos}'
                                col_v = f'vitesse_signee_{r}_{pos}'
                                col_proj = f'projection_ortho_{r}_{pos}'
                                col_dif = f'dif_proj_{r}'
                                col_x = f'{r}-{pos[0]}X'
                                col_y = f'{r}-{pos[0]}Y'
                                cols_requises = (col_acc, col_v, col_proj, col_dif, col_x, col_y)
                                if any(c not in df.columns for c in cols_requises):
                                    continue

                                for idx in idx_events:
                                    if idx - n_avant < 20:
                                        continue
                                    idx_debut = max(0, idx - n_avant)
                                    idx_fin = min(len(df), idx + n_apres)
                                    fenetre = df.iloc[idx_debut:idx_fin]
                                    if len(fenetre) < n_avant + 10:
                                        continue

                                    # Deux prochains évènements
                                    prochains_idx = sorted(
                                        [i for i, _ in tous_les_evenements if i > idx])[:2]
                                    deux_prochains = [(i - idx) * 10 for i in prochains_idx]
                                    while len(deux_prochains) < 2:
                                        deux_prochains.append(None)

                                    # Groupe
                                    groupes = []
                                    if not taches_tuple[1]: groupes.append(tache)
                                    if not conditions_tuple[1]: groupes.append(cond)
                                    if not positions_tuple[1]: groupes.append(pos)
                                    if not type_events_tuple[1]: groupes.append(te)
                                    if not roles_tuple[1]: groupes.append(r)
                                    cle = '_'.join(groupes) if groupes else ''

                                    if cle not in bruts:
                                        bruts[cle] = {'temps': [], 'acc': [], 'v': [],
                                                      'proj': [], 'dif': [], 'x': [], 'y': [],
                                                      'deux_prochains': []}
                                    n_f = len(fenetre)
                                    t_rel = (np.arange(n_f) - (idx - idx_debut)) * 10
                                    bruts[cle]['temps'].append(t_rel)
                                    bruts[cle]['acc'].append(fenetre[col_acc].values)
                                    bruts[cle]['v'].append(fenetre[col_v].values)
                                    bruts[cle]['proj'].append(fenetre[col_proj].values)
                                    bruts[cle]['dif'].append(fenetre[col_dif].values)
                                    bruts[cle]['x'].append(fenetre[col_x].values)
                                    bruts[cle]['y'].append(fenetre[col_y].values)
                                    bruts[cle]['deux_prochains'].append(deux_prochains)

                # ===== MODE ESSAI =====
                else:
                    # Convertir en positions
                    idx_rup = np.asarray(df.index.get_indexer(
                        np.asarray(idx_ruptures, dtype=df.index.dtype)), dtype=int)
                    idx_rep = np.asarray(df.index.get_indexer(
                        np.asarray(idx_reprises, dtype=df.index.dtype)), dtype=int)
                    idx_rup = idx_rup[idx_rup >= 0]
                    idx_rep = idx_rep[idx_rep >= 0]
                    temps = df['temps_relatif_ms'].to_numpy()

                    for pos in positions_lst:
                        for r in roles_list:
                            col_acc = f'acceleration_{r}_{pos}'
                            col_v = f'vitesse_signee_{r}_{pos}'
                            col_proj = f'projection_ortho_{r}_{pos}'
                            col_dif = f'dif_proj_{r}'
                            col_x = f'{r}-{pos[0]}X'
                            col_y = f'{r}-{pos[0]}Y'
                            if col_v not in df.columns:
                                continue

                            groupes = []
                            if not taches_tuple[1]: groupes.append(tache)
                            if not conditions_tuple[1]: groupes.append(cond)
                            if not positions_tuple[1]: groupes.append(pos)
                            if not roles_tuple[1]: groupes.append(r)
                            cle = '_'.join(groupes) if groupes else ''

                            if cle not in bruts:
                                bruts[cle] = {'temps': [], 'acc': [], 'v': [],
                                              'proj': [], 'dif': [], 'x': [], 'y': [],
                                              'ruptures': [], 'reprises': []}
                            bruts[cle]['temps'].append(temps)
                            bruts[cle]['acc'].append(df[col_acc].to_numpy() if col_acc in df.columns else np.full(len(df), np.nan))
                            bruts[cle]['v'].append(df[col_v].to_numpy())
                            bruts[cle]['proj'].append(df[col_proj].to_numpy() if col_proj in df.columns else np.full(len(df), np.nan))
                            bruts[cle]['dif'].append(df[col_dif].to_numpy() if col_dif in df.columns else np.full(len(df), np.nan))
                            bruts[cle]['x'].append(df[col_x].to_numpy() if col_x in df.columns else np.full(len(df), np.nan))
                            bruts[cle]['y'].append(df[col_y].to_numpy() if col_y in df.columns else np.full(len(df), np.nan))
                            bruts[cle]['ruptures'].append(temps[idx_rup])
                            bruts[cle]['reprises'].append(temps[idx_rep])

    # ========== 2) CALCUL DES MOYENNES ET SEM ==========
    def _moy_sem(liste_tableaux):
        if not liste_tableaux:
            return None, None
        n_max = max(len(a) for a in liste_tableaux)
        mat = np.full((len(liste_tableaux), n_max), np.nan)
        for i, a in enumerate(liste_tableaux):
            mat[i, :len(a)] = a
        moy = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        n_obs = np.sum(~np.isnan(mat), axis=0)
        sem = std / np.sqrt(np.where(n_obs > 0, n_obs, 1))
        return moy, sem

    moyennes = {}
    for cle, d in bruts.items():
        temps_moy, _ = _moy_sem(d['temps'])
        acc_moy, acc_sem = _moy_sem(d['acc'])
        v_moy, v_sem = _moy_sem(d['v'])
        proj_moy, proj_sem = _moy_sem(d['proj'])
        dif_moy, dif_sem = _moy_sem(d['dif'])
        x_moy, x_sem = _moy_sem(d['x'])
        y_moy, y_sem = _moy_sem(d['y'])

        moyennes[cle] = {
            'temps': temps_moy,
            'acc_moy': acc_moy, 'acc_sem': acc_sem,
            'v_moy': v_moy, 'v_sem': v_sem,
            'proj_moy': proj_moy, 'proj_sem': proj_sem,
            'dif_moy': dif_moy, 'dif_sem': dif_sem,
            'x_moy': x_moy, 'x_sem': x_sem,
            'y_moy': y_moy, 'y_sem': y_sem,
            'n': len(d['acc']),
            'deux_prochains': d.get('deux_prochains', []),
            'ruptures': d.get('ruptures', []),
            'reprises': d.get('reprises', []),
        }

    # Libérer explicitement la mémoire des bruts avant de quitter
    del bruts
    import gc; gc.collect()

    return moyennes

# LISTE DES ESSAIS
def lister_essais(conditions_par_tache: dict):
    """Essais des couples (tâche, condition) demandés : {tâche: [conditions]}."""
    essais = []
    for tache, conds in conditions_par_tache.items():
        for cond in conds:
            for fichier in Path(f"cache/{tache}/{cond}").glob("*.parquet"):
                essais.append({
                    'tache': tache,
                    'condition': cond,
                    'essai': fichier.stem
                })
    return pd.DataFrame(essais)


@st.cache_data(max_entries=2, ttl=3600)
def lister_evenements(essai, type_event, tache, cond):
    if essai == 'Tous':
        return ['Tous']
    
    df = pd.read_parquet(f"cache/{tache}/{cond}/{essai}.parquet")
    
    if 'contact_AB' not in df.columns:
        return ['Tous']
    
    if 'contact_AB' not in df.columns:
        return ['Tous']
    
    idx_ruptures, idx_reprises = detecter_ruptures_contact(df, 'contact_AB', col_temps='temps_relatif_ms', duree_min_ms= 500)
    
    if type_event == 'rupture':
        idx_events = idx_ruptures
    elif type_event == 'reprise':
        idx_events = idx_reprises
    else:
        idx_events = list(idx_ruptures) + list(idx_reprises)
    
    return ['Tous'] + list(range(len(idx_events)))


def uniques(seq):
    """Retourne une liste sans doublons, en gardant l'ordre d'apparition."""
    vus = []
    for x in seq:
        if x not in vus:
            vus.append(x)
    return vus


def comparer_courbes(donnees, cles, fenetre_avant):
    resultats = []
    for i, cle1 in enumerate(cles):
        for cle2 in cles[i+1:]:
            data1 = donnees[cle1]
            data2 = donnees[cle2]
            
            acc1 = np.concatenate(data1['acc']) if data1['acc'] else np.array([])
            acc2 = np.concatenate(data2['acc']) if data2['acc'] else np.array([])
            v1 = np.concatenate(data1['v']) if data1['v'] else np.array([])
            v2 = np.concatenate(data2['v']) if data2['v'] else np.array([])
            
            n = min(len(acc1), len(acc2))
            if n == 0:
                continue
            
            acc1, acc2 = acc1[:n], acc2[:n]
            v1, v2 = v1[:n], v2[:n]
            
            resultats.append({
                'courbe_1': cle1,
                'courbe_2': cle2,
                'ratio_acc': np.nanmax(acc2) / (np.nanmax(acc1) + 1e-9),
                'ratio_v': np.nanmax(v2) / (np.nanmax(v1) + 1e-9),
                'delta_acc': np.nanmean(acc2) - np.nanmean(acc1),
                'delta_v': np.nanmean(v2) - np.nanmean(v1),
                'corr_acc': np.corrcoef(acc1, acc2)[0, 1] if n > 2 else np.nan,
                'corr_v': np.corrcoef(v1, v2)[0, 1] if n > 2 else np.nan,
            })
    return pd.DataFrame(resultats)
    

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Affichage")
mode_affichage = st.sidebar.radio("Mode d'affichage :", ['Courbe', 'Nuage de points'], index=0)

if mode_affichage == 'Nuage de points':
    extremes = st.checkbox(label='Afficher les extrêmes', value=True)
    lisse = False
else:
    extremes = True
    lisse = st.checkbox(label="Lisser la courbe", value = True)

periode = st.sidebar.radio("Période d'affichage :", ['Evenement', 'Essai'], index=0)

prochains = False
if periode=='Evenement':
    prochains = st.checkbox(label='Afficher les deux prochains évènements', value=False)

st.sidebar.header("Paramètres")

projections = {}
st.sidebar.subheader("Tâche (défaut=Toutes)")
gsv = st.sidebar.checkbox("GSV")
projections["GSV"] = st.sidebar.radio("Choix projection des vitesses :", ["Vers l'autre", "Vers la cible"], index=0)
bst = st.sidebar.checkbox("BST")
projections["BST"] = st.sidebar.radio("Choix projection des vitesses :", ["Vers l'autre", "Vers la bestiole"], index=0)
col = st.sidebar.checkbox("COL")
projections["COL"] = st.sidebar.radio("Choix projection des vitesses :", ["Vers l'autre", "Vers la colline"], index=0)

taches_list = []
if gsv: taches_list.append('GSV')
if bst: taches_list.append('BST')
if col: taches_list.append('COL')
if taches_list == []:
    taches_list = TACHES
    taches_tuple = (taches_list, True)
else:
    taches_tuple = (taches_list, False)

rupture, reprise = False, False
if periode == "Evenement":
    st.sidebar.subheader("Type d'évènement (défaut=Tous)")
    rupture = st.sidebar.checkbox("Ruptures", False)
    reprise = st.sidebar.checkbox("Reprises", False)

st.sidebar.subheader("Positions d'analyse (défaut=Toutes)")
tt = st.sidebar.checkbox('Curseur-Curseur', False)
rr = st.sidebar.checkbox('Souris-Souris', False)
rt = st.sidebar.checkbox('Souris guide-Curseur suiveur', False)
tr = st.sidebar.checkbox('Curseur guide-Souris suiveur', False)

if taches_list == ['GSV']:
    st.sidebar.subheader("Mode de jeu")
    gs = st.sidebar.checkbox('Guide-Suiveur')
    gg = st.sidebar.checkbox('Guide-Guide')
    if gs and not gg:
        st.sidebar.subheader("Sujets (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide', False)
        role2 = st.sidebar.checkbox('Suiveur', False)
    elif gg and not gs:
        st.sidebar.subheader("Sujets (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide A', False)
        role2 = st.sidebar.checkbox('Guide B', False)
    else:
        st.sidebar.subheader("Sujets (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide/Guide A', False)
        role2 = st.sidebar.checkbox('Suiveur/Guide B', False)

elif len(taches_list) == 1:
    st.sidebar.subheader("Sujets (défaut=Tous)")
    role1 = st.sidebar.checkbox("Sujet A", False)
    role2 = st.sidebar.checkbox("Sujet B", False)


st.sidebar.subheader("Conditions (défaut=Toutes)")
conds_dispo_par_tache = {}
conds_choisies_par_tache = {}
for tache in taches_list:
    st.sidebar.markdown(f"**{tache}**")
    # Charger la config de la tâche pour connaître ses conditions
    try:
        module = importlib.import_module(f"taches.{tache}.config_{tache}")
        classe = getattr(module, f"Config_{tache}")
        conds_tache = list(classe.CONDITIONS)
    except (ModuleNotFoundError, AttributeError):
        st.sidebar.warning(f"Config {tache} introuvable")
        conds_tache = []

    conds_dispo_par_tache[tache] = conds_tache
    conds_choisies_par_tache[tache] = [
        c for c in conds_tache
        if st.sidebar.checkbox(c, key=f"cond_{tache}_{c}")
    ]

# Règle de sélection :
#  - aucune condition cochée           -> toutes les conditions de toutes les tâches ;
#  - tâche avec des conditions cochées -> seulement celles-là ;
#  - tâche sans condition cochée       -> toutes ses conditions si la tâche est cochée
#    explicitement, sinon la tâche est exclue.
aucune_condition = not any(conds_choisies_par_tache.values())
conditions_par_tache = {}
for tache in taches_list:
    choisies = conds_choisies_par_tache[tache]
    if choisies:
        conditions_par_tache[tache] = choisies
    elif aucune_condition or not taches_tuple[1]:
        conditions_par_tache[tache] = conds_dispo_par_tache[tache]
    else:
        conditions_par_tache[tache] = []

toutes_conditions = uniques([c for t in taches_list for c in conds_dispo_par_tache[t]])
if aucune_condition:
    conditions_list = toutes_conditions
else:
    conditions_list = uniques([c for t in taches_list for c in conds_choisies_par_tache[t]])
conditions_tuple = (conditions_list, aucune_condition)


# ============================================================
# FILTRAGE
# ============================================================
positions_list = []
if tt: positions_list.append('tt')
if rr: positions_list.append('rr')
if rt: positions_list.append('rt')
if tr: positions_list.append('tr')
if positions_list == []:
    positions_list = Config.POSITIONS
    positions_tuple = (positions_list, True)
else:
    positions_tuple = (positions_list, False)

if rupture:
    if reprise:
        type_events_tuple = (TYPES_EVENTS, False)
    else:
        type_events_tuple = (['rupture'], False)
elif reprise:
    type_events_tuple = (['reprise'], False)
else:
    type_events_tuple = (TYPES_EVENTS, True)

if taches_list == ['GSV'] and ((gs and not gg) or (gg and not gs)) or len(taches_list) == 1:
    if role1:
        if role2:
            roles_tuple = (ROLES, False)
        else:
            roles_tuple = (['A'], False)
    elif role2:
        roles_tuple = (['B'], False)
    else : roles_tuple = (ROLES, True)
else:
    roles_tuple = (ROLES, True)

# Essai
df_essais = lister_essais(conditions_par_tache)
if 'essai' in df_essais.columns:
    essais_disponibles = ['Tous'] + sorted(df_essais['essai'].unique().tolist())
else:
    essais_disponibles = ['Tous']

essai = st.sidebar.selectbox("Essai :", essais_disponibles)

# Evenement
evenement = 'Tous'
if periode == 'Evenement' and essai != 'Tous' and (rupture or reprise):
    type_event_pour_event = 'rupture' if rupture else 'reprise'
    ligne = df_essais[df_essais['essai'] == essai]
    condition_essai = ligne['condition'].iloc[0] if not ligne.empty else None
    tache_essai = ligne['tache'].iloc[0] if not ligne.empty else None
    evenements_disponibles = lister_evenements(essai, type_event_pour_event, tache_essai, condition_essai)
    evenement = st.sidebar.selectbox("Événement :", evenements_disponibles)

# Echantillonage
if mode_affichage == "Nuage de points":
    echantillon = st.sidebar.slider("Echantillonage (ms)", 10, 500, 100, 10, help="Temps entre les différents enregistrements")

# Fenêtre
if periode == "Evenement":
    # Fenêtre
    fenetre_avant = st.sidebar.slider("Fenêtre avant (ms)", 100, 1000, 500, 100, help="Fenêtre de temps affichée avant l'évènement sur le graphique")
    fenetre_apres = st.sidebar.slider("Fenêtre après (ms)", 500, 3000, 1500, 100, help="Fenêtre de temps affichée après l'évènement sur le graphique")
else:
    fenetre_avant = 0
    fenetre_apres = 0

# Pas direction
pas_direction = st.sidebar.slider("Fenêtre glissante d'enregistrement de la vitesse (ms)", 40, 500, 200, 20, help="Temps entre les enregistrements des positions d'un individu pour le calcul de la vitesse") //10

taches_effectives = [t for t in taches_list if conditions_par_tache[t]]

donnees_preparees = calculer_moyennes(
    sorted(taches_list),
    conditions_par_tache,
    positions_tuple,
    projections,
    type_events_tuple,
    roles_tuple,
    periode,
    essai_filtre,
    evenement_filtre,
    fenetre_avant_ms,
    fenetre_apres_ms,
    pas_direction
)
# ============================================================
# EXTRACTION DES FENÊTRES
# ============================================================

@st.cache_data(max_entries=2, ttl=3600)
def extraire_fenetres(_donnees_preparees, taches_tuple, conditions_tuple, positions_tuple, type_events_tuple,
                       roles_tuple, essai_filtre, evenement_filtre,
                       fenetre_avant_ms, fenetre_apres_ms, pas_direction):

    compteurs = {
        'essais': 0, 'sans_contact_AB': 0,
        'essais_avec_ruptures': 0, 'essais_avec_reprises': 0,
        'rejets_manque_colonnes': 0, 'rejets_idx_trop_tot': 0,
        'rejets_fenetre_courte': 0, 'fenetres_gardees': 0,
        'colonnes_manquantes_A': set(), 'colonnes_manquantes_B': set(),
    }
    positions_list = positions_tuple[0]
    type_events_list = type_events_tuple[0]
    roles_list = roles_tuple[0]
    
    n_avant = int(fenetre_avant_ms / 10)
    n_apres = int(fenetre_apres_ms / 10)
    
    donnees = {}
    
    for (tache, cond, nom), df in _donnees_preparees.items():
        compteurs['essais'] += 1

        if essai_filtre != 'Tous' and nom != essai_filtre:
            continue

        if 'contact_AB' not in df.columns:
            compteurs['sans_contact_AB'] += 1
            continue

        idx_ruptures, idx_reprises = detecter_ruptures_contact(df, 'contact_AB', col_temps='temps_relatif_ms')
        if len(idx_ruptures): compteurs['essais_avec_ruptures'] += 1
        if len(idx_reprises): compteurs['essais_avec_reprises'] += 1
        
        idx_ruptures, idx_reprises = detecter_ruptures_contact(df, 'contact_AB', col_temps='temps_relatif_ms')
        tous_les_evenements = ([(idx, 'rupture') for idx in idx_ruptures] + [(idx, 'reprise') for idx in idx_reprises])
        tous_les_evenements.sort()
                
        for pos in positions_list:
            for te in type_events_list:
                idx_events = idx_ruptures if te == 'rupture' else idx_reprises
                        
                if evenement_filtre != 'Tous':
                    if evenement_filtre < len(idx_events):
                        idx_events = [idx_events[evenement_filtre]]
                    else:
                        idx_events = []
                    
                for r in roles_list:
                    if r == 'A':
                        col_acc = f'acc_A_vers_B_{pos}'
                        col_v   = f'vitesse_A_vers_B_{pos}'
                        col_x   = f'A-{pos[0]}X'
                        col_y   = f'A-{pos[0]}Y'
                    else:
                        col_acc = f'acc_B_vers_A_{pos}'
                        col_v   = f'vitesse_B_vers_A_{pos}'
                        col_x   = f'B-{pos[1]}X'
                        col_y   = f'B-{pos[1]}Y'

                    if (col_acc not in df.columns or col_v not in df.columns
                            or col_x not in df.columns or col_y not in df.columns):
                        compteurs['rejets_manque_colonnes'] += 1
                        if r == 'A':
                            for c in (col_acc, col_v, col_x, col_y):
                                if c not in df.columns:
                                    compteurs['colonnes_manquantes_A'].add(c)
                        else:
                            for c in (col_acc, col_v, col_x, col_y):
                                if c not in df.columns:
                                    compteurs['colonnes_manquantes_B'].add(c)
                        continue

                    if col_acc not in df.columns or col_v not in df.columns or col_x not in df.columns:
                        continue
                            
                    for idx in idx_events:
                        if idx - n_avant < 20:
                            compteurs['rejets_idx_trop_tot'] += 1
                            continue
                                
                        idx_debut = max(0, idx - n_avant)
                        idx_fin = min(len(df), idx + n_apres)
                        fenetre = df.iloc[idx_debut:idx_fin]
                        
                        if len(fenetre) < n_avant + 10:
                            compteurs['rejets_fenetre_courte'] += 1
                            continue
                            
                        # Autres événements
                        autres_evenements = []
                        for idx_autre, type_autre in tous_les_evenements:
                            if idx_autre == idx:
                                continue
                            if idx_debut <= idx_autre < idx_fin:
                                temps_rel = (idx_autre - idx) * 10
                                autres_evenements.append({
                                    'temps_relatif_ms': temps_rel,
                                    'type': type_autre,
                                })
                                
                        # Les deux prochains événements (peu importe le type)
                        prochains = [idx_autre for idx_autre, _ in tous_les_evenements if idx_autre > idx]
                        prochains = sorted(prochains)[:2]
                                
                        deux_prochains = [(idx_autre - idx) * 10 for idx_autre in prochains]
                        while len(deux_prochains) < 2:
                            deux_prochains.append(None)
                                
                        # Groupe
                        groupes = []
                        if not taches_tuple[1]:
                            groupes.append(tache)
                        if not conditions_tuple[1]:
                            groupes.append(cond)
                        if not positions_tuple[1]:
                            groupes.append(pos)
                        if not type_events_tuple[1]:
                            groupes.append(te)
                        if not roles_tuple[1]:
                            groupes.append(r)
                                
                        cle = '_'.join(groupes) if groupes else ''
                                
                        if cle not in donnees:
                            donnees[cle] = {
                                'acc': [], 'v': [], 'x': [], 'y': [], 'temps': [],
                                'autres_evenements': [],
                                'deux_prochains': [],
                            }

                        # Temps relatif à l'évènement (ms), cohérent avec le mode Essai
                        n_f = len(fenetre)
                        t_rel = (np.arange(n_f) - (idx - idx_debut)) * 10

                        donnees[cle]['temps'].append(t_rel)
                        donnees[cle]['acc'].append(fenetre[col_acc].values)
                        donnees[cle]['v'].append(fenetre[col_v].values)
                        donnees[cle]['x'].append(fenetre[col_x].values)
                        donnees[cle]['y'].append(fenetre[col_y].values)
                        donnees[cle]['autres_evenements'].append(autres_evenements)
                        donnees[cle]['deux_prochains'].append(deux_prochains)
    return donnees


@st.cache_data(max_entries=2, ttl=3600)
def extraire_essai(_donnees_preparees, taches_tuple, conditions_tuple, positions_tuple, roles_tuple, essai_filtre, pas_direction):
    """Retourne, pour chaque groupe, les séries complètes + les évènements."""
    positions_list = positions_tuple[0]
    roles_list = roles_tuple[0]

    donnees = {}

    for (tache, cond, nom), df in _donnees_preparees.items():
        if essai_filtre != 'Tous' and nom != essai_filtre:
            continue
        if 'contact_AB' not in df.columns:
            continue

        idx_ruptures, idx_reprises = detecter_ruptures_contact(df, 'contact_AB', col_temps='temps_relatif_ms')

        # Convertir les labels d'index en positions (robuste à tout type d'index)
        idx_ruptures = np.asarray(
            df.index.get_indexer(np.asarray(idx_ruptures, dtype=df.index.dtype)),
            dtype=int)
        idx_reprises = np.asarray(
            df.index.get_indexer(np.asarray(idx_reprises, dtype=df.index.dtype)),
            dtype=int)

        # Nettoyer les -1 (labels absents, ne devrait pas arriver mais ceinture+bretelles)
        idx_ruptures = idx_ruptures[idx_ruptures >= 0]
        idx_reprises = idx_reprises[idx_reprises >= 0]

        temps = df['temps_relatif_ms'].to_numpy()

        for pos in positions_list:
            for r in roles_list:
                # Colonnes selon le rôle et la position
                if r == 'A':
                    col_acc = f'acc_A_vers_B_{pos}'
                    col_v   = f'vitesse_A_vers_B_{pos}'
                    col_x   = f'A-{pos[0]}X'
                    col_y   = f'A-{pos[0]}Y'
                else:
                    col_acc = f'acc_B_vers_A_{pos}'
                    col_v   = f'vitesse_B_vers_A_{pos}'
                    col_x   = f'B-{pos[1]}X'
                    col_y   = f'B-{pos[1]}Y'

                if col_v not in df.columns:
                    continue

                groupes = []
                if not taches_tuple[1]:
                    groupes.append(tache)
                if not conditions_tuple[1]:
                    groupes.append(cond)
                if not positions_tuple[1]:
                    groupes.append(pos)
                if not roles_tuple[1]:
                    groupes.append(r)

                cle = '_'.join(groupes) if groupes else ''

                if cle not in donnees:
                    donnees[cle] = {
                        'temps': [], 'acc': [], 'v': [], 'x': [], 'y': [],
                        'ruptures': [], 'reprises': [],
                    }

                donnees[cle]['temps'].append(temps)
                donnees[cle]['acc'].append(df[col_acc].to_numpy() if col_acc in df.columns else np.full(len(df), np.nan))
                donnees[cle]['v'].append(df[col_v].to_numpy())
                donnees[cle]['x'].append(df[col_x].to_numpy() if col_x in df.columns else np.full(len(df), np.nan))
                donnees[cle]['y'].append(df[col_y].to_numpy() if col_y in df.columns else np.full(len(df), np.nan))
                donnees[cle]['ruptures'].append(temps[idx_ruptures])
                donnees[cle]['reprises'].append(temps[idx_reprises])

    return donnees

# --- Dictionnaire de traduction pour l'affichage ---
LABELS_LISIBLES = {
    'tt': 'Curseur-Curseur',
    'rt': 'Souris guide-Curseur suiveur',
    'tr': 'Curseur guide-Souris suiveur',
    'rr': 'Souris-Souris',
    'A': 'A',
    'B': 'B',
    'rupture': 'Ruptures',
    'reprise': 'Reprises',
    'BB': 'BB', 'BH': 'BH', 'HB': 'HB', 'HH': 'HH',
}
if taches_list == ['GSV']:
    if gg and not gs:
        LABELS_LISIBLES['A'] = 'Guide A'
        LABELS_LISIBLES['B'] = 'Guide B'
    elif gs and not gg :
        LABELS_LISIBLES['A'] = 'Guide'
        LABELS_LISIBLES['B'] = 'Suiveur'
    else :
        LABELS_LISIBLES['A'] = 'Guide/Guide A'
        LABELS_LISIBLES['B'] = 'Suiveur/Guide B'

def traduire(cle):
    if not cle:
        return 'Toutes les données'
    parties = cle.split('_')
    return ' | '.join(LABELS_LISIBLES.get(p, p) for p in parties)

def lisser(y, largeur=4):
    return pd.Series(y).rolling(largeur, center=True, min_periods=1).mean().to_numpy()


# STYLE DES COURBES
# Code sémantique :
#   - individu : A = teinte chaude, B = teinte froide, neutre (gris) si A et B sont fusionnés
#   - support  : curseur = trait plein, souris = pointillé (marqueur 'x' en nuage de points)
#   - grandeur : x et y = sombre, vitesse = intermédiaire, accélération = clair
#                (chaque grandeur est sur son propre graphique : la nuance ne compte
#                 donc pas pour distinguer deux courbes entre elles)
# Il est utilisé si chaque courbe a une identité (individu, support) différente de celle
# des autres ; sinon on garde la palette tab10/tab20 d'origine (même couleur sur les 4 graphiques).
# Palettes par individu : teintes en degrés (0..360)
PALETTE_TEINTES = {
    'A': [210, 230, 250],           # bleu, violet (froid)
    'B': [0, 20, 40],              # orange, rouge (chaud)
    None: [0],                 # gris (non utilisé, sat=0)
}
LUM = {'x': 0.30, 'y': 0.30, 'dif': 0.45, 'proj': 0.45, 'v': 0.60, 'acc': 0.75}  # sombre / intermédiaire / clair
SAT = 0.75
STYLE_SUPPORT = {'curseur': ('-', 'o'), 'souris': ('--', 'x'), None: ('-', 'o')}

# ============================================================
# AFFICHAGE
# ============================================================

if not donnees:
    st.warning("Aucune donnée")
else:
    cles_triees = sorted(donnees.keys())
    
    # Palette (inchangé)
    if len(cles_triees) <= 10:
        palette = [plt.cm.tab10(i) for i in range(len(cles_triees))]
    elif len(cles_triees) <= 20:
        palette = [plt.cm.tab20(i) for i in range(len(cles_triees))]
    else:
        palette = [plt.cm.turbo(x) for x in np.linspace(0, 1, len(cles_triees))]
    couleurs = {cle: palette[i] for i, cle in enumerate(cles_triees)}
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(14, 26), sharex=True)
    
    for cle in cles_triees:
        d = donnees[cle]
        couleur = couleurs[cle]
        label = traduire(cle)
        n = d['n']
        
        # Récupérer les séries (déjà moyennées)
        temps = d['temps']
        if lisse:
            for k in ['acc_moy', 'v_moy', 'proj_moy', 'dif_moy', 'x_moy', 'y_moy',
                      'acc_sem', 'v_sem', 'proj_sem', 'dif_sem', 'x_sem', 'y_sem']:
                d[k] = lisser(d[k])
        
        ls = '--' if 'Souris-Souris' in label else '-'
        
        ax1.plot(temps, d['acc_moy'], color=couleur, linewidth=2, linestyle=ls, label=f'{label} (n={n})')
        ax2.plot(temps, d['v_moy'], color=couleur, linewidth=2, linestyle=ls)
        ax3.plot(temps, d['proj_moy'], color=couleur, linewidth=2, linestyle=ls)
        ax4.plot(temps, d['dif_moy'], color=couleur, linewidth=2, linestyle=ls)
        ax5.plot(temps, d['x_moy'], color=couleur, linewidth=2, linestyle=ls)
        ax6.plot(temps, d['y_moy'], color=couleur, linewidth=2, linestyle=ls)
        
        ax1.fill_between(temps, d['acc_moy']-d['acc_sem'], d['acc_moy']+d['acc_sem'], color=couleur, alpha=0.15)
        ax2.fill_between(temps, d['v_moy']-d['v_sem'], d['v_moy']+d['v_sem'], color=couleur, alpha=0.15)
        ax3.fill_between(temps, d['proj_moy']-d['proj_sem'], d['proj_moy']+d['proj_sem'], color=couleur, alpha=0.15)
        ax4.fill_between(temps, d['dif_moy']-d['dif_sem'], d['dif_moy']+d['dif_sem'], color=couleur, alpha=0.15)
        ax5.fill_between(temps, d['x_moy']-d['x_sem'], d['x_moy']+d['x_sem'], color=couleur, alpha=0.15)
        ax6.fill_between(temps, d['y_moy']-d['y_sem'], d['y_moy']+d['y_sem'], color=couleur, alpha=0.15)
        
        # Prochains évènements (mode Évènement)
        if periode == 'Evenement' and d.get('deux_prochains'):
            t1 = [t[0] for t in d['deux_prochains'] if t[0] is not None]
            t2 = [t[1] for t in d['deux_prochains'] if t[1] is not None]
            if t1:
                m1 = np.mean(t1)
                ax1.axvline(m1, color=couleur, linestyle='--', alpha=0.6, linewidth=1.5)
                ax2.axvline(m1, color=couleur, linestyle='--', alpha=0.6, linewidth=1.5)
            if t2:
                m2 = np.mean(t2)
                ax1.axvline(m2, color=couleur, linestyle=':', alpha=0.6, linewidth=1.5)
                ax2.axvline(m2, color=couleur, linestyle=':', alpha=0.6, linewidth=1.5)

    # --- Lignes de référence ---
    if periode == 'Evenement':
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvline(0, color='red', linestyle='--', alpha=1)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax3.axhline(Config.ESPACE_X/2, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax4.axhline(Config.ESPACE_Y/2, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

    if periode == 'Essai':
        # --- BARRES VERTICALES DES RUPTURES/REPRISES (mode Essai) ---
        ruptures = data.get('ruptures', [[]])[0] if data.get('ruptures') else []
        reprises = data.get('reprises', [[]])[0] if data.get('reprises') else []
        for t in ruptures:
            for ax in (ax1, ax2, ax3, ax4):
                ax.axvline(t, color='red', linestyle='--', alpha=0.5, linewidth=1)
        for t in reprises:
            for ax in (ax1, ax2, ax3, ax4):
                ax.axvline(t, color='green', linestyle='-', alpha=0.5, linewidth=1)
    
    # --- Labels ---
    ax1.set_ylabel('Accélération (px/ms²)', fontsize=12)
    ax1.set_title('Accélération', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=12)
    
    ax2.set_ylabel('Vitesse signée (px/ms)', fontsize=12)
    ax2.set_xlabel('Temps relatif (ms)', fontsize=12)
    ax2.set_title('Vitesse signée', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=12)

    ax3.set_ylabel('Position x (px)', fontsize=12)
    ax3.set_title('Position x', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=12)

    ax4.set_ylabel('Position y (px)', fontsize=12)
    ax4.set_xlabel('Temps relatif (ms)' if periode == 'Evenement' else 'Temps (ms)', fontsize=12)
    ax4.set_title('Position y', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=12)

    # --- Ajustement des limites Y ---
    def ajuster_limites(ax, centre, marge=0.05):
        ymin, ymax = ax.get_ylim()
        demi = max(centre - ymin, ymax - centre)
        if demi == 0:
            demi = 1
        demi *= (1 + marge)
        ax.set_ylim(centre - demi, centre + demi)

    ajuster_limites(ax1, 0)
    ajuster_limites(ax2, 0)
    ajuster_limites(ax3, Config.ESPACE_X/2)
    ajuster_limites(ax4, Config.ESPACE_Y/2)
    
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Bouton pour enregistrer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    st.download_button(
        label="📂 Télécharger le graphique", data=buf, mime="image/png")
