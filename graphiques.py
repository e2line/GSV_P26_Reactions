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
print("prepa données")
@st.cache_data(show_spinner="Préparation des données…")
def preparer_donnees(taches_lst, conditions_par_tache, positions_lst, pas_direction):
    """
    Vérifie que le cache existe pour chaque tâche sélectionnée, le construit
    si nécessaire, puis enrichit chaque essai avec les colonnes calculées
    (vitesse signée, accélération, etc.) pour les positions demandées.

    Retourne un dict :
        { (tache, condition, essai) : df_enrichi }
    """
    print("exécuté")
    donnees = {}
    for tache in taches_lst:
        Config.charger_tache(tache)
        cache_tache = Path(CACHE_DIR)
        if not cache_tache.exists():
            # Construire le cache pour cette tâche
            construire_charger_cache(force_reload=False, tronquer_debut=True)

        # Parcourir les conditions demandées
        for cond in conditions_par_tache[tache]:
            chemin_cond = Path(CACHE_DIR) / cond
            if not chemin_cond.exists():
                continue   # condition absente pour cette tâche, on saute

            for fichier in chemin_cond.glob("*.parquet"):
                essai = fichier.stem
                df = pd.read_parquet(fichier)
                for pos in positions_lst:
                    df = enrichir_dataframe(df, pos, pas_direction)
                donnees[(tache, cond, essai)] = df

    return donnees

# LISTE DES ESSAIS
@st.cache_data
def lister_essais(taches_lst: list[str], cond_lst: list[str]):
    essais = []
    for tache in taches_lst:
        for cond in cond_lst:
            for fichier in Path(f"{CACHE_DIR}/{tache}/{cond}").glob("*.parquet"):
                essais.append({
                    'tache': tache,
                    'condition': cond,
                    'essai': fichier.stem
                })
    return pd.DataFrame(essais)


@st.cache_data
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


def construire_gif_reaction(fenetres_v, fenetre_avant_ms, pas_direction_ms, titre="", fps=100, rayon_corps_percu=20.0, rayon_corps_percevant=50.0, taille_case=20.0, n_cases=4, distance_initiale=60.0):
    """
    Construit un GIF (BytesIO) modélisant la réaction radiale d'un individu
    à un objet fixe, à partir des vitesses signées (px/ms).

    Paramètres
    ----------
    fenetres_v : list[np.ndarray]
        Liste des fenêtres de vitesse signée (px/ms), une par essai.
    fenetre_avant_ms : int
        Durée de la fenêtre avant l'évènement (ms).
    pas_direction_ms : int
        Pas d'échantillonnage temporel (ms), typiquement 10.
    titre : str
        Titre affiché sur la figure.
    fps : int
        Images par seconde du GIF.
    rayon_corps : float
        Rayon du corps perçu (px).
    taille_case : float
        Côté d'une case de la matrice de l'objet (px).
    n_cases : int
        Nombre de cases par côté (matrice n_cases × n_cases).
    distance_initiale : float
        Distance centre-à-centre à t=0 (px).

    Retour
    ------
    BytesIO ou None
        Buffer du GIF prêt à être lu, ou None si aucune donnée exploitable.
    """
    if not fenetres_v:
        return None

    dt = float(pas_direction_ms)

    # --- Vitesse moyenne sur la grille de temps ---
    n_max = max(len(v) for v in fenetres_v)
    v_mat = np.full((len(fenetres_v), n_max), np.nan)
    for i, v in enumerate(fenetres_v):
        v_mat[i, :len(v)] = v
    v_moy = np.nanmean(v_mat, axis=0)

    temps_ms = np.arange(n_max) * dt - fenetre_avant_ms
    idx_zero = int(np.argmin(np.abs(temps_ms)))

    # --- Reconstruction de la distance radiale ---
    # Convention : v < 0 = approche, v > 0 = éloignement
    distances = np.full(n_max, np.nan)
    distances[idx_zero] = distance_initiale

    for i in range(idx_zero + 1, n_max):
        distances[i] = (distances[i-1] if np.isnan(v_moy[i])
                        else distances[i-1] - v_moy[i] * dt)
    for i in range(idx_zero - 1, -1, -1):
        distances[i] = (distances[i+1] if np.isnan(v_moy[i+1])
                        else distances[i+1] + v_moy[i+1] * dt)
    distances = np.clip(distances, 0, None)

    # --- Préparation de la figure ---
    taille_grille = taille_case * n_cases
    d_max = max(200.0, float(np.nanmax(distances)) + 60.0)

    fig_anim, ax = plt.subplots(figsize=(8, 4))

    couleur_case_A = '#E88A2E'
    couleur_case_B = '#B5651D'
    for i in range(n_cases):
        for j in range(n_cases):
            x0 = -taille_grille/2 + j * taille_case
            y0 = -taille_grille/2 + i * taille_case
            couleur = couleur_case_A if (i + j) % 2 == 0 else couleur_case_B
            ax.add_patch(Rectangle((x0, y0), taille_case, taille_case,
                                   facecolor=couleur, edgecolor='white',
                                   linewidth=0.5, zorder=1))

    ax.add_patch(Circle((0, 0), rayon_corps_percevant, fill=False, edgecolor='purple', linewidth=1.2, linestyle='--', zorder=2))
    ax.add_patch(Circle((0, 0), rayon_corps_percu, facecolor='blue', edgecolor='none', zorder=3))

    individu_patch = Circle((distance_initiale, 0), rayon_corps_percu, facecolor='blue', edgecolor='none', linewidth=1.5, zorder=4)
    ax.add_patch(individu_patch)

    ligne_contact, = ax.plot([], [], color='gray', linestyle=':',
                             linewidth=1, alpha=0.6, zorder=0)
    txt_info = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='white',
                                 alpha=0.8))

    ax.set_title(titre, fontsize=12)
    ax.set_xlim(-d_max, d_max)
    ax.set_ylim(-d_max * 0.3, d_max * 0.3)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Position x (px)')
    ax.set_yticks([])

    def update(frame_idx):
        d = distances[frame_idx]
        t = temps_ms[frame_idx]
        v = v_moy[frame_idx] if not np.isnan(v_moy[frame_idx]) else 0.0

        individu_patch.center = (d, 0)
        ligne_contact.set_data([0, d], [0, 0])
        if v < -0.01:
            ligne_contact.set_color('green')
        elif v > 0.01:
            ligne_contact.set_color('red')
        else:
            ligne_contact.set_color('gray')

        txt_info.set_text(f"t = {t:+.0f} ms\nd = {d:.1f} px\nv = {v:+.3f} px/ms")
        return individu_patch, ligne_contact, txt_info

    anim = FuncAnimation(fig_anim, update, frames=n_max, interval=1000 // fps, blit=True)

    # PillowWriter exige un chemin de fichier : on passe par un fichier temporaire
    buf_gif = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        chemin_tmp = os.path.join(tmpdir, "anim.gif")
        anim.save(chemin_tmp, writer=PillowWriter(fps=fps))
        with open(chemin_tmp, "rb") as f:
            buf_gif.write(f.read())

    buf_gif.seek(0)
    plt.close(fig_anim)
    return buf_gif

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
prochains = st.checkbox(label='Afficher les deux prochains évènements', value=False)


st.sidebar.header("Paramètres")

st.sidebar.subheader("Tâche (défaut=Toutes)")
gsv = st.sidebar.checkbox("GSV")
bst = st.sidebar.checkbox("BST")
col = st.sidebar.checkbox("COL")

taches_list = []
if gsv: taches_list.append('GSV')
if bst: taches_list.append('BST')
if col: taches_list.append('COL')
if taches_list == []:
    taches_list = TACHES
    taches_tuple = (taches_list, True)
else:
    taches_tuple = (taches_list, False)

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
        st.sidebar.subheader("Rôles (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide', False)
        role2 = st.sidebar.checkbox('Suiveur', False)
    elif gg and not gs:
        st.sidebar.subheader("Individus (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide A', False)
        role2 = st.sidebar.checkbox('Guide B', False)
    else:
        st.sidebar.subheader("Individus (défaut=Tous)")
        role1 = st.sidebar.checkbox('Guide/Guide A', False)
        role2 = st.sidebar.checkbox('Suiveur/Guide B', False)

elif len(taches_list) == 1:
    st.sidebar.subheader("Sujets (défaut=Tous)")
    role1 = st.sidebar.checkbox("Sujet A", False)
    role2 = st.sidebar.checkbox("Sujet B", False)


st.sidebar.subheader("Conditions (défaut=Toutes)")
conditions_list = []
toutes_conditions = []
conditions_par_tache = {}
for tache in taches_list:
    st.sidebar.markdown(f"**{tache}**")
    # Charger la config de la tâche pour connaître ses conditions
    try:
        module = importlib.import_module(f"taches.{tache}.config_{tache}")
        classe = getattr(module, f"Config_{tache}")
        conds_tache = classe.CONDITIONS
    except (ModuleNotFoundError, AttributeError):
        st.sidebar.warning(f"Config {tache} introuvable")
        conds_tache = []
    
    conds_choisies = [
        c for c in conds_tache
        if st.sidebar.checkbox(c, key=f"cond_{tache}_{c}")
    ]
    conditions_par_tache[tache] = conds_choisies or conds_tache
    conditions_list += conds_choisies
    toutes_conditions += conds_tache

conditions_list = uniques(conditions_list)
toutes_conditions = uniques(toutes_conditions)
if not conditions_list:
    conditions_list = toutes_conditions
    conditions_tuple = (conditions_list, True)
else:
    conditions_tuple = (conditions_list, False)


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
df_essais = lister_essais(taches_list, conditions_list)
if 'essai' in df_essais.columns:
    essais_disponibles = ['Tous'] + sorted(df_essais['essai'].unique().tolist())
else:
    essais_disponibles = ['Tous']

essai = st.sidebar.selectbox("Essai :", essais_disponibles)

# Evenement
evenement = 'Tous'
if essai != 'Tous' and (rupture or reprise):
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
fenetre_avant = st.sidebar.slider("Fenêtre avant (ms)", 100, 1000, 500, 100, help="Fenêtre de temps affichée avant l'évènement sur le graphique")
fenetre_apres = st.sidebar.slider("Fenêtre après (ms)", 500, 3000, 1500, 100, help="Fenêtre de temps affichée après l'évènement sur le graphique")

# Pas direction
pas_direction = st.sidebar.slider("Fenêtre glissante d'enregistrement de la vitesse (ms)", 40, 500, 200, 20, help="Temps entre les enregistrements des positions d'un individu pour le calcul de la vitesse") //10

st.cache_data.clear()

donnees_preparees = preparer_donnees(
    sorted(taches_list),
    conditions_par_tache,
    tuple(positions_list),
    pas_direction
)

# ============================================================
# EXTRACTION DES FENÊTRES
# ============================================================

@st.cache_data
def extraire_fenetres(_donnees_preparees, taches_tuple, conditions_tuple, positions_tuple, type_events_tuple,
                       roles_tuple, essai_filtre, evenement_filtre,
                       fenetre_avant_ms, fenetre_apres_ms, pas_direction):

    taches_list = taches_tuple[0]
    conditions_list = conditions_tuple[0]
    positions_list = positions_tuple[0]
    type_events_list = type_events_tuple[0]
    roles_list = roles_tuple[0]
    
    n_avant = int(fenetre_avant_ms / 10)
    n_apres = int(fenetre_apres_ms / 10)
    
    donnees = {}
    
    for (tache, cond, nom), df in _donnees_preparees.items():
        if essai_filtre != 'Tous' and nom != essai_filtre:
            continue
        if 'contact_AB' not in df.columns:
            continue
        
        idx_ruptures, idx_reprises = detecter_ruptures_contact(df, 'contact_AB', col_temps='temps_relatif_ms', duree_min_ms= 500)
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
                        col_v = f'v_signed_A_vers_B_{pos}'
                    else:
                        col_acc = f'acc_B_vers_A_{pos}'
                        col_v = f'v_signed_B_vers_A_{pos}'
                            
                    if col_acc not in df.columns or col_v not in df.columns:
                        continue
                            
                    for idx in idx_events:
                        if idx - n_avant < 20:
                            continue
                                
                        idx_debut = max(0, idx - n_avant)
                        idx_fin = min(len(df), idx + n_apres)
                        fenetre = df.iloc[idx_debut:idx_fin]
                        
                        if len(fenetre) < n_avant + 10:
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
                                'acc': [], 'v': [],
                                'autres_evenements': [],
                                'deux_prochains': [],
                            }
                                
                        donnees[cle]['acc'].append(fenetre[col_acc].values)
                        donnees[cle]['v'].append(fenetre[col_v].values)
                        donnees[cle]['autres_evenements'].append(autres_evenements)
                        donnees[cle]['deux_prochains'].append(deux_prochains)
    
    return donnees

# Extraire les fenêtres
donnees = extraire_fenetres(
    donnees_preparees,
    taches_tuple,
    conditions_tuple,
    positions_tuple,
    type_events_tuple,
    roles_tuple,
    essai,
    evenement,
    fenetre_avant,
    fenetre_apres,
    pas_direction
)

# --- Dictionnaire de traduction pour l'affichage ---
LABELS_LISIBLES = {
    'tt': 'Curseur-Curseur',
    'rt': 'Souris guide-Curseur suiveur',
    'tr': 'Curseur guide-Souris suiveur',
    'rr': 'Souris-Souris',
    'A': 'Guide',
    'B': 'Suiveur',
    'rupture': 'Ruptures',
    'reprise': 'Reprises',
    'BB': 'BB', 'BH': 'BH', 'HB': 'HB', 'HH': 'HH',
}
if taches_list == ['GSV']:
    if gg and not gs:
        LABELS_LISIBLES['A'] = 'Guide A'
        LABELS_LISIBLES['B'] = 'Guide B'
    else :
        LABELS_LISIBLES['A'] = 'Guide'
        LABELS_LISIBLES['B'] = 'Suiveur'

def traduire(cle):
    if not cle:
        return 'Toutes les données'
    parties = cle.split('_')
    return ' | '.join(LABELS_LISIBLES.get(p, p) for p in parties)

def lisser(y, largeur=4):
    return pd.Series(y).rolling(largeur, center=True, min_periods=1).mean().to_numpy()

# ============================================================
# AFFICHAGE
# ============================================================

if not donnees:
    st.warning("Aucune donnée")
else:
    n_avant = int(fenetre_avant / 10)

    cles_triees = sorted(donnees.keys())
    
    if len(cles_triees) <= 10:
        palette = [plt.cm.tab10(i) for i in range(len(cles_triees))]
    elif len(cles_triees) <= 20:
        palette = [plt.cm.tab20(i) for i in range(len(cles_triees))]
    else:
        palette = [plt.cm.turbo(x) for x in np.linspace(0, 1, len(cles_triees))]
    
    couleurs = {cle: palette[i] for i, cle in enumerate(cles_triees)}

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # --- Boucle sur les groupes ---
    for cle in cles_triees:
        data = donnees[cle]
        fenetres_acc = data['acc']
        fenetres_v = data['v']
        
        if not fenetres_acc:
            continue
        
        # --- Matrices ---
        n_max = max(len(f) for f in fenetres_acc)
        acc_mat = np.full((len(fenetres_acc), n_max), np.nan)
        v_mat = np.full((len(fenetres_v), n_max), np.nan)
        
        for i, (a, v) in enumerate(zip(fenetres_acc, fenetres_v)):
            acc_mat[i, :len(a)] = a
            v_mat[i, :len(v)] = v
        
        # --- Moyennes ---
        acc_moy = np.nanmean(acc_mat, axis=0)
        v_moy = np.nanmean(v_mat, axis=0)
        
        # --- SEM ---
        acc_std = np.nanstd(acc_mat, axis=0)
        v_std = np.nanstd(v_mat, axis=0)
        n_obs = np.sum(~np.isnan(acc_mat), axis=0)
        n_obs_safe = np.where(n_obs > 0, n_obs, 1)
        acc_sem = acc_std / np.sqrt(n_obs_safe)
        v_sem = v_std / np.sqrt(n_obs_safe)

        if lisse :
            acc_moy = lisser(acc_moy)
            v_moy = lisser(v_moy)
            acc_sem = lisser(acc_sem)
            v_sem = lisser(v_sem)
        
        # --- Temps ---
        temps = np.arange(n_max) * 10 - fenetre_avant
        
        # --- Couleur et label ---
        couleur = couleurs[cle]
        label = traduire(cle)
        
        # --- Tracé des courbes ---
        if mode_affichage == 'Courbe':
            if label.find('Souris-Souris') >= 0:
                ax1.plot(temps, acc_moy, color=couleur, linewidth=2, linestyle='--', label=f'{label} (n={len(fenetres_acc)})')
                ax2.plot(temps, v_moy, color=couleur, linewidth=2, linestyle='--', label=f'{label} (n={len(fenetres_v)})')

            else :
                ax1.plot(temps, acc_moy, color=couleur, linewidth=2, label=f'{label} (n={len(fenetres_acc)})')
                ax2.plot(temps, v_moy, color=couleur, linewidth=2, label=f'{label} (n={len(fenetres_v)})')

            ax1.fill_between(temps, acc_moy - acc_sem, acc_moy + acc_sem, color=couleur, alpha=0.15)
            ax2.fill_between(temps, v_moy - v_sem, v_moy + v_sem, color=couleur, alpha=0.15)

        else:
            pas = max(1, echantillon // 10)
            points_temps = []
            points_acc = []
            points_v = []
            
            for a, v in zip(fenetres_acc, fenetres_v):
                n = len(a)
                t_fenetre = (np.arange(n) - n_avant) * 10
                idx_sous = np.arange(0, n, pas)
                points_temps.extend(t_fenetre[idx_sous])
                points_acc.extend(a[idx_sous])
                points_v.extend(v[idx_sous])

            if not extremes:
                points_temps = np.array(points_temps)
                points_acc = np.array(points_acc)
                points_v = np.array(points_v)
                
                p1_acc, p99_acc = np.nanpercentile(points_acc, [1, 99])
                p1_v, p99_v = np.nanpercentile(points_v, [1, 99])
                
                masque = (
                    (points_acc >= p1_acc) & (points_acc <= p99_acc) &
                    (points_v >= p1_v) & (points_v <= p99_v)
                )
                
                points_temps = points_temps[masque]
                points_acc = points_acc[masque]
                points_v = points_v[masque]
            
            ax1.scatter(points_temps, points_acc,
                        color=couleur, alpha=0.3, s=5,
                        label=f'{label} (n={len(fenetres_acc)})')
            ax2.scatter(points_temps, points_v,
                        color=couleur, alpha=0.3, s=5,
                        label=f'{label} (n={len(fenetres_v)})')
        
        # --- BARRES VERTICALES DES PROCHAINS ÉVÉNEMENTS ---
        if prochains:
            deux_prochains = data['deux_prochains']
            temps_1 = [t[0] for t in deux_prochains if t[0] is not None]
            temps_2 = [t[1] for t in deux_prochains if t[1] is not None]
        
            if temps_1:
                moy_1 = np.mean(temps_1)
                ax1.axvline(moy_1, color=couleur, linestyle='--',
                            alpha=0.6, linewidth=1.5,
                            label=f'Prochain évènement contraire')
                ax2.axvline(moy_1, color=couleur, linestyle='--',
                            alpha=0.6, linewidth=1.5)
            
            if temps_2:
                moy_2 = np.mean(temps_2)
                ax1.axvline(moy_2, color=couleur, linestyle=':',
                            alpha=0.6, linewidth=1.5,
                            label=f'Prochain même évènement')
                ax2.axvline(moy_2, color=couleur, linestyle=':',
                            alpha=0.6, linewidth=1.5)

    # --- Lignes de référence ---
    ax1.axvline(0, color='red', linestyle='--', alpha=1)
    ax2.axvline(0, color='red', linestyle='--', alpha=1)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    
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

    # --- Ajustement des limites Y ---
    def ajuster_limites(ax, marge=0.05):
        ymin, ymax = ax.get_ylim()
        amplitude = max(abs(ymin), abs(ymax))
        if amplitude == 0:
            amplitude = 1
        amplitude *= (1 + marge)
        ax.set_ylim(-amplitude, amplitude)

    ajuster_limites(ax1)
    ajuster_limites(ax2)
    
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    # Bouton pour enregistrer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    st.download_button(
        label="📂 Télécharger le graphique", data=buf, mime="image/png")

    # ANIMATION RÉACTION À UN OBJET FIXE
    with st.subheader("Animations de la réaction"):
        onglets = st.tabs([traduire(c) or "Toutes" for c in cles_triees])
        for onglet, c in zip(onglets, cles_triees):
            with onglet:
                gif_buf = construire_gif_reaction(fenetres_v=donnees[c]['v'], fenetre_avant_ms=fenetre_avant, pas_direction_ms=10, titre=traduire(c), fps=100,)
                if gif_buf is None:
                    st.info("Pas de données pour cette sélection.")
                else:
                    st.image(gif_buf, width="stretch")
                    gif_buf.seek(0)
                    st.download_button(label="📂 Télécharger l'animation", data=gif_buf, mime="image/gif", file_name=f"animation_{c or 'toutes'}.gif", key=f"dl_gif_{c}",)

    # ============================================================
    # STATISTIQUES
    # ============================================================
    
    st.subheader("Statistiques")
    
    idx_event = fenetre_avant // 10
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acc avant (px/s²)", f"{np.nanmean(acc_moy[:idx_event]) * 1e6:.2f}")
    col2.metric("Acc après (px/s²)", f"{np.nanmean(acc_moy[idx_event+40:]) * 1e6:.2f}")
    col3.metric("V avant (px/s)", f"{np.nanmean(v_moy[:idx_event]) * 1000:.2f}")
    col4.metric("V après (px/s)", f"{np.nanmean(v_moy[idx_event+20:]) * 1000:.2f}")
