import sys
from pathlib import Path

# --- Ajouter la racine au PYTHONPATH ---
RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from traitement_fichier import enrichir_dataframe

st.cache_data.clear()

# --- Configuration ---
CACHE_DIR = RACINE / "caches"
CONDITIONS = ['BB', 'BH', 'HB', 'HH', 'GB', 'GH']
POSITIONS = ['tt', 'rt', 'tr', 'rr']
TYPES_EVENTS = ['rupture', 'reprise']
ROLES = ['A', 'B']

st.title("Analyse temporelle des réactions")

# ============================================================
# CHARGEMENT
# ============================================================

# LISTE DES ESSAIS
@st.cache_data
def lister_essais():
    essais = []
    for cond in CONDITIONS:
        for fichier in CACHE_DIR.glob(f"{cond}_*.parquet"):
            nom = fichier.stem.replace(f"{cond}_", "")
            essais.append({
                'condition': cond,
                'essai': nom,
            })
    return pd.DataFrame(essais)

df_essais = lister_essais()

@st.cache_data
def lister_evenements(essai, type_event):
    if essai == 'Tous':
        return ['Tous']

    fichier = None
    for cond in CONDITIONS:
        candidat = CACHE_DIR / f"{cond}_{essai}.parquet"
        if candidat.exists():
            fichier = candidat
            break

    if fichier is None:
        return ['Tous']

    df = pd.read_parquet(fichier)

    if 'contact_AB' not in df.columns:
        return ['Tous']

    idx_ruptures, idx_reprises = detecter_ruptures_contact(
        df, 'contact_AB', col_temps='temps_relatif_ms'
    )

    if type_event == 'rupture':
        idx_events = idx_ruptures
    elif type_event == 'reprise':
        idx_events = idx_reprises
    else:
        idx_events = list(idx_ruptures) + list(idx_reprises)

    return ['Tous'] + list(range(len(idx_events)))


def detecter_ruptures_contact(df: pd.DataFrame, col_contact: str,
                               col_temps: str = "temps_relatif_ms",
                               duree_min_ms: float = 0) -> tuple[np.ndarray, np.ndarray]:
    diff = df[col_contact].diff()
    idx_ruptures = df.index[diff == -1].tolist()
    idx_reprises = df.index[diff == 1].tolist()

    temps = pd.to_numeric(df[col_temps], errors='coerce').to_numpy()

    ruptures_valides = []
    reprises_valides = []

    for idx_r in idx_ruptures:
        suivant = next((i for i in idx_reprises if i > idx_r), None)
        fin_idx = suivant if suivant is not None else len(df) - 1
        duree_ms = temps[fin_idx] - temps[idx_r]

        if duree_ms >= duree_min_ms:
            ruptures_valides.append(idx_r)
            if suivant is not None:
                reprises_valides.append(suivant)

    return np.array(ruptures_valides), np.array(reprises_valides)


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
mode_affichage = st.sidebar.radio(
    "Mode d'affichage :", ['Courbe', 'Nuage de points'], index=0
)

if mode_affichage == 'Nuage de points':
    extremes = st.checkbox(label='Afficher extrèmes', value=True)
    lisse = False
else:
    extremes = True
    lisse = st.checkbox(label="Lisser la courbe", value = True)
prochains = st.checkbox(label='Afficher les deux prochains évènements', value=False)


st.sidebar.header("Paramètres")

st.sidebar.subheader("Type d'évènement (défaut=Tous)")
rupture = st.sidebar.checkbox("Ruptures", False)
reprise = st.sidebar.checkbox("Reprises", False)

st.sidebar.subheader("Positions d'analyse (défaut=Toutes)")
tt = st.sidebar.checkbox('Curseur-Curseur', False)
rr = st.sidebar.checkbox('Souris-Souris', False)
rt = st.sidebar.checkbox('Souris guide-Curseur suiveur', False)
tr = st.sidebar.checkbox('Curseur guide-Souris suiveur', False)

st.sidebar.subheader("Mode de jeu")
gsv = st.sidebar.checkbox('Guide-Suiveur')
gg = st.sidebar.checkbox('Guide-Guide')

if gsv and not gg:
    st.sidebar.subheader("Rôles (défaut=Tous)")
    role1 = st.sidebar.checkbox('Guide', False)
    role2 = st.sidebar.checkbox('Suiveur', False)
elif gg and not gsv:
    st.sidebar.subheader("Individus (défaut=Tous)")
    role1 = st.sidebar.checkbox('Guide A', False)
    role2 = st.sidebar.checkbox('Guide B', False)


st.sidebar.subheader("Conditions (défaut=Toutes)")
if gsv or not gg:
    bb = st.sidebar.checkbox('BB', False)
    bh = st.sidebar.checkbox('BH', False)
    hb = st.sidebar.checkbox('HB', False)
    hh = st.sidebar.checkbox('HH', False)
if gg or not gsv:
    gb = st.sidebar.checkbox('Guide-Guide B', False)
    gh = st.sidebar.checkbox('Guide-Guide H', False)


# ============================================================
# FILTRAGE
# ============================================================
conditions_list = []
conditions_list_gsv = []
if gsv:
    if bb: conditions_list_gsv.append('BB')
    if bh: conditions_list_gsv.append('BH')
    if hb: conditions_list_gsv.append('HB')
    if hh: conditions_list_gsv.append('HH')
    if conditions_list_gsv == []:
        conditions_list_gsv = ['BB', 'BH', 'HB', 'HH']
        conditions_tuple_gsv = (conditions_list_gsv, True)
    else:
        conditions_tuple_gsv = (conditions_list_gsv, False)
else:
    conditions_tuple_gsv = (conditions_list_gsv, True)

conditions_list_gg = []
if gg:
    if gb: conditions_list_gg.append('GB')
    if gh: conditions_list_gg.append('GH')
    if conditions_list_gg == []:
        conditions_list_gg = ['GB', 'GH']
        conditions_tuple_gg = (conditions_list_gg, True)
    else:
        conditions_tuple_gg = (conditions_list_gg, False)
else:
    conditions_tuple_gg = (conditions_list_gg, True)

if conditions_tuple_gsv[1]:
    conditions_list += conditions_list_gg
if conditions_tuple_gg[1]:
    conditions_list += conditions_list_gsv

if gsv or gg:
    conditions_tuple = (conditions_list, False)
else :
    conditions_list = CONDITIONS
    conditions_tuple = (conditions_list, True)

print(conditions_tuple_gsv)
print(conditions_tuple_gg)
print(conditions_tuple)

positions_list = []
if tt: positions_list.append('tt')
if rr: positions_list.append('rr')
if rt: positions_list.append('rt')
if tr: positions_list.append('tr')
if positions_list == []:
    positions_tuple = (POSITIONS, True)
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

if (gsv and not gg) or (gg and not gsv):
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
if conditions_list == CONDITIONS:
    essais_disponibles = ['Tous'] + sorted(df_essais['essai'].unique().tolist())
else:
    essais_filtres = df_essais[df_essais['condition'].isin(conditions_list)]['essai'].unique()
    essais_disponibles = ['Tous'] + sorted(essais_filtres.tolist())

essai = st.sidebar.selectbox("Essai :", essais_disponibles)

# Evenement
evenement = 'Tous'
if essai != 'Tous' and (rupture or reprise):
    type_event_pour_event = 'rupture' if rupture else 'reprise'
    evenements_disponibles = lister_evenements(essai, type_event_pour_event)
    evenement = st.sidebar.selectbox("Événement :", evenements_disponibles)

# Echantillonage
if mode_affichage == "Nuage de points":
    echantillon = st.sidebar.slider("Echantillonage (ms)", 10, 500, 100, 10, help="Temps entre les différents enregistrements")

# Fenêtre
fenetre_avant = st.sidebar.slider("Fenêtre avant (ms)", 100, 1000, 500, 100, help="Fenêtre de temps affichée avant l'évènement sur le graphique")
fenetre_apres = st.sidebar.slider("Fenêtre après (ms)", 500, 3000, 1500, 100, help="Fenêtre de temps affichée après l'évènement sur le graphique")

# Pas direction
pas_direction = st.sidebar.slider("Fenêtre glissante d'enregistrement de la vitesse (ms)", 40, 500, 200, 20, help="Temps entre les enregistrements des positions d'un individu pour le calcul de la vitesse") //10

# ============================================================
# EXTRACTION DES FENÊTRES
# ============================================================

@st.cache_data
def extraire_fenetres(conditions_tuple_gsv, conditions_tuple_gg, conditions_tuple, positions_tuple, type_events_tuple,
                       roles_tuple, essai_filtre, evenement_filtre,
                       fenetre_avant_ms, fenetre_apres_ms, pas_direction):

    conditions_list = conditions_tuple[0]
    positions_list = positions_tuple[0]
    type_events_list = type_events_tuple[0]
    roles_list = roles_tuple[0]
    
    n_avant = int(fenetre_avant_ms / 10)
    n_apres = int(fenetre_apres_ms / 10)
    
    donnees = {}
    
    for cond in conditions_list:
        fichiers = list(CACHE_DIR.glob(f"{cond}_*.parquet"))
            
        for fichier in fichiers:
            nom = fichier.stem.replace(f"{cond}_", "")
            if essai_filtre != 'Tous' and nom != essai_filtre:
                continue
            
            df = pd.read_parquet(fichier)
            if 'contact_AB' not in df.columns:
                continue

            # Recalcul en mémoire des colonnes enrichies avec le pas_direction choisi
            for pos in positions_list:
                df = enrichir_dataframe(df, pos, pas_direction)
            
            idx_ruptures, idx_reprises = detecter_ruptures_contact(
                df, 'contact_AB', col_temps='temps_relatif_ms'
            )
            tous_les_evenements = (
                [(idx, 'rupture') for idx in idx_ruptures] +
                [(idx, 'reprise') for idx in idx_reprises]
            )
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
                            prochains = [
                                idx_autre for idx_autre, _ in tous_les_evenements
                                if idx_autre > idx
                            ]
                            prochains = sorted(prochains)[:2]
                            
                            deux_prochains = [
                                (idx_autre - idx) * 10 for idx_autre in prochains
                            ]
                            while len(deux_prochains) < 2:
                                deux_prochains.append(None)
                            
                            # Groupe
                            groupes = []
                            if cond in conditions_tuple_gsv[0]:
                                if not conditions_tuple[1]:
                                    groupes.append('GSV')
                                if not conditions_tuple_gsv[1]:
                                    groupes.append(cond)
                            if cond in conditions_tuple_gg[0]:
                                if not conditions_tuple[1]:
                                    groupes.append('GG')
                                if not conditions_tuple_gg[1]:
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
    conditions_tuple_gsv,
    conditions_tuple_gg,
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
if gg and not gsv:
    LABELS_LISIBLES['A'] = 'Guide A'
    LABELS_LISIBLES['B'] = 'Guide B'

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
            ax1.plot(temps, acc_moy, color=couleur, linewidth=2,
                     label=f'{label} (n={len(fenetres_acc)})')
            ax1.fill_between(temps, acc_moy - acc_sem, acc_moy + acc_sem,
                             color=couleur, alpha=0.15)
            
            ax2.plot(temps, v_moy, color=couleur, linewidth=2,
                     label=f'{label} (n={len(fenetres_v)})')
            ax2.fill_between(temps, v_moy - v_sem, v_moy + v_sem,
                             color=couleur, alpha=0.15)
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
        label="📂 Télécharger le graphique",
        data=buf,
        mime="image/png"
    )
    
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
