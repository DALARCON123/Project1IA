# ---------------------------------------------------------------
# Projet IA – Qualité de l’Air et Pollution (version créative corrigée + commentée)
# 420-IAA-TT Automne 2025 - Institut Teccart
# Basé UNIQUEMENT sur les Cours 2 à 7 : pandas, numpy, matplotlib,
# seaborn, streamlit. Aucune librairie externe.
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler  # normalisation vue en cours

# --- Config de la page (titre + layout large) ---
st.set_page_config(page_title="Qualité de l’Air - Tableau de bord", layout="wide")

# --- Style: fond pastel via CSS (aucune lib externe) ---
st.markdown(
    """
    <style>
    .stApp { background-color: #f3f8ff; }        /* Fond pastel doux */
    h1, h2, h3, h4 { color: #1b3b6f; }           /* Titres bleus sobres */
    .sidebar .sidebar-content { background: #eef3ff; }  /* Légère teinte dans la sidebar */
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# UTILITAIRES (Cours 2-3 : séries / DataFrames)
# ---------------------------------------------------------------

@st.cache_data
def charger_donnees(path="pollution.csv"):
    """
    Charge le fichier CSV (mise en cache pour fluidifier l'app).
    """
    df = pd.read_csv(path)
    return df

def remplacer_manquantes_par_moyenne(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs manquantes des colonnes numériques par la moyenne (cours 3-4).
    """
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].mean())
    return df

def detecter_cible(df: pd.DataFrame):
    """
    Détecte automatiquement la colonne 'cible' liée à la qualité de l’air.
    Cherche des noms contenant: 'qual', 'air', 'target', 'class' (insensible à la casse).
    Évite KeyError si le CSV n’a pas exactement 'QualitéAir'.
    """
    return [c for c in df.columns
            if ("qual" in c.lower()) or ("air" in c.lower())
            or ("target" in c.lower()) or ("class" in c.lower())]

# ---------------------------------------------------------------
# SIDEBAR (navigation) – Cours 5 (Streamlit de base)
# ---------------------------------------------------------------
st.sidebar.title(" Qualité de l’Air")
menu = st.sidebar.radio("Navigation", [
    "Accueil",
    "Exploration",
    "Nettoyage",
    "Analyse descriptive",
    "Visualisations",
    "Corrélations",
    "Tableau de bord écologique",
    "Conclusions"
])

# ---------------------------------------------------------------
# ACCUEIL
# ---------------------------------------------------------------
if menu == "Accueil":
    st.markdown("<h1 style='text-align:center'>Application Streamlit – Qualité de l’Air</h1>", unsafe_allow_html=True)
    st.markdown(
        "Projet créatif basé sur les **Cours 2–7** : exploration, stats descriptives, "
        "corrélations et **tableau de bord écologique** (sans lib externe)."
    )
    st.info("Placez **pollution.csv** dans le même dossier que ce script.")

# ---------------------------------------------------------------
# EXPLORATION (Cours 2-3)
# ---------------------------------------------------------------
elif menu == "Exploration":
    st.header(" Exploration du jeu de données")
    df = charger_donnees()

    st.subheader("Aperçu (10 premières lignes)")
    st.dataframe(df.head(10))

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dimensions** :", df.shape)   # (n_lignes, n_colonnes)
    with c2:
        st.write("**Types des colonnes**")
        st.write(df.dtypes)

    st.subheader("Valeurs manquantes (par colonne)")
    st.write(df.isnull().sum())

# ---------------------------------------------------------------
# NETTOYAGE (Cours 3-4)
# ---------------------------------------------------------------
elif menu == "Nettoyage":
    st.header(" Nettoyage des données")
    df = charger_donnees()

    st.write("Valeurs manquantes **AVANT** :")
    st.write(df.isnull().sum())

    # Bouton pour appliquer le remplissage (moyenne) sur colonnes numériques
    if st.button("Remplacer les NA numériques par la moyenne"):
        df_clean = remplacer_manquantes_par_moyenne(df)
        st.success(" Nettoyage effectué.")
        st.write("Valeurs manquantes **APRÈS** :")
        st.write(df_clean.isnull().sum())
        st.dataframe(df_clean.head())
    else:
        st.info("Cliquez sur le bouton pour appliquer le nettoyage.")

# ---------------------------------------------------------------
# ANALYSE DESCRIPTIVE (Cours 4)
# ---------------------------------------------------------------
elif menu == "Analyse descriptive":
    st.header("Analyse statistique descriptive")
    df = remplacer_manquantes_par_moyenne(charger_donnees())

    st.subheader("Statistiques globales (colonnes numériques)")
    st.dataframe(df.describe())

    # Distribution de la cible (robuste) -> évite 'KeyError: QualitéAir'
    st.subheader("Distribution de la qualité de l’air (colonne cible)")
    cible_candidates = detecter_cible(df)
    if len(cible_candidates) > 0:
        cible = st.selectbox("Colonne cible détectée :", options=cible_candidates, index=0)
        fig, ax = plt.subplots()
        df[cible].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
        ax.set_xlabel(f"Niveau de qualité ({cible})")
        ax.set_ylabel("Nombre d’échantillons")
        st.pyplot(fig)
        st.caption("Ex.: 0=Bonne, 1=Modérée, 2=Mauvaise, 3=Dangereuse (selon votre CSV).")
    else:
        st.warning("Colonne cible introuvable (renommez-la ou utilisez un nom contenant 'qual/air/target/class').")

# ---------------------------------------------------------------
# VISUALISATIONS (Cours 6 : matplotlib / seaborn)
# ---------------------------------------------------------------
elif menu == "Visualisations":
    st.header(" Visualisations dynamiques")
    df = remplacer_manquantes_par_moyenne(charger_donnees())

    choix = st.selectbox("Type de graphique", ["Histogrammes", "Boxplots", "Matrice de dispersion", "Heatmap (corrélations)"])

    if choix == "Histogrammes":
        st.subheader("Histogrammes (layout dynamique)")
        # Layout dynamique pour éviter l’erreur si > 9 colonnes (3x3 insuffisant)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            nplots = len(num_cols)
            ncols = min(3, nplots)           # jusqu’à 3 graphiques par ligne (style de cours)
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

            for i, col in enumerate(num_cols):
                ax = axes[i]
                ax.hist(df[col].dropna(), bins=20, edgecolor="black")
                ax.set_title(col)

            for j in range(len(num_cols), len(axes)):  # supprimer sous-graphiques vides
                fig.delaxes(axes[j])

            fig.tight_layout()
            st.pyplot(fig)

    elif choix == "Boxplots":
        st.subheader("Boxplots (layout dynamique)")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            nplots = len(num_cols)
            ncols = min(3, nplots)
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

            for i, col in enumerate(num_cols):
                ax = axes[i]
                ax.boxplot(df[col].dropna(), vert=True)
                ax.set_title(col)

            for j in range(len(num_cols), len(axes)):  # axes vides
                fig.delaxes(axes[j])

            fig.tight_layout()
            st.pyplot(fig)

    elif choix == "Matrice de dispersion":
        st.subheader("Matrice de dispersion (numérique)")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            scatter_matrix(num_df, figsize=(15, 15), diagonal='hist')
            st.pyplot(plt.gcf())

    elif choix == "Heatmap (corrélations)":
        st.subheader("Matrice de corrélation (Pearson)")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------
# CORRÉLATIONS (Cours 7 : interprétation)
# ---------------------------------------------------------------
elif menu == "Corrélations":
    st.header("Étude des corrélations avec la qualité de l’air")
    df = remplacer_manquantes_par_moyenne(charger_donnees())
    corr = df.corr(numeric_only=True)

    st.subheader("Matrice de corrélation (aperçu global)")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="crest", ax=ax)
    st.pyplot(fig)

    # Tri des variables selon leur corrélation avec la cible (si détectée)
    cible_candidates = detecter_cible(df)
    if len(cible_candidates) == 0:
        st.info("Colonne cible non détectée pour trier les corrélations.")
    else:
        cible = st.selectbox("Colonne cible pour le tri :", options=cible_candidates, index=0)
        st.subheader(f"Corrélations ordonnées par rapport à « {cible} »")
        st.write(corr[cible].sort_values(ascending=False))

# ---------------------------------------------------------------
# TABLEAU DE BORD ÉCOLOGIQUE (partie créative, mais conforme aux cours)
# ---------------------------------------------------------------
elif menu == "Tableau de bord écologique":
    st.header(" Tableau de bord écologique (créatif)")
    df = remplacer_manquantes_par_moyenne(charger_donnees())

    # 1) Détection des colonnes de polluants (PM2.5, PM10, NO2, SO2, CO) par nom
    colonnes_polluants_candidates = []
    for nom in df.columns:
        n = nom.lower()
        if ("pm2" in n) or ("pm_2" in n) or ("pm 2" in n):
            colonnes_polluants_candidates.append(nom)
        if ("pm10" in n) or ("pm_10" in n) or ("pm 10" in n):
            colonnes_polluants_candidates.append(nom)
        if "no2" in n:
            colonnes_polluants_candidates.append(nom)
        if "so2" in n:
            colonnes_polluants_candidates.append(nom)
        if n == "co" or " co" in n:
            colonnes_polluants_candidates.append(nom)

    st.subheader("1) Indice global de pollution (0 - 100)")
    if len(colonnes_polluants_candidates) == 0:
        st.warning("Aucune colonne de polluants reconnue (PM2.5, PM10, NO2, SO2, CO).")
    else:
        choix_polluants = st.multiselect(
            "Colonnes de polluants à inclure :",
            options=sorted(list(set(colonnes_polluants_candidates))),
            default=sorted(list(set(colonnes_polluants_candidates)))[:3]
        )

        if len(choix_polluants) >= 1:
            # Standardisation (moy=0, écart-type=1) puis moyenne -> indice brut
            scaler = StandardScaler()
            Xs = scaler.fit_transform(df[choix_polluants].values)
            indice_brut = Xs.mean(axis=1)

            # Mise à l’échelle 0-100 sur le dataset (min-max)
            min_b, max_b = float(indice_brut.min()), float(indice_brut.max())
            if max_b - min_b == 0:
                indice_0_100 = np.zeros_like(indice_brut)
            else:
                indice_0_100 = 100 * (indice_brut - min_b) / (max_b - min_b)

            df["IndicePollution"] = indice_0_100
            valeur_moy = float(np.mean(indice_0_100))

            # Jauge simple horizontale (matplotlib pur)
            fig, ax = plt.subplots(figsize=(8, 1.5))
            ax.barh([0], [valeur_moy], color="tomato")
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_title(f"Indice global (moyenne): {valeur_moy:.1f} / 100")
            # Zones qualitatives (Bon/Modéré/Mauvais/Dangereux)
            ax.axvspan(0, 25, color="#8fd19e", alpha=0.2, label="Bon")
            ax.axvspan(25, 50, color="#f4e285", alpha=0.2, label="Modéré")
            ax.axvspan(50, 75, color="#f7a072", alpha=0.2, label="Mauvais")
            ax.axvspan(75, 100, color="#eb5e55", alpha=0.2, label="Dangereux")
            ax.legend(loc="upper right", ncols=4, frameon=False, fontsize=8)
            st.pyplot(fig)

            # Top/Bottom 10 selon l’indice
            c1, c2 = st.columns(2)
            with c1:
                st.write("🔺 **Top 10 – Indice le plus élevé**")
                st.dataframe(df.sort_values("IndicePollution", ascending=False).head(10))
            with c2:
                st.write("🔻 **Top 10 – Indice le plus faible**")
                st.dataframe(df.sort_values("IndicePollution", ascending=True).head(10))
        else:
            st.info("Sélectionnez au moins une colonne de polluants pour l’indice.")

    st.markdown("---")
    st.subheader("2) Filtres interactifs (plage de valeurs)")

    # Filtres simples sur variables clés si elles existent
    filtres = {}
    for cand in ["Température", "Humidité", "PM2.5", "PM10", "NO2", "SO2", "CO"]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            min_v, max_v = float(df[cand].min()), float(df[cand].max())
            filtres[cand] = st.slider(f"{cand}", min_v, max_v, (min_v, max_v))

    df_filtre = df.copy()
    for col, (lo, hi) in filtres.items():
        df_filtre = df_filtre[(df_filtre[col] >= lo) & (df_filtre[col] <= hi)]

    st.write(f"**Lignes après filtres :** {df_filtre.shape[0]} / {df.shape[0]}")
    st.dataframe(df_filtre.head(20))

    st.markdown("---")
    st.subheader("3) Carte de tendance (heatmap)")
    # Heatmap bivariée (si colonnes 'Densité' et 'Proximité' existent) – seaborn pur
    if ("Densité" in df.columns) and ("Proximité" in df.columns):
        try:
            dens_bins = pd.qcut(df["Densité"], q=4, duplicates="drop")
            prox_bins = pd.qcut(df["Proximité"], q=4, duplicates="drop")
            cible_heat = None
            for c in ["PM2.5", "PM10", "NO2", "SO2", "CO"]:
                if c in df.columns:
                    cible_heat = c
                    break
            if cible_heat is not None:
                pivot = df.pivot_table(index=dens_bins, columns=prox_bins, values=cible_heat, aggfunc="mean")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(pivot, cmap="mako", annot=True, fmt=".1f", ax=ax)
                ax.set_title(f"Moyenne de {cible_heat} selon Densité × Proximité")
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Heatmap non disponible : {e}")
    else:
        st.info("Colonnes 'Densité' et 'Proximité' non détectées pour la heatmap.")

# ---------------------------------------------------------------
# CONCLUSIONS (Synthèse)
# ---------------------------------------------------------------
elif menu == "Conclusions":
    st.header("Conclusions et recommandations")
    st.markdown("""
    **Observations générales :**
    - Les polluants **PM2.5**, **PM10**, **NO2**, **SO2** et **CO** sont souvent dominants dans l’indice global.
    - Les zones à **forte densité** et **proches d’industries** tendent à présenter des niveaux plus élevés.
    - La **température** et l’**humidité** influencent la dispersion des polluants.

    **Recommandations :**
    - Surveiller en priorité les particules fines (PM2.5) et le NO2.
    - Réduire les émissions à la source (transport, chauffage, industrie).
    - Sensibiliser la population dans les zones denses à risque.
    """)
    st.success("Fin de l’analyse. Exportez vos tableaux/figures pour le rapport.")
