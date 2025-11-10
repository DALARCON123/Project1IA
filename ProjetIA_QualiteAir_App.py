# ---------------------------------------------------------------
# Projet IA – Qualité de l’Air et Pollution
# 420-IAA-TT Automne 2025 - Institut Teccart
# Basé sur les Cours 2 à 7 (pandas, numpy, seaborn, matplotlib, sklearn, streamlit)
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

st.set_page_config(page_title="Analyse de la Qualité de l’Air", layout="wide")

# === Style (fond pastel) : seulement du CSS via Streamlit, rien d’externe ===
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f8ff; /* pastel doux pour fond */
    }
    /* titres légèrement plus foncés */
    h1, h2, h3, h4 { color: #1b3b6f; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------
# Fonctions utilitaires (Cours 2-3 : séries/dataframes)
# ---------------------------------------------------------------
@st.cache_data
def charger_donnees():
    """
    Charge le CSV pollution.csv depuis le dossier courant.
    Utilise la mise en cache de Streamlit pour éviter de recharger à chaque interaction.
    """
    data = pd.read_csv("pollution.csv")
    return data

def nettoyer_donnees(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cours 3-4 : Remplace les NA numériques par la moyenne de chaque colonne.
    (Stratégie simple, vue en classe ; pas d’algos externes.)
    """
    data = data.copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].fillna(data[col].mean())
    return data

def detecter_colonne_cible(df: pd.DataFrame):
    """
    Détecte automatiquement la colonne ‘cible’ de qualité de l’air pour éviter KeyError.
    Cherche des noms contenant 'qual', 'air', 'target' ou 'class' (insensible à la casse).
    """
    candidates = [c for c in df.columns
                  if ("qual" in c.lower()) or ("air" in c.lower())
                  or ("target" in c.lower()) or ("class" in c.lower())]
    return candidates

# ---------------------------------------------------------------
# Barre latérale (Cours 5 : Streamlit de base)
# ---------------------------------------------------------------
st.sidebar.title("🌍 Projet IA – Qualité de l’Air")
menu = st.sidebar.radio("Navigation", [
    "Accueil",
    "Exploration des données",
    "Analyse descriptive",
    "Visualisations dynamiques",
    "Corrélations",
    "Conclusions"
])

# ---------------------------------------------------------------
# ACCUEIL
# ---------------------------------------------------------------
if menu == "Accueil":
    # Cours 5 : mise en page Streamlit + image en ligne (pas de lib externe)
    st.markdown("<h1 style='text-align:center;'>Application Streamlit – Étude de la Qualité de l’Air</h1>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7c/Air_pollution.jpg", use_container_width=True)
    st.markdown("""
    Cette application a été conçue dans le cadre du cours **420-IAA-TT (Automne 2025)**.

    Elle permet d’explorer un jeu de données sur la **pollution atmosphérique** au Canada,
    d’effectuer des analyses statistiques descriptives, d’étudier les corrélations
    et de visualiser les tendances entre plusieurs facteurs environnementaux
    (température, humidité, polluants chimiques, densité de population, etc.)
    et la **qualité de l’air**.
    """)

# ---------------------------------------------------------------
# EXPLORATION DES DONNÉES (Cours 2-3)
# ---------------------------------------------------------------
elif menu == "Exploration des données":
    st.header("🔎 Exploration du jeu de données")
    data = charger_donnees()
    st.write("Aperçu du jeu de données (5 premières lignes) :")
    st.dataframe(data.head())

    st.subheader("Dimensions et types des variables")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Nombre d’échantillons :** {data.shape[0]}")
    with c2:
        st.write(f"**Nombre de colonnes :** {data.shape[1]}")
    st.write(data.dtypes)

    st.subheader("Valeurs manquantes (par colonne)")
    st.write(data.isnull().sum())

    # Bouton pour appliquer le nettoyage (remplacement NA par moyenne)
    if st.button("Nettoyer les données manquantes"):
        data_clean = nettoyer_donnees(data)
        st.success("✅ Valeurs manquantes remplacées par la moyenne (colonnes numériques).")
        st.write(data_clean.isnull().sum())
        st.dataframe(data_clean.head())

# ---------------------------------------------------------------
# ANALYSE DESCRIPTIVE (Cours 4)
# ---------------------------------------------------------------
elif menu == "Analyse descriptive":
    st.header("📊 Analyse statistique descriptive")
    data = nettoyer_donnees(charger_donnees())

    st.write("Statistiques descriptives générales (toutes colonnes numériques) :")
    st.dataframe(data.describe())

    # Distribution de la cible (robuste, évite KeyError si le nom varie)
    st.markdown("#### Distribution des niveaux de qualité de l’air")
    cible_candidates = detecter_colonne_cible(data)
    if len(cible_candidates) > 0:
        cible = st.selectbox("Colonne 'qualité de l’air' détectée :", options=cible_candidates, index=0)
        fig, ax = plt.subplots()
        # Tri par index pour que 0,1,2,3 restent dans l’ordre
        data[cible].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
        ax.set_xlabel(f"Niveau de Qualité ({cible})")
        ax.set_ylabel("Nombre d’échantillons")
        st.pyplot(fig)
        st.caption("Ex.: 0=Bonne, 1=Modérée, 2=Mauvaise, 3=Dangereuse (selon votre CSV).")
    else:
        st.warning("Colonne de qualité d’air non trouvée (renommez-la ou laissez un nom contenant 'qual/air/target/class').")

# ---------------------------------------------------------------
# VISUALISATIONS DYNAMIQUES (Cours 6 : matplotlib / seaborn)
# ---------------------------------------------------------------
elif menu == "Visualisations dynamiques":
    st.header("📈 Visualisations dynamiques")
    data = nettoyer_donnees(charger_donnees())

    choix = st.selectbox("Choisir un type de graphique :", 
                         ["Histogrammes", "Boxplots", "Heatmap", "Scatter Matrix"])

    if choix == "Histogrammes":
        st.subheader("Histogrammes des variables numériques")
        # Layout dynamique pour éviter l’erreur quand il y a > 9 colonnes
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            nplots = len(num_cols)
            ncols = min(3, nplots)                  # jusqu’à 3 par ligne (style vu en classe)
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
            axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
            for i, col in enumerate(num_cols):
                ax = axes[i]
                ax.hist(data[col].dropna(), bins=20, edgecolor="black")
                ax.set_title(col)
            # Supprimer axes vides si sobran
            for j in range(len(num_cols), len(axes)):
                fig.delaxes(axes[j])
            fig.tight_layout()
            st.pyplot(fig)

    elif choix == "Boxplots":
        st.subheader("Boxplots des variables numériques")
        # Layout dynamique (évite ValueError: Layout 3x3 > nplots)
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
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
                ax.boxplot(data[col].dropna(), vert=True)
                ax.set_title(col)
            for j in range(len(num_cols), len(axes)):
                fig.delaxes(axes[j])
            fig.tight_layout()
            st.pyplot(fig)

    elif choix == "Heatmap":
        st.subheader("Heatmap des corrélations (Pearson)")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif choix == "Scatter Matrix":
        st.subheader("Matrice de dispersion (numérique)")
        num_df = data.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            st.info("Aucune colonne numérique détectée.")
        else:
            scatter_matrix(num_df, figsize=(15,15), diagonal='kde')
            st.pyplot(plt.gcf())

# ---------------------------------------------------------------
# CORRÉLATIONS (Cours 7 : interprétation de corrélations)
# ---------------------------------------------------------------
elif menu == "Corrélations":
    st.header("🔬 Étude des corrélations")
    data = nettoyer_donnees(charger_donnees())
    corr = data.corr(numeric_only=True)

    st.write("Matrice de corrélation (Pearson)")
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(corr, cmap='crest', annot=True, ax=ax)
    st.pyplot(fig)

    st.markdown("#### Variables les plus corrélées avec la qualité de l’air")
    cible_candidates = detecter_colonne_cible(data)
    if len(cible_candidates) > 0:
        cible = st.selectbox("Colonne cible pour tri des corrélations :", options=cible_candidates, index=0)
        st.write(corr[cible].sort_values(ascending=False))
    else:
        st.info("Colonne 'qualité de l’air' non trouvée pour le tri des corrélations.")

# ---------------------------------------------------------------
# CONCLUSIONS (synthèse textuelle simple)
# ---------------------------------------------------------------
elif menu == "Conclusions":
    st.header("📘 Conclusions et recommandations")
    st.markdown("""
    **Résumé automatique :**
    - Les facteurs souvent corrélés à la qualité de l’air incluent **PM2.5**, **PM10**, **NO₂** et **SO₂** (selon votre dataset).
    - L’humidité et la température influencent la dispersion des polluants.
    - La densité de population et la proximité industrielle peuvent augmenter le niveau de pollution.
    
    **Recommandations :**
    - Promouvoir la réduction des émissions industrielles proches des zones urbaines.
    - Surveiller régulièrement les particules fines (PM2.5) et le dioxyde d’azote (NO₂).
    - Développer des politiques locales pour diminuer les sources de combustion (transports, chauffage, etc.).
    """)

    st.success("🎯 Analyse terminée – vous pouvez maintenant exporter vos résultats ou faire des captures pour le rapport final.")
