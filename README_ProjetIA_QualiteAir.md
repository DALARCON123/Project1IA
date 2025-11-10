README – Projet IA : Qualité de l’Air et Pollution
Cours : 420-IAA-TT (Automne 2025 – Institut Teccart)
Auteure : Diana Marcela Alarcón Marin
Application développée avec : pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit
Objectif du projet
Ce projet applique les notions des cours 2 à 7 du module Apprentissage Automatique 1 pour créer une application Streamlit d’analyse de la pollution atmosphérique et de visualisation interactive.
Structure du dossier

project-1-pollution/
│
├── pollution.csv
├── requirements.txt
├── ProjetIA_QualiteAir_App_creatif.py
└── README_ProjetIA_QualiteAir.docx

Installation et exécution
Créer un environnement virtuel et l’activer :
python -m venv envIA1
envIA1\Scripts\activate
Installer les dépendances : pip install -r requirements.txt
Lancer l’application : streamlit run ProjetIA_QualiteAir_App_creatif.py
Navigation de l’application
Section	Description	Concepts du cours
Accueil	Présentation du projet, image et description du jeu de données	Streamlit – mise en page
Exploration	Dimensions, types de colonnes, valeurs manquantes	Cours 2-3 : séries et DataFrames
Nettoyage	Remplacement des valeurs manquantes par la moyenne	Cours 3-4 : statistiques descriptives
Analyse descriptive	Statistiques globales et distribution de la variable cible	Cours 4 : moyenne, médiane, écart-type
Visualisations	Histogrammes, boxplots, heatmap, scatter matrix	Cours 6 : matplotlib et seaborn
Corrélations	Matrice et tri des corrélations	Cours 7 : corrélation de Pearson
Tableau de bord écologique	Indice global de pollution + filtres interactifs	Cours 6-7 : normalisation et moyenne
Conclusions	Résumé et recommandations finales	Synthèse finale
Description du tableau de bord écologique

 Indice global de pollution (0–100) : basé sur PM2.5, PM10, NO₂, SO₂ et CO, calculé avec StandardScaler.
 Filtres interactifs : permettent d’observer les variations selon température, humidité ou polluants.
 Carte de tendance (heatmap) : moyenne d’un polluant selon Densité × Proximité.

Points à observer pour répondre à l’énoncé du professeur

- Présence des valeurs manquantes → Exploration
- Nettoyage du jeu de données → Nettoyage
- Moyenne, médiane, écart-type → Analyse descriptive
- Visualisation de distributions → Visualisations
- Corrélation entre variables → Corrélations
- Résumé final et recommandations → Conclusions
- Indice global et filtres → Tableau de bord écologique

Aspect visuel
Fond pastel bleu clair (#f3f8ff), titres bleus foncés (#1b3b6f), sidebar claire (#eef3ff). Styles appliqués via CSS intégré.
 Conseils de présentation orale

1. Présenter le contexte dans 'Accueil'.
2. Montrer la détection automatique de la colonne QualitéAir.
3. Expliquer le nettoyage (NA → moyenne).
4. Comparer les graphiques (avant/après).
5. Montrer l’indice global et la heatmap.
6. Conclure avec les recommandations.

