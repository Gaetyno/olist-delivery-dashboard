# Olist Data Project

Projet Data Engineering, Data Analytics et Machine Learning basé sur le dataset Olist Brazilian E-Commerce.

## Objectif du projet

L’objectif est de construire un pipeline complet permettant d’analyser la performance logistique d’une marketplace e-commerce et de prédire les commandes à risque de retard de livraison.

Problématique métier :

> Comment identifier les commandes à risque de retard de livraison afin de réduire l’insatisfaction client et prioriser les actions opérationnelles ?

## Architecture du projet

Le projet suit une logique en couches :


data/source/
    ↓
Bronze — ingestion des CSV bruts
    ↓
Silver — nettoyage et typage des données
    ↓
Gold — tables analytiques pour KPIs, dashboard et ML
    ↓
ML — entraînement du modèle de prédiction du retard


## Lancer le dashboard Streamlit

Le projet contient un dashboard interactif Streamlit permettant d’explorer :

- les KPIs globaux de livraison ;
- l’impact des retards sur la satisfaction client ;
- les analyses par État client, catégorie produit et vendeur ;
- les alertes opérationnelles.

Avant de lancer le dashboard, exécuter le pipeline complet :

```bash
python run_pipeline.py

Ensuite, lancer l’application Streamlit :

```bash
python -m streamlit run app/streamlit_app.py
```

Le dashboard utilise automatiquement le dernier dossier Gold disponible dans :

```text
data/gold/
```
