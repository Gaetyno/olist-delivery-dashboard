---
title: Olist Delivery Performance Dashboard
emoji: 📦
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---


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

Le dashboard Streamlit inclut également un onglet de prédiction ML permettant de simuler le risque de retard d'une commande à partir des variables utilisées par le modèle.

## PostgreSQL

Le projet utilise PostgreSQL via Docker pour stocker les tables Gold générées par le pipeline.

### Lancer PostgreSQL

```bash
docker compose up -d

### Vérifier que PostgreSQL tourne

```bash
docker ps
```

Le conteneur attendu est :

```text
olist_postgres
```

### Charger les tables Gold dans PostgreSQL

Le chargement est intégré au pipeline complet :

```bash
python run_pipeline.py
```

Il est aussi possible de lancer uniquement le chargement PostgreSQL :

```bash
python src/loading/load_gold_to_postgres.py
```

Les tables chargées sont :

```text
orders_enriched
delivery_kpis_by_state
delivery_kpis_by_seller
delivery_kpis_by_category
ml_dataset
```

### Source des données du dashboard

Le dashboard Streamlit lit en priorité les données depuis PostgreSQL.

Si PostgreSQL n’est pas disponible, l’application utilise automatiquement les fichiers CSV Gold comme fallback.

Ordre de chargement :

```text
1. PostgreSQL — table orders_enriched
2. CSV Gold — data/gold/gold_.../orders_enriched.csv
```

Pour utiliser PostgreSQL avec Streamlit, lancer d’abord la base :

```bash
docker compose up -d
```

Puis s’assurer que les tables Gold ont bien été chargées :

```bash
python run_pipeline.py
```

Enfin, lancer le dashboard :

```bash
python -m streamlit run app/streamlit_app.py
```
