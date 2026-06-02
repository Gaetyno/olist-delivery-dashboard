# Olist Data Project

Projet data engineering, analytics et machine learning basé sur le dataset Olist Brazilian E-Commerce.

## Objectif

Construire un pipeline complet de traitement de données :

- compréhension des données ;
- ingestion Bronze ;
- nettoyage Silver ;
- modélisation Gold ;
- machine learning ;
- dashboard final.

## Structure du projet

```text
data/
  source/   Données CSV brutes non versionnées
  bronze/   Données brutes ingérées
  silver/   Données nettoyées
  gold/     Données analytiques

notebooks/
  Notebooks d'exploration et d'analyse

src/
  Scripts Python modulaires

docs/
  Documentation métier et technique

reports/
  Rapports générés

models/
  Modèles ML sauvegardés