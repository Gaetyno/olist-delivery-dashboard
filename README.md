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