# SAS_IA

[![CI](https://github.com/<votre-utilisateur>/<votre-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<votre-utilisateur>/<votre-repo>/actions/workflows/ci.yml)

# 🚀 Générateur & Prédicteur de Dataset Linéaire

---

## 🗂️ Sommaire
- [Description](#-description)
- [Fonctionnalités](#-fonctionnalités)
- [Installation rapide](#-installation-rapide)
- [Interface Utilisateur (Streamlit)](#-interface-utilisateur-streamlit)
- [Exemples d’utilisation](#-exemples-dutilisation)
- [Tests unitaires](#-tests-unitaires)
- [CI/CD](#cicd)
- [Journalisation](#journalisation)
- [Structure du projet](#-structure-du-projet)
- [Auteur](#-auteur)

---

## 📝 Description

API moderne pour :
- Générer des datasets linéaires à 2 features (dont une change de signe selon l’heure)
- Stocker chaque dataset en base SQLite avec identifiant unique
- Prédire la classe (0/1) via régression logistique scikit-learn
- Réentraîner le modèle à chaud, suivi des expériences avec MLflow
- Vérifier la santé de l’API
- Interface utilisateur sécurisée (Streamlit) avec authentification et journalisation

---

**Légende :**
- L’utilisateur interagit avec l’API FastAPI ou l’interface Streamlit.
- FastAPI gère la génération, la prédiction et le réentraînement.
- Les datasets sont stockés dans SQLite.
- Les modèles sont entraînés avec scikit-learn.
- Les expériences et modèles sont suivis avec MLflow.
- Toutes les actions sur l’UI sont journalisées.

---

## ⚡ Fonctionnalités

- **POST /generate** : Génère et stocke un dataset (features, target, inversion selon l’heure)
- **GET /predict** : Prédit la classe sur le dernier dataset généré
- **POST /retrain** : Réentraîne le modèle sur le dernier dataset, logue dans MLflow
- **GET /health** : Vérifie la disponibilité de l’API
- **UI Streamlit** : Interface web avec authentification, boutons pour chaque action, et logs

---

## 🚀 Installation rapide

```bash
# Cloner le repo
 git clone <url_du_repo>
 cd <nom_du_repo>

# Installer les dépendances
 pip install -r requirements.txt

# Lancer l’API FastAPI
 uvicorn main:app --reload

# (Optionnel) Lancer MLflow UI
 mlflow ui
```

---

## 🖥️ Interface Utilisateur (Streamlit)

L’interface web permet d’utiliser toutes les fonctionnalités de l’API via une page sécurisée.

```bash
pip install streamlit requests
streamlit run ui_streamlit.py
```

- Authentification requise (login : `admin`, mot de passe : `password123` par défaut)
- Boutons pour générer un dataset, prédire, réentraîner, vérifier la santé
- Toutes les actions sont journalisées dans `ui_actions.log`

---

## 🧪 Exemples d’utilisation

- **Générer un dataset**
  ```bash
  curl -X POST "http://127.0.0.1:8000/generate" -H "Content-Type: application/json" -d "{\"n_samples\": 100}"
  ```
- **Prédire**
  ```bash
  curl -X GET "http://127.0.0.1:8000/predict"
  ```
- **Réentraîner**
  ```bash
  curl -X POST "http://127.0.0.1:8000/retrain"
  ```
- **Healthcheck**
  ```bash
  curl -X GET "http://127.0.0.1:8000/health"
  ```

---

## 🧑‍🔬 Tests unitaires

- Placez vos tests dans `tests_unitaires.py`
- Lancez-les avec :
  ```bash
  pytest tests_unitaires.py
  ```

---

## CI/CD

Le projet inclut un pipeline GitHub Actions (`.github/workflows/ci.yml`) qui :
- Installe les dépendances Python
- Lance les tests unitaires
- Vérifie le formatage du code avec `black`
- Construit l’image Docker du projet

Le badge CI en haut du README indique le statut des builds.

---

## Journalisation

- Toutes les actions réalisées via l’interface Streamlit sont enregistrées dans le fichier `ui_actions.log` (connexion, génération, prédiction, etc).
- Les logs techniques de l’API sont gérés par Loguru dans `main.py`.

---

## 📂 Structure du projet

```
main.py
ui_streamlit.py
requirements.txt
tests_unitaires.py
datasets.db
model.pkl
README.md
.github/workflows/ci.yml
```

---

## 👨‍💻 Auteur

- Projet réalisé par [Niels/Alan]

---
