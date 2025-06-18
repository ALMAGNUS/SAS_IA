# SAS_IA
# 🚀 Générateur & Prédicteur de Dataset Linéaire

---

## 📝 Description

API moderne pour :
- Générer des datasets linéaires à 2 features (dont une change de signe selon l’heure)
- Stocker chaque dataset en base SQLite avec identifiant unique
- Prédire la classe (0/1) via régression logistique scikit-learn
- Réentraîner le modèle à chaud, suivi des expériences avec MLflow
- Vérifier la santé de l’API

---

**Légende :**
- L’utilisateur interagit avec l’API FastAPI.
- FastAPI gère la génération, la prédiction et le réentraînement.
- Les datasets sont stockés dans SQLite.
- Les modèles sont entraînés avec scikit-learn.
- Les expériences et modèles sont suivis avec MLflow.

---

## ⚡ Fonctionnalités

- **POST /generate** : Génère et stocke un dataset (features, target, inversion selon l’heure)
- **GET /predict** : Prédit la classe sur le dernier dataset généré
- **POST /retrain** : Réentraîne le modèle sur le dernier dataset, logue dans MLflow
- **GET /health** : Vérifie la disponibilité de l’API

---

## 🚀 Installation rapide

```bash
git clone <url_du_repo>
cd <nom_du_repo>
pip install fastapi uvicorn sqlalchemy pandas scikit-learn mlflow
mlflow ui  
uvicorn main:app --reload
```

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

## 📂 Structure du projet

```
main.py
tests_unitaires.py
datasets.db
README.md
```

---

## 👨‍💻 Auteur

- Projet réalisé par [Niels/Alan]

---