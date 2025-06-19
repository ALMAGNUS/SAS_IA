from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn
import os
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from sklearn.model_selection import train_test_split
import numpy as np
import requests

app = FastAPI()
Base = declarative_base()
Instrumentator().instrument(app).expose(app)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(String)  # Stockage sous forme de string (JSON)
    test_data = Column(String)  # Stockage des données de test


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True)
    is_active = Column(Integer, default=1)


DATABASE_URL = "sqlite:///./datasets.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class GenerateRequest(BaseModel):
    n_samples: int = 100


def make_prediction(X_test):
    try:
        # Chargement du modèle
        with open("model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        # Prédiction
        y_pred = loaded_model.predict(X_test)
        return y_pred
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return None

def send_discord_embed(message, status):
    """Envoyer un message à un canal Discord via un Webhook."""
    data = {"embeds": [{
                "title": "Résultats du pipeline",
                "description": message,
                "color": 5814783,
                "fields": status}]}
    response = requests.post("https://discord.com/api/webhooks/1384074986242969661/tr39sXz9sm2Q4ONQuhqYXTptDyb8XQQ_HTy872FNVYHd1lUq3agxSNYPD-bh2-209WI1", json=data)
    if response.status_code != 204:
        print(f"Erreur lors de l'envoi de l'embed : {response.status_code}")
    else:
        print("Embed envoyé avec succès !")

@app.post("/generate")
def generate_dataset(req: GenerateRequest):
    try:
        n = req.n_samples
        hour = datetime.now().hour
        sign = 1 if (hour % 2 == 0) else -1

        # Génération d'un dataset plus complexe
        X, y = make_classification(
            n_samples=n,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=2,  # Plus de clusters
            flip_y=0.1,  # Ajout de bruit
            class_sep=1.0,  # Séparation moins nette
            random_state=None  # Random state variable
        )

        df = pd.DataFrame(X, columns=["feature1", "feature2"])
        df["target"] = y
        logger.info("Dataset generated")

        # Ajout de bruit gaussien
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        df["feature1"] = X[:, 0]
        df["feature2"] = X[:, 1]

        columns_inverted = False
        if hour % 2 == 1:
            df[["feature1", "feature2"]] = df[["feature2", "feature1"]]
            df = df.rename(columns={"feature2": "feature1", "feature1": "feature2"})
            columns_inverted = True
            logger.info("Columns inverted")

        # Split train/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Stockage en base
        db = SessionLocal()
        dataset = Dataset(
            data=train_df.to_json(orient="records"),
            test_data=test_df.to_json(orient="records")
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.close()
        logger.info("Dataset stored in database")

        return {
            "dataset_id": dataset.id,
            "n_samples": n,
            "hour": hour,
            "sign": sign,
            "columns_inverted": columns_inverted,
            "train_size": len(train_df),
            "test_size": len(test_df)
        }
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        return {"error": str(e)}


@app.get("/predict")
def predict():
    try:
        db = SessionLocal()
        last_dataset = db.query(Dataset).order_by(Dataset.id.desc()).first()
        db.close()

        if not last_dataset:
            return {"error": "Aucun dataset trouvé."}

        # Utilisation des données de test
        test_df = pd.read_json(last_dataset.test_data)
        X_test = test_df[["feature1", "feature2"]]
        y_test = test_df["target"]

        y_pred = make_prediction(X_test)
        if y_pred is None:
            return {"error": "Erreur lors de la prédiction"}

        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "prediction": y_pred.tolist(),
            "actual_values": y_test.tolist(),
            "accuracy": accuracy
        }
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return {"error": str(e)}


@app.get("/health")
def health():
    return JSONResponse(content="OK", status_code=200)


@app.post("/retrain")
def retrain_model():
    PERFORMANCE_THRESHOLD = 0.85  # Seuil plus réaliste

    try:
        db = SessionLocal()
        last_dataset = db.query(Dataset).order_by(Dataset.id.desc()).first()
        db.close()

        if not last_dataset:
            return {"error": "Aucun dataset trouvé."}

        # Charger les données d'entraînement
        train_df = pd.read_json(last_dataset.data)
        test_df = pd.read_json(last_dataset.test_data)

        X_train = train_df[["feature1", "feature2"]]
        y_train = train_df["target"]
        X_test = test_df[["feature1", "feature2"]]
        y_test = test_df["target"]

        # Test du modèle actuel
        current_pred = make_prediction(X_test)
        if current_pred is not None:
            current_acc = accuracy_score(y_test, current_pred)
            if current_acc >= PERFORMANCE_THRESHOLD:
                send_discord_embed(message="Le modèle actuel est performant", status={
                    "status": "Pas de réentraînement nécessaire",
                    "accuracy": current_acc,
                    "threshold": PERFORMANCE_THRESHOLD
                }
)
                return {
                    "status": "Pas de réentraînement nécessaire",
                    "accuracy": current_acc,
                    "threshold": PERFORMANCE_THRESHOLD
                }

        # Création d'un nouveau modèle (sans warm_start)
        model = LogisticRegression(
            random_state=None,  # Pour plus de variabilité
            max_iter=1000,      # Plus d'itérations si nécessaire
            C=1.0               # Régularisation standard
        )
        model.fit(X_train, y_train)

        # Évaluation sur les données de test
        test_accuracy = model.score(X_test, y_test)
        train_accuracy = model.score(X_train, y_train)

        # Sauvegarde du modèle
        pickle.dump(model, open("model.pkl", "wb"))

        # Log avec MLflow
        with mlflow.start_run(run_name=f"retrain_dataset_{last_dataset.id}"):
            mlflow.sklearn.log_model(model, "logistic_regression_model")
            mlflow.log_param("dataset_id", last_dataset.id)
            mlflow.log_param("n_samples_train", len(train_df))
            mlflow.log_param("n_samples_test", len(test_df))
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
        send_discord_embed(message="Le modèle a été réentraîné avec succès", status={
            "status": "retrained",
            "dataset_id": last_dataset.id,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        })
        return {
            "status": "retrained",
            "dataset_id": last_dataset.id,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }
    except Exception as e:
        logger.error(f"Erreur lors du réentraînement : {str(e)}")
        return {"error": str(e)} 