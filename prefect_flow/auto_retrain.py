from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from api.models.Dataset import Dataset
from sklearn.model_selection import train_test_split
from api.utils import make_prediction, send_discord_embed
import pandas as pd
import os
import pickle
import mlflow
import mlflow.sklearn
from prefect import flow, task
from loguru import logger

Base = declarative_base()
DATABASE_URL = "sqlite:///./api/data/datasets.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@task
def retrain_model():
    PERFORMANCE_THRESHOLD = 0.5


    db = SessionLocal()
    # Récupérer le dernier dataset
    last_dataset = db.query(Dataset).order_by(Dataset.id.desc()).first()
    db.close()
    if not last_dataset:
        return {"error": "Aucun dataset trouvé."}

    # Charger les données en DataFrame
    df = pd.read_json(last_dataset.data)
    X = df[["feature1", "feature2"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = make_prediction(X_test)
    if y_pred.any(): 
        acc = accuracy_score(y_test, y_pred)

        if acc >= PERFORMANCE_THRESHOLD:
            send_discord_embed(message="Le modèle actuel est performant", name="Pas de réentraînement nécessaire", value=f"Accuracy: {acc}")
            return {
                "status": "Pas de réentraînement nécessaire",
                "accuracy": acc,
                "threshold": PERFORMANCE_THRESHOLD
            }

    # Chargement du modèle
    if os.path.exists("api/model.pkl"):
        with open("api/model.pkl", "rb") as f:
            model = pickle.load(f)
        model.fit(X_train, y_train)
        pickle.dump(model, open("api/model.pkl", "wb"))
    else:
        model = LogisticRegression(warm_start=True)
        model.fit(X_train, y_train)
        pickle.dump(model, open("api/model.pkl", "wb"))

    # Log du modèle avec MLflow
    with mlflow.start_run(run_name=f"retrain_dataset_{last_dataset.id}"):
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        mlflow.log_param("dataset_id", last_dataset.id)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
    
    send_discord_embed(message="Le modèle a été réentraîné avec succès", name="Réentraînement réussi", value=f"accuracy: {model.score(X_train, y_train)}"
    )

@flow
def periodic_check():
    retrain_model()
    logger.info("Task completed successfully.")

if __name__ == "__main__":
    periodic_check.serve(
        name="every-30s",
        interval=30
    )
