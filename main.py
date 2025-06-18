from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import mlflow.sklearn
import os

app = FastAPI()
Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(String)  # Stockage sous forme de string (JSON)

DATABASE_URL = "sqlite:///./datasets.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class GenerateRequest(BaseModel):
    n_samples: int = 100

@app.post("/generate")
def generate_dataset(req: GenerateRequest):
    n = req.n_samples
    hour = datetime.now().hour
    sign = 1 if (hour % 2 == 0) else -1

    X, y = make_classification(
        n_samples=n,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0,
        class_sep=2,
        random_state=42
        )

    df = pd.DataFrame(X, columns=["feature1", "feature2"])
    df["target"] = y

    # Si l'heure est impaire, on inverse les colonnes de la classe (feature1 <-> feature2)
    if hour % 2 == 1:
        df[["feature1", "feature2"]] = df[["feature2", "feature1"]]
        df = df.rename(columns={"feature2": "feature1", "feature1": "feature2"})

    # Conversion du DataFrame en JSON (string)
    json_data = df.to_json(orient="records")

    # Stockage en base
    db = SessionLocal()
    dataset = Dataset(data=json_data)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    db.close()

    return {"dataset_id": dataset.id, "n_samples": n, "hour": hour, "sign": sign}


@app.get("/predict")
def predict():
    db = SessionLocal()
    dataset = db.query(Dataset).order_by(Dataset.id.desc()).first()
    db.close()

    if not dataset:
        return {"error": "No dataset found"}

    # Conversion du JSON en DataFrame
    df = pd.read_json(dataset.data, orient="records")
    df = df[["feature1", "feature2"]]

    # Chargement du modèle
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # Prédiction
    y_pred = loaded_model.predict(df)

    print(f"Prédiction : {y_pred}")

    return y_pred


@app.get("/health")
def health():
    return "OK", 200


@app.post("/retrain")
def retrain_model():
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

    # Chargement du modèle
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        model.fit(X, y)
        pickle.dump(model, open("model.pkl", "wb"))
    else:
        model = LogisticRegression(warm_start=True)
        model.fit(X, y)
        pickle.dump(model, open("model.pkl", "wb"))

    # Log du modèle avec MLflow
    with mlflow.start_run(run_name=f"retrain_dataset_{last_dataset.id}"):
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        mlflow.log_param("dataset_id", last_dataset.id)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metric("train_accuracy", model.score(X, y))

    return {
        "status": "retrained",
        "dataset_id": last_dataset.id,
        "train_accuracy": model.score(X, y)
    }
