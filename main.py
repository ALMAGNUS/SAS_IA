from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from datetime import datetime

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

    # Génération des features
    feature1 = np.random.rand(n)
    feature2 = np.random.rand(n) * sign

    # Génération de la cible binaire (par exemple, selon la somme des features)
    target = ((feature1 + feature2) > 1).astype(int)

    # Préparation des données à stocker
    data = [
        {"feature1": float(f1), "feature2": float(f2), "target": int(t)}
        for f1, f2, t in zip(feature1, feature2, target)
    ]

    # Stockage en base
    db = SessionLocal()
    dataset = Dataset(data=str(data))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    db.close()

    return {"dataset_id": dataset.id, "n_samples": n, "hour": hour, "sign": sign}

