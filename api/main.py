from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from sklearn.datasets import make_classification
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger
from sklearn.model_selection import train_test_split
from passlib.context import CryptContext
from jose import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta
from api.models.Dataset import Dataset, Base
from api.utils import make_prediction

app = FastAPI()
Instrumentator().instrument(app).expose(app)


DATABASE_URL = "sqlite:///./api/data/datasets.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class GenerateRequest(BaseModel):
    n_samples: int = 100


# Pour le hash des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pour l'authentification OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Clé secrète et algorithme pour le JWT
SECRET_KEY = "change_this_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = {
            "username":"admin",
            "password": "password123"
            }
    if not user or not (form_data.password == user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/generate")
def generate_dataset(req: GenerateRequest):
    try:
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
            class_sep=2
            )

        df = pd.DataFrame(X, columns=["feature1", "feature2"])
        df["target"] = y
        logger.info("Dataset generated")

        columns_inverted = False
        # Si l'heure est impaire, on inverse les colonnes de la classe (feature1 <-> feature2)
        if hour % 2 == 1:
            df[["feature1", "feature2"]] = df[["feature2", "feature1"]]
            df = df.rename(columns={"feature2": "feature1", "feature1": "feature2"})
            columns_inverted = True
            logger.info("Columns inverted")

        # Conversion du DataFrame en JSON (string)
        json_data = df.to_json(orient="records")

        # Stockage en base
        db = SessionLocal()
        dataset = Dataset(data=json_data)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.close()
        logger.info("Dataset stored in database")

        return {"dataset_id": dataset.id, "n_samples": n, "hour": hour, "sign": sign, "columns_inverted": columns_inverted}
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")


@app.get("/predict")
def predict():
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

    print(f"Prédiction : {y_pred}")

    return {"prediction": y_pred.tolist()}


@app.get("/health")
def health():
    return JSONResponse(content="OK", status_code=200)
