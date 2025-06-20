from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.datasets import make_classification
from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from loguru import logger
from sklearn.model_selection import train_test_split
from api.models.Dataset import Dataset
from api.models.Users import Users
from api.utils import make_prediction, create_access_token, verify_password
from datetime import datetime
from datetime import timedelta
from api.data.db_manager import SessionLocal
import os

router = APIRouter()

# Pour l'authentification OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

n_samples = 100


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = db.query(Users).filter(Users.username == form_data.username).first()
    db.close()
    if not user or not (verify_password(form_data.password, user.password_hash)):
        logger.error("Incorrect username or password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")))
    )
    logger.info("Login successful")
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/generate")
def generate_dataset():
    try:
        n = n_samples
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


@router.get("/predict")
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


@router.get("/health")
def health():
    return JSONResponse(content="OK", status_code=200)
