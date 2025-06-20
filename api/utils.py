import os
import pickle
from loguru import logger
import requests
from passlib.context import CryptContext
from datetime import datetime
from jose import jwt
from datetime import timedelta
import os

# Pour le hash des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def make_prediction(X_test):

    # Chargement du modèle
    if os.path.exists("api/model.pkl"):
        with open("api/model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

    # Prédiction
        y_pred = loaded_model.predict(X_test)

        return y_pred
    else:
        logger.error("Model not found.")
        return None
    
def send_discord_embed(message, name, value):
    """Envoyer un message à un canal Discord via un Webhook."""
    data = {"embeds": [{
                "title": "Résultats du pipeline",
                "description": message,
                "color": 5814783,
                "fields": [{
                        "name": name,
                        "value": value,
                        "inline": True
                    }]}]}
    response = requests.post("https://discord.com/api/webhooks/1384074986242969661/tr39sXz9sm2Q4ONQuhqYXTptDyb8XQQ_HTy872FNVYHd1lUq3agxSNYPD-bh2-209WI1", json=data)
    if response.status_code != 204:
        print(f"Erreur lors de l'envoi de l'embed : {response.status_code}")
    else:
        print("Embed envoyé avec succès !")


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
    encoded_jwt = jwt.encode(to_encode, os.environ.get("SECRET_KEY"), algorithm=os.environ.get("ALGORITHM"))
    return encoded_jwt

