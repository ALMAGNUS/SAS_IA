import os
import pickle
from loguru import logger
import requests

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
