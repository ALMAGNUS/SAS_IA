import streamlit as st
import requests
from loguru import logger
from datetime import datetime



API_URL = "http://localhost:8000"  # À adapter si besoin

# Utilisateurs autorisés (à améliorer pour production)
USERS = {"admin": "password123"}

def log_action(user, action):
    logger.info(f"User: {user} - Action: {action}")

def login():
    st.title("Authentification")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            log_action(username, "Connexion réussie")
            st.success("Connecté !")
        else:
            log_action(username, "Échec de connexion")
            st.error("Identifiants invalides")

def main_app():
    st.title("Interface de gestion API IA")
    st.write(f"Connecté en tant que : {st.session_state['username']}")
    
    if st.button("Générer un dataset"):
        n_samples = st.number_input("Nombre d'échantillons", min_value=10, max_value=1000, value=100)
        resp = requests.post(f"{API_URL}/generate", json={"n_samples": int(n_samples)})
        st.write(resp.json())
        log_action(st.session_state['username'], f"POST /generate n_samples={n_samples}")

    if st.button("Prédire avec le dernier dataset"):
        resp = requests.get(f"{API_URL}/predict")
        st.write(resp.json())
        log_action(st.session_state['username'], "GET /predict")

    if st.button("Réentraîner le modèle"):
        resp = requests.post(f"{API_URL}/retrain")
        st.write(resp.json())
        log_action(st.session_state['username'], "POST /retrain")

    if st.button("Vérifier la santé de l'API"):
        resp = requests.get(f"{API_URL}/health")
        st.write(resp.text)
        log_action(st.session_state['username'], "GET /health")

    if st.button("Se déconnecter"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.experimental_rerun()

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
    if not st.session_state["authenticated"]:
        login()
    else:
        main_app()

if __name__ == "__main__":
    main()
