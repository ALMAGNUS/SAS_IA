import streamlit as st
import requests

API_URL = "http://localhost:8000"

def get_token(username, password):
    response = requests.post(
        f"{API_URL}/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        return None

def main():
    st.title("Interface API sécurisée (JWT)")
    if "token" not in st.session_state:
        st.session_state["token"] = None

    if st.session_state["token"] is None:
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            token = get_token(username, password)
            if token:
                st.session_state["token"] = token
                st.success("Connecté !")
            else:
                st.error("Identifiants invalides ou accès refusé.")
    else:
        st.success("Connecté à l'API !")
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}

        st.subheader("Actions API protégées")

        if st.button("Générer un dataset"):
            n_samples = st.number_input("Nombre d'échantillons", min_value=10, max_value=1000, value=100)
            resp = requests.post(f"{API_URL}/generate", json={"n_samples": int(n_samples)}, headers=headers)
            try:
                st.write(resp.json())
            except Exception:
                st.error("Erreur lors de l'appel à /generate")

        if st.button("Prédire avec le dernier dataset"):
            resp = requests.get(f"{API_URL}/predict", headers=headers)
            try:
                st.write(resp.json())
            except Exception:
                st.error("Erreur lors de l'appel à /predict")

        if st.button("Réentraîner le modèle"):
            resp = requests.post(f"{API_URL}/retrain", headers=headers)
            try:
                st.write(resp.json())
            except Exception:
                st.error("Erreur lors de l'appel à /retrain")

        if st.button("Vérifier la santé de l'API"):
            resp = requests.get(f"{API_URL}/health", headers=headers)
            try:
                st.write(resp.text)
            except Exception:
                st.error("Erreur lors de l'appel à /health")

        if st.button("Tester la route protégée"):
            resp = requests.get(f"{API_URL}/protected", headers=headers)
            try:
                st.write(resp.json())
            except Exception:
                st.error("Erreur lors de l'appel à /protected")

        if st.button("Se déconnecter"):
            st.session_state["token"] = None
            st.experimental_rerun()

if __name__ == "__main__":
    main() 