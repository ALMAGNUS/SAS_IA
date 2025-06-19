from prefect import flow, task
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import requests
import os
from sqlalchemy import create_engine

# Paramètres à adapter
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/TON_TOKEN"
MODEL_PATH = "model.pkl"
DB_PATH = "datasets.db"
THRESHOLD = 0.85

@task
def load_data():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    df = pd.read_sql("SELECT * FROM datasets ORDER BY id DESC LIMIT 1", engine)
    if df.empty:
        raise ValueError("Aucun dataset trouvé.")
    train_df = pd.read_json(df.iloc[0]['data'])
    test_df = pd.read_json(df.iloc[0]['test_data'])
    return train_df, test_df

@task
def train_and_check_drift(train_df, test_df, threshold=THRESHOLD):
    X_train = train_df[["feature1", "feature2"]]
    y_train = train_df["target"]
    X_test = test_df[["feature1", "feature2"]]
    y_test = test_df["target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    pickle.dump(model, open(MODEL_PATH, "wb"))

    # Drift detection
    drifting = acc < threshold
    return acc, drifting

@task
def notify_discord(acc, drifting):
    message = f"Accuracy: {acc:.2f} - {'Drifting detected!' if drifting else 'No drift.'}"
    color = 16711680 if drifting else 65280
    data = {
        "embeds": [{
            "title": "ML Monitoring",
            "description": message,
            "color": color
        }]
    }
    resp = requests.post(DISCORD_WEBHOOK, json=data)
    if resp.status_code not in (200, 204):
        print("Erreur Discord:", resp.text)

@flow
def retrain_flow():
    train_df, test_df = load_data()
    acc, drifting = train_and_check_drift(train_df, test_df)
    notify_discord(acc, drifting)

if __name__ == "__main__":
    retrain_flow() 