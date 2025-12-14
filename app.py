import streamlit as st
import requests
import time
import pandas as pd
from pymongo import MongoClient

FASTAPI_URL = "https://strifeful-nonincandescently-lisbeth.ngrok-free.dev/predict"  # ← remplace ici

st.title("🔮 Dashboard BiLSTM")

# Map class index to human-readable labels
LABELS = {
    2: "Positif",
    0: "Negatif",
    1: "Neutre",
}

text = st.text_area("Entre un texte")

if st.button("Prédire"):
    # try with a small retry/backoff in case the remote connection is flaky
    res = None
    last_exc = None
    for attempt in range(2):
        try:
            r = requests.post(FASTAPI_URL, json={"text": text}, timeout=10)
            r.raise_for_status()
            res = r.json()
            last_exc = None
            break
        except requests.exceptions.RequestException as e:
            last_exc = e
            # small backoff before retry
            time.sleep(1 * (attempt + 1))

    if last_exc is not None:
        st.error(
            (
                "Erreur de connexion à l'API de prédiction :\n"
                f"{last_exc}\n\n"
                "Vérifiez que le service FastAPI est en marche, que l'URL `FASTAPI_URL` est correcte, "
                "et que votre tunnel (ngrok) est actif."
            )
        )
        st.stop()
    # normalize index to int (in case API returns string)
    try:
        idx = int(res.get('predicted_class_index', res.get('predicted_class')))
    except Exception:
        idx = res.get('predicted_class')

    label = LABELS.get(idx, str(res.get('predicted_class')))

    st.success(
        f"Prédiction : {label} "
        f"(classe {res.get('predicted_class_index')}) "
        f"avec probabilités : {res['probabilities']}"
    )


# ===== Dashboard =====
MONGO_URI = "mongodb+srv://bilstmm_db:DoudouM@cluster0.x6hswmk.mongodb.net/?appName=Cluster0&tlsAllowInvalidCertificates=true"
client = MongoClient(MONGO_URI) # Re-instantiate MongoClient within Streamlit app
db = client["bilstmm_db"]
collection = db["predictions"]

data = list(collection.find({}, {"_id": 0}))

if data:
    df = pd.DataFrame(data)
    # add human-readable label column when possible
    if 'predicted_class_index' in df.columns:
        try:
            df['predicted_label'] = df['predicted_class_index'].astype(int).map(LABELS)
        except Exception:
            df['predicted_label'] = df['predicted_class_index'].map(lambda x: LABELS.get(x, x))
    st.dataframe(df)
    # Correcting column name for bar chart
    if 'predicted_class_index' in df.columns:
        st.bar_chart(df["predicted_class_index"].value_counts())
    else:
        st.warning("Column 'predicted_class_index' not found in the data for charting.")
else:
    st.info("Aucune donnée pour le moment.")