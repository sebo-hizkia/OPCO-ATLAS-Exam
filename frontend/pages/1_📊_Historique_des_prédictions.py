import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Historique des prédictions", layout="wide")
st.title("📊 Historique des prédictions")

LOG_FILE = Path("/app/logs/predictions.jsonl")

st.markdown(
    """
    Cette page affiche l’historique des prédictions réalisées par l’API.
    Les données affichées sont limitées aux métadonnées afin de respecter
    les principes de confidentialité.
    """
)

if not LOG_FILE.exists():
    st.warning("Aucun fichier de log trouvé.")
    st.stop()

records = []

with open(LOG_FILE, "r") as f:
    for line in f:
        try:
            log = json.loads(line)

            extra = log.get("record", {}).get("extra", {})
            if extra.get("event") == "prediction":
                records.append({
                    "Date": extra.get("timestamp"),
                    "Session ID": extra.get("session_id"),
                    "Endpoint": extra.get("endpoint"),
                    "Modèle": extra.get("model"),
                    "Prédiction": extra.get("prediction"),
                })

        except json.JSONDecodeError:
            continue



if not records:
    st.info("Aucune prédiction enregistrée pour le moment.")
    st.stop()

df = pd.DataFrame(records)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date", ascending=False)

# Filtres simples
col1, col2 = st.columns(2)
with col1:
    model_filter = st.selectbox(
        "Filtrer par modèle",
        ["Tous"] + sorted(df["Modèle"].dropna().unique().tolist())
    )

with col2:
    limit = st.slider("Nombre de lignes", 5, 100, 20)

if model_filter != "Tous":
    df = df[df["Modèle"] == model_filter]

st.dataframe(df.head(limit), use_container_width=True)
