import streamlit as st
import requests

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="Ré-entrainement", layout="centered")
st.title("🔄 Ré-entrainement du modèle")

st.markdown(
    """
    Cette page permet de ré-entrainer le modèle à partir
    d'un fichier CSV fourni par l'utilisateur.
    """
)

# Upload CSV
uploaded_file = st.file_uploader(
    "📂 Charger un fichier CSV",
    type=["csv"]
)

include_g2 = st.checkbox(
    "Inclure la note du second trimestre (G2)",
    value=True
)

if uploaded_file and st.button("🚀 Lancer le ré-entrainement"):
    with st.spinner("Ré-entrainement en cours..."):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
        }

        data = {
            "include_g2": str(include_g2).lower()
        }

        try:
            response = requests.post(
                f"{BACKEND_URL}/retrain",
                files=files,
                data=data,
                timeout=300
            )

            if response.status_code == 200:
                res = response.json()

                st.success("Ré-entrainement terminé 🎉")
                st.metric("F1-score moyen", f"{res['f1_mean']:.3f}")
                st.metric("Écart-type", f"{res['f1_std']:.3f}")
                st.write("Modèle sauvegardé :", res["model_path"])

            else:
                st.error(response.text)

        except Exception as e:
            st.error(f"Erreur backend : {e}")
