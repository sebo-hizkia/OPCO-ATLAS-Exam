import streamlit as st
import requests

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="Ré-entrainement", layout="centered")
st.title("🔄 Ré-entrainement des modèles")

st.markdown(
    """
    Cette page permet de **ré-entraîner automatiquement les modèles**
    à partir d’un fichier CSV fourni par l’utilisateur.

    Le backend :
    - entraîne **un modèle sans G2**
    - entraîne **un modèle avec G2** si la colonne est présente
    - journalise les métriques (**F1-score, Recall**) dans **MLflow**
    """
)

# Upload CSV
uploaded_file = st.file_uploader(
    "📂 Charger un fichier CSV (`;` comme séparateur)",
    type=["csv"]
)

if uploaded_file and st.button("🚀 Lancer le ré-entrainement"):
    with st.spinner("Ré-entrainement en cours..."):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
        }

        try:
            response = requests.post(
                f"{BACKEND_URL}/retrain",
                files=files,
                timeout=300
            )

            if response.status_code == 200:
                res = response.json()

                st.success("✅ Ré-entrainement terminé")

                # Résumé clair des résultats
                results = res.get("results", {})

                for model_name, metrics in results.items():
                    st.subheader(f"📦 {model_name}")

                    if metrics.get("status") == "skipped":
                        st.warning(metrics.get("reason"))
                        continue

                    st.metric("F1-score moyen", f"{metrics['f1_mean']:.3f}")
                    st.metric("Recall moyen", f"{metrics['recall_mean']:.3f}")
                    st.caption(f"Modèle sauvegardé : `{metrics['model_path']}`")

            else:
                st.error(f"Erreur backend ({response.status_code})")
                st.text(response.text)

        except Exception as e:
            st.error(f"Erreur backend : {e}")
