import streamlit as st
import requests

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BACKEND_URL = "http://backend:8000"

st.set_page_config(
    page_title="Prédiction de la réussite scolaire",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------------------------------------------
# Titre & description
# -------------------------------------------------------------------

st.title("🎓 Prédiction de la réussite scolaire")

st.markdown(
    """
    Cette application permet d’estimer la probabilité de réussite scolaire
    d’un élève à partir des informations disponibles.

    Deux modes sont proposés :
    - **Prédiction précoce** (sans la note du second trimestre – G2)
    - **Prédiction complète** (avec la note du second trimestre – G2)

    ℹ️ L’outil constitue une **aide à la décision** et ne doit pas être
    utilisé comme un outil de sanction.
    """
)

# -------------------------------------------------------------------
# Choix du mode
# -------------------------------------------------------------------

mode = st.radio(
    "🧩 Mode de prédiction",
    (
        "Prédiction précoce (sans G2)",
        "Prédiction complète (avec G2)"
    )
)

st.divider()

# -------------------------------------------------------------------
# Saisie des données
# -------------------------------------------------------------------

st.subheader("📋 Informations de l’élève")

source = st.selectbox(
    "Cursus",
    ("mat", "por")
)

famsize = st.selectbox(
    "Taille de la famille",
    ("LE3", "GT3")
)

studytime = st.slider(
    "Temps d’étude hebdomadaire",
    min_value=1,
    max_value=4,
    value=2,
    help="1 = <2h, 4 = >10h"
)

failures = st.number_input(
    "Nombre d’échecs scolaires passés",
    min_value=0,
    max_value=4,
    value=0
)

activities = st.selectbox(
    "Activités extrascolaires",
    ("yes", "no")
)

higher = st.selectbox(
    "Souhaite poursuivre des études supérieures",
    ("yes", "no")
)

internet = st.selectbox(
    "Accès à internet à la maison",
    ("yes", "no")
)

famrel = st.slider(
    "Qualité des relations familiales",
    min_value=1,
    max_value=5,
    value=3
)

freetime = st.slider(
    "Temps libre après l’école",
    min_value=1,
    max_value=5,
    value=3
)

goout = st.slider(
    "Fréquence des sorties avec des amis",
    min_value=1,
    max_value=5,
    value=3
)

absences = st.number_input(
    "Nombre d’absences",
    min_value=0,
    max_value=100,
    value=0
)

G1 = st.number_input(
    "Note du 1er trimestre (G1)",
    min_value=0,
    max_value=20,
    value=10
)

# G2 uniquement si mode complet
if mode == "Prédiction complète (avec G2)":
    G2 = st.number_input(
        "Note du 2ᵉ trimestre (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

st.divider()

# -------------------------------------------------------------------
# Prédiction
# -------------------------------------------------------------------

if st.button("🔮 Lancer la prédiction"):
    payload = {
        "source": source,
        "famsize": famsize,
        "studytime": studytime,
        "failures": failures,
        "activities": activities,
        "higher": higher,
        "internet": internet,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "absences": absences,
        "G1": G1
    }

    if mode == "Prédiction complète (avec G2)":
        payload["G2"] = G2
        endpoint = "/predict-with-g2"
    else:
        endpoint = "/predict-without-g2"

    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Prédiction réalisée avec succès ✅")

            if result["prediction"] == 1:
                st.markdown("### 🟢 Réussite probable")
            else:
                st.markdown("### 🔴 Risque d’échec")

            st.markdown(
                f"""
                **Mode utilisé :** {result['mode']}
                **Interprétation :** {result['interpretation']}
                """
            )

            if mode == "Prédiction précoce (sans G2)":
                st.info(
                    "ℹ️ Cette prédiction est basée sur un niveau "
                    "d’information limité et doit être interprétée "
                    "avec prudence."
                )

        else:
            st.error("Erreur lors de la prédiction")

    except requests.exceptions.RequestException as e:
        st.error("Impossible de contacter l’API backend")
        st.text(str(e))

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.divider()
st.caption(
    "Projet Atlas IA – Expert IT | "
    "Application d’aide à la décision – usage non punitif"
)
