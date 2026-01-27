import pandas as pd
from typing import Tuple

# Variables sensibles (copie exacte du notebook)
VARIABLES_SENSIBLES = [
    "sex", "age", "address", "Pstatus", "Mjob", "Fjob", "guardian",
    "romantic", "health", "Dalc", "Walc", "school", "reason",
    "nursery", "traveltime", "schoolsup", "famsup",
    "Medu", "Fedu", "famsize", "internet", "paid"
]

def prepare_dataset(
    df: pd.DataFrame,
    include_g2: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare le dataset pour l'entraînement ou le ré-entrainement :
    - création de la cible
    - suppression des variables sensibles
    - sélection des features selon le scénario
    """

    # Création de la cible si absente
    if "target" not in df.columns:
        df = df.copy()
        df["target"] = (df["G3"] >= 10).astype(int)

    y = df["target"]

    # Colonnes à retirer
    columns_to_drop = ["G3", "target"] + VARIABLES_SENSIBLES

    if not include_g2:
        columns_to_drop.append("G2")

    X = df.drop(columns=columns_to_drop, errors="ignore")

    return X, y
