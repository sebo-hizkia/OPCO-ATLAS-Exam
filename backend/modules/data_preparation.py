import pandas as pd
from typing import Tuple

# Liste des features utilisées en production
FEATURES_WITHOUT_G2 = [
    "source",
    "famsize",
    "studytime",
    "failures",
    "activities",
    "higher",
    "internet",
    "famrel",
    "freetime",
    "goout",
    "absences",
    "G1",
]

FEATURES_WITH_G2 = FEATURES_WITHOUT_G2 + ["G2"]


def prepare_dataset(
    df: pd.DataFrame,
    include_g2: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare le dataset pour l'entraînement / ré-entrainement :
    - création de la cible
    - sélection stricte des features autorisées
    """

    expected_features = FEATURES_WITH_G2 if include_g2 else FEATURES_WITHOUT_G2

    # Vérification des colonnes requises
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Création de la cible si absente
    if "target" not in df.columns:
        if "G3" not in df.columns:
            raise ValueError("Colonne 'G3' absente pour créer la cible")
        df = df.copy()
        df["target"] = (df["G3"] >= 10).astype(int)

    X = df[expected_features]
    y = df["target"]

    return X, y
