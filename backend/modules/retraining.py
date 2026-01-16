import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# Préprocessing
# -------------------------------------------------------------------

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Crée dynamiquement un préprocesseur sklearn à partir des types de colonnes :
    - standardisation des variables numériques
    - encodage one-hot des variables catégorielles
    """
    cat_features = X.select_dtypes(include="object").columns.tolist()
    num_features = X.select_dtypes(exclude="object").columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


# -------------------------------------------------------------------
# Ré-entrainement
# -------------------------------------------------------------------

def retrain_model(
    csv_path: str,
    include_g2: bool,
    model_output_path: Path
) -> dict:
    """
    Ré-entraîne un pipeline de classification avec monitoring MLflow.

    Args:
        csv_path (str): chemin vers le CSV de données
        include_g2 (bool): inclure ou non la note G2
        model_output_path (Path): chemin de sauvegarde du modèle

    Returns:
        dict: métriques principales du ré-entrainement
    """

    # Chargement des données
    df = pd.read_csv(csv_path, sep=";")

    if "target" not in df.columns:
        raise ValueError("La colonne 'target' est absente du jeu de données.")

    y = df["target"]

    # Sélection des features selon le scénario
    if include_g2:
        X = df.drop(columns=["G3", "target"], errors="ignore")
        run_name = "retrain_with_g2"
    else:
        X = df.drop(columns=["G3", "G2", "target"], errors="ignore")
        run_name = "retrain_without_g2"

    if len(X) < 2:
        raise ValueError("Nombre d'observations insuffisant pour le ré-entrainement.")

    # Création du pipeline complet
    preprocessor = make_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Validation croisée robuste
    cv = min(5, len(X))

    with mlflow.start_run(run_name=run_name):

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1"
        )

        mlflow.log_param("include_g2", include_g2)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", cv)

        mlflow.log_metric("f1_mean", scores.mean())
        mlflow.log_metric("f1_std", scores.std())

        # Entrainement final sur toutes les données
        pipeline.fit(X, y)

        # Sauvegarde du modèle
        joblib.dump(pipeline, model_output_path)

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

    return {
        "f1_mean": float(scores.mean()),
        "f1_std": float(scores.std()),
        "model_path": model_output_path.name,
        "cv_folds": cv
    }
