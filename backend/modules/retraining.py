from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from modules.data_preparation import prepare_dataset
from modules.preprocessing import make_preprocessor


def retrain_model(
    df,
    include_g2: bool,
    model_output_path: Path,
    run_name: str
) -> dict:
    """
    Ré-entraîne un modèle de régression logistique pour un scénario donné
    (avec ou sans G2), avec validation croisée et monitoring MLflow.

    Le modèle est entraîné from scratch afin de garantir cohérence,
    reproductibilité et alignement avec le notebook.
    """

    # ------------------------------------------------------------------
    # Préparation des données (logique métier centralisée)
    # ------------------------------------------------------------------
    X, y = prepare_dataset(df, include_g2=include_g2)

    if len(X) < 5:
        raise ValueError(
            "Nombre d'observations insuffisant pour effectuer un ré-entrainement."
        )

    # ------------------------------------------------------------------
    # Pipeline (préprocessing + modèle)
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Validation croisée (adaptative si petit dataset)
    cv = min(5, len(X))

    # ------------------------------------------------------------------
    # Entraînement + monitoring MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("student-success")

    with mlflow.start_run(run_name=run_name):

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={
                "f1": "f1",
                "recall": "recall"
            },
            return_train_score=False
        )

        # -------------------------
        # Logging paramètres
        # -------------------------
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("include_g2", include_g2)
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])

        # -------------------------
        # Logging métriques
        # -------------------------
        mlflow.log_metric("f1_mean", scores["test_f1"].mean())
        mlflow.log_metric("f1_std", scores["test_f1"].std())
        mlflow.log_metric("recall_mean", scores["test_recall"].mean())
        mlflow.log_metric("recall_std", scores["test_recall"].std())

        # -------------------------
        # Entraînement final
        # -------------------------
        pipeline.fit(X, y)

        # Sauvegarde du modèle
        joblib.dump(pipeline, model_output_path)

        # Enregistrement du modèle dans MLflow
        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

    # ------------------------------------------------------------------
    # Résumé retourné à l'API
    # ------------------------------------------------------------------
    return {
        "scenario": "with_g2" if include_g2 else "without_g2",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "cv_folds": cv,
        "f1_mean": float(scores["test_f1"].mean()),
        "f1_std": float(scores["test_f1"].std()),
        "recall_mean": float(scores["test_recall"].mean()),
        "recall_std": float(scores["test_recall"].std()),
        "model_path": model_output_path.name,
    }
