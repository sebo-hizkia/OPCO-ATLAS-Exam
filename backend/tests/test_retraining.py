from pathlib import Path
import pandas as pd
from modules.retraining import retrain_model


def test_retrain_model_without_g2(dummy_dataset, tmp_path):
    # dummy_dataset est un Path vers un CSV
    df = pd.read_csv(dummy_dataset, sep=";")

    model_path = tmp_path / "model_without_g2.pkl"

    results = retrain_model(
        df=df,
        include_g2=False,
        model_output_path=model_path,
        run_name="test_without_g2"
    )

    # Vérifications essentielles
    assert "f1_mean" in results
    assert "f1_std" in results
    assert "recall_mean" in results

    assert model_path.exists()
    assert 0.0 <= results["f1_mean"] <= 1.0
    assert 0.0 <= results["recall_mean"] <= 1.0
