from pathlib import Path
from modules.retraining import retrain_model


def test_retrain_model_without_g2(dummy_dataset, tmp_path):
    model_path = tmp_path / "model.pkl"

    results = retrain_model(
        csv_path=str(dummy_dataset),
        include_g2=False,
        model_output_path=model_path
    )

    # Vérifications essentielles
    assert "f1_mean" in results
    assert "f1_std" in results
    assert model_path.exists()
    assert 0.0 <= results["f1_mean"] <= 1.0
