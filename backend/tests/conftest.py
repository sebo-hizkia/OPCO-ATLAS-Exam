import pandas as pd
import pytest
from pathlib import Path
import numpy as np


@pytest.fixture
def dummy_dataset(tmp_path: Path):
    """
    Génère un jeu de données fictif mais réaliste
    avec 20 observations, suffisant pour une
    validation croisée à 5 folds.
    """

    np.random.seed(42)

    n_samples = 20

    df = pd.DataFrame({
        "studytime": np.random.randint(1, 5, size=n_samples),
        "failures": np.random.randint(0, 3, size=n_samples),
        "absences": np.random.randint(0, 10, size=n_samples),
        "G1": np.random.randint(0, 20, size=n_samples),
        "G2": np.random.randint(0, 20, size=n_samples),
        "source": np.random.choice(["mat", "por"], size=n_samples),
        "target": np.random.choice([0, 1], size=n_samples)
    })

    csv_path = tmp_path / "dummy_students.csv"
    df.to_csv(csv_path, sep=";", index=False)

    return csv_path
