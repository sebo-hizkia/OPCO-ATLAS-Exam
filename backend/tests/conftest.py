import pandas as pd
import pytest

@pytest.fixture
def dummy_dataset(tmp_path):
    df = pd.DataFrame({
        "source": ["mat", "por"] * 10,
        "famsize": ["GT3", "LE3"] * 10,
        "studytime": [2, 3] * 10,
        "failures": [0, 1] * 10,
        "activities": ["yes", "no"] * 10,
        "higher": ["yes", "yes"] * 10,
        "internet": ["yes", "no"] * 10,
        "famrel": [4, 3] * 10,
        "freetime": [3, 2] * 10,
        "goout": [2, 4] * 10,
        "absences": [1, 5] * 10,
        "G1": [12, 8] * 10,
        "G2": [11, 7] * 10,
        "G3": [13, 6] * 10,
    })

    path = tmp_path / "dummy_students.csv"
    df.to_csv(path, sep=";", index=False)
    return path
