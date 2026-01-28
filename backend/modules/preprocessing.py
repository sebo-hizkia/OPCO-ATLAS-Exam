import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
