from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
import sys
from middleware.audit_middleware import audit_requests
from modules.retraining import retrain_model

from fastapi import UploadFile, File, Form
import tempfile
import shutil

# -------------------------------------------------------------------
# Configuration des chemins
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

MODEL_WITH_G2_PATH = MODELS_DIR / "model_with_g2.pkl"
MODEL_WITHOUT_G2_PATH = MODELS_DIR / "model_without_g2.pkl"

# -------------------------------------------------------------------
# Configuration Loguru
# -------------------------------------------------------------------

logger.remove()  # supprime la config par défaut

# Console (stdout → Docker)
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "<cyan>{module}</cyan> | {message}"
)

# Fichier avec rotation
logger.add(
    LOGS_DIR / "app.log",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
)

# -------------------------------------------------------------------
# Chargement des modèles (une seule fois au démarrage)
# -------------------------------------------------------------------

logger.info("Chargement des modèles...")
model_with_g2 = joblib.load(MODEL_WITH_G2_PATH)
model_without_g2 = joblib.load(MODEL_WITHOUT_G2_PATH)
logger.info("Modèles chargés avec succès")

# -------------------------------------------------------------------
# Initialisation FastAPI
# -------------------------------------------------------------------

app = FastAPI(
    title="API Prédiction Réussite Scolaire",
    description="API de prédiction avec ou sans note du second trimestre (G2)",
    version="1.0.0"
)

# Journalisation des requêtes HTTP
app.middleware("http")(audit_requests)

# -------------------------------------------------------------------
# Schémas d'entrée
# -------------------------------------------------------------------

class StudentInputWithoutG2(BaseModel):
    source: str
    famsize: str
    studytime: int
    failures: int
    activities: str
    higher: str
    internet: str
    famrel: int
    freetime: int
    goout: int
    absences: int
    G1: int


class StudentInputWithG2(StudentInputWithoutG2):
    G2: int

class RetrainRequest(BaseModel):
    csv_path: str
    include_g2: bool = True

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
def root():
    logger.info("Route / appelée")
    return {"message": "API OK"}


@app.get("/health")
def health():
    logger.debug("Healthcheck OK")
    return {"status": "ok"}


@app.post("/predict-with-g2")
def predict_with_g2(student: StudentInputWithG2):
    """
    Prédiction avec la note du second trimestre (G2).
    Prédiction plus fiable, information disponible plus tardivement.
    """
    logger.info(f"Prediction request (with G2): {student.dict()}")

    df = pd.DataFrame([student.dict()])
    prediction = model_with_g2.predict(df)[0]

    logger.info(f"Prediction result (with G2): {prediction}")

    return {
        "prediction": int(prediction),
        "mode": "with_g2",
        "interpretation": "Réussite probable" if prediction == 1 else "Risque d’échec"
    }


@app.post("/predict-without-g2")
def predict_without_g2(student: StudentInputWithoutG2):
    """
    Prédiction sans la note du second trimestre (G2).
    Prédiction plus précoce, niveau d'incertitude plus élevé.
    """
    logger.info(f"Prediction request (without G2): {student.dict()}")

    df = pd.DataFrame([student.dict()])
    prediction = model_without_g2.predict(df)[0]

    logger.info(f"Prediction result (without G2): {prediction}")

    return {
        "prediction": int(prediction),
        "mode": "without_g2",
        "interpretation": "Réussite probable" if prediction == 1 else "Risque d’échec"
    }

@app.post("/retrain")
def retrain(
    file: UploadFile = File(...),
    include_g2: bool = Form(...)
):
    """
    Ré-entrainement du modèle à partir d'un CSV uploadé
    """
    logger.info(
        f"Retraining request | file={file.filename} | include_g2={include_g2}"
    )

    # Sauvegarde temporaire du fichier
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        csv_path = tmp.name

    if include_g2:
        model_output_path = MODELS_DIR / "model_with_g2.pkl"
    else:
        model_output_path = MODELS_DIR / "model_without_g2.pkl"

    results = retrain_model(
        csv_path=csv_path,
        include_g2=include_g2,
        model_output_path=model_output_path
    )

    return {
        "status": "success",
        "metrics": results,
        "model_path": str(model_output_path)
    }



