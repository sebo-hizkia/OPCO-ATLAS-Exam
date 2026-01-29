from fastapi import FastAPI, HTTPException, Request
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
import uuid
from datetime import datetime

# -------------------------------------------------------------------
# Configuration des chemins
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
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

# Fichier log général (log des requêtes HTTP)
logger.add(
    LOGS_DIR / "app.log",
    level="INFO",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
)

# Fichier dédié pour les logs de prédiction
PREDICTION_LOG = Path("/app/logs/predictions.jsonl")

logger.add(
    PREDICTION_LOG,
    level="INFO",
    serialize=True,   # Pour générer sous format JSON
    filter=lambda record: record["extra"].get("event") == "prediction"
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
# Méthodes
# -------------------------------------------------------------------

def run_prediction(
    *,
    request: Request,
    model,
    student_dict: dict,
    model_name: str
):
    df = pd.DataFrame([student_dict])
    prediction = model.predict(df)[0]

    logger.bind(
        event="prediction",
        timestamp=datetime.utcnow().isoformat(),
        session_id=request.headers.get("X-Session-ID"),
        endpoint=request.url.path,
        model=model_name,
        prediction=int(prediction)
    ).info("prediction")

    return {
        "prediction": int(prediction),
        "mode": model_name,
        "interpretation": "Réussite probable" if prediction == 1 else "Risque d’échec"
    }


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
def predict_with_g2(
    student: StudentInputWithG2,
    request: Request
):
    return run_prediction(
        request=request,
        model=model_with_g2,
        student_dict=student.dict(),
        model_name="with_g2"
    )

@app.post("/predict-without-g2")
def predict_without_g2(
    student: StudentInputWithoutG2,
    request: Request
):
    return run_prediction(
        request=request,
        model=model_without_g2,
        student_dict=student.dict(),
        model_name="without_g2"
    )

@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """
    Ré-entraîne automatiquement les modèles de prédiction à partir d'un CSV :
    - modèle sans G2 (prédiction précoce)
    - modèle avec G2 (si disponible dans le fichier)
    """

    # ------------------------------------------------------------------
    # Sauvegarde temporaire du fichier CSV
    # ------------------------------------------------------------------
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV.")

    tmp_filename = f"{uuid.uuid4()}_{file.filename}"
    tmp_csv_path = DATA_DIR / tmp_filename

    try:
        with open(tmp_csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(tmp_csv_path, sep=";")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture CSV : {e}")

    # ------------------------------------------------------------------
    # Ré-entrainement des modèles
    # ------------------------------------------------------------------
    results = {}

    try:
        # --------- Modèle SANS G2 (toujours entraîné)
        results["without_g2"] = retrain_model(
            df=df,
            include_g2=False,
            model_output_path=MODELS_DIR / "model_without_g2.pkl",
            run_name="retrain_without_g2"
        )

        # --------- Modèle AVEC G2 (si disponible)
        if "G2" in df.columns:
            results["with_g2"] = retrain_model(
                df=df,
                include_g2=True,
                model_output_path=MODELS_DIR / "model_with_g2.pkl",
                run_name="retrain_with_g2"
            )
        else:
            results["with_g2"] = {
                "status": "skipped",
                "reason": "Colonne G2 absente du fichier CSV"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Nettoyage du fichier temporaire
        if tmp_csv_path.exists():
            tmp_csv_path.unlink()

    # ------------------------------------------------------------------
    # Réponse API
    # ------------------------------------------------------------------
    return {
        "status": "success",
        "models_trained": list(results.keys()),
        "results": results
    }




