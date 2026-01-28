# 🎓 Prédiction de la réussite scolaire

Ce projet vise à concevoir une solution d’intelligence artificielle permettant de prédire la réussite scolaire d’un élève à partir de données sociodémographiques, comportementales et académiques, dans un cadre éthique et conforme aux bonnes pratiques de la data science.

Le projet s’appuie sur des jeux de données publics issus d’établissements scolaires portugais et s’inscrit dans le cadre de l’épreuve majeure **Atlas IA – Expert IT**.

---

## 📂 Structure du projet
````
.
├── data/
│ ├── student-mat.csv
│ └── student-por.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
├── journal-de-bord.ipynb
├── docker-compose.yml
│
├── backend/
│   ├── main.py                # API FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── middleware/
│   │   └── audit_middleware.py
│   │
│   ├── models/
│   │   ├── model_with_g2.pkl
│   │   └── model_without_g2.pkl
│   │
│   ├── logs/
│   │   └── app.log
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── data_validation.py
│   │   ├── preprocessing.py
│   │   └── retraining.py      # Retrain
│   │
│   └── tests/                 # Tests
│
│
├── frontend/
│   ├── app.py                 # Interface Streamlit
│   ├── Dockerfile
│   └── requirements.txt
│
├── mlruns/                    # MLflow
│
└── .github/
    └── workflows/
        └── test.yml            # CI (tests automatisés)
````

---

## 📝 Notebook

Le notebook principal contient l’ensemble des étapes du projet :
- chargement et fusion des données
- exploration et visualisation
- préparation des données (cible, scénarios, encodage)
- modélisation et comparaison des performances
- analyse des résultats et conclusions

### 🔧 Pré-requis pour l’exécution avec JupyterLab

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user
jupyter lab
```

Le notebook est conçu pour être exécuté de bout en bout, sans modification, une fois les dépendances installées.

## 📓 Journal de bord

Le journal de bord documente la démarche suivie tout au long du projet :

- choix méthodologiques
- arbitrages techniques et éthiques
- difficultés rencontrées et solutions apportées

Il complète le notebook en apportant une lecture réflexive et professionnelle du travail réalisé.
---

## 🐳 Lancement avec Docker
### 1️⃣ Prérequis

- Docker
- Docker Compose

### 2️⃣ Construction et démarrage

À la racine du projet :
````bash
docker compose up --build
````

Les services sont automatiquement lancés :

- backend (API)
- frontend (interface utilisateur)

---

### 🌐 Accès aux services

| Service               | URL                                                          |
| --------------------- | ------------------------------------------------------------ |
| Interface Streamlit   | [http://localhost:8501](http://localhost:8501)               |
| API FastAPI           | [http://localhost:8000](http://localhost:8000)               |
| Documentation Swagger | [http://localhost:8000/docs](http://localhost:8000/docs)     |
| Healthcheck           | [http://localhost:8000/health](http://localhost:8000/health) |

---

## 🔌 API — Routes disponibles

---

### 🔹 Healthcheck

```http
GET /health
```

**Réponse :**

```json
{
  "status": "ok"
}
```

---

### 🔹 Prédiction sans G2 (prédiction précoce)

```http
POST /predict-without-g2
```

**Payload attendu :**

```json
{
  "source": "mat",
  "famsize": "GT3",
  "studytime": 2,
  "failures": 0,
  "activities": "yes",
  "higher": "yes",
  "internet": "yes",
  "famrel": 4,
  "freetime": 3,
  "goout": 2,
  "absences": 3,
  "G1": 12
}
```

---

### 🔹 Prédiction avec G2 (prédiction complète)

```http
POST /predict-with-g2
```

**Payload attendu :**

```json
{
  "source": "mat",
  "famsize": "GT3",
  "studytime": 2,
  "failures": 0,
  "activities": "yes",
  "higher": "yes",
  "internet": "yes",
  "famrel": 4,
  "freetime": 3,
  "goout": 2,
  "absences": 3,
  "G1": 12,
  "G2": 13
}
```

---

### 🔹 Réponse type (prédiction)

```json
{
  "prediction": 1,
  "mode": "with_g2",
  "interpretation": "Réussite probable"
}
```

---

## 🔁 Ré-entraînement des modèles (monitoré avec MLflow)

L’API permet de **ré-entraîner automatiquement les modèles à partir d’un nouveau fichier CSV**.

* Le modèle **sans G2** est toujours entraîné (prédiction précoce)
* Le modèle **avec G2** est entraîné uniquement si la colonne `G2` est présente
* Les métriques **F1-score** et **Recall** sont évaluées par validation croisée et loggées dans **MLflow**

Lancement de MLFlow

```
mlflow ui
```

Puis http://127.0.0.1:5000

---

### 🔹 Ré-entrainement via API

```http
POST /retrain
```

**Form-data attendu :**

* `file` : fichier CSV (`;` comme séparateur)

---

### 📌 Exemple avec `curl`

```bash
curl -X POST http://localhost:8000/retrain \
  -F "file=@student-mat.csv"
```

---

### 🔹 Réponse type

```json
{
  "status": "success",
  "models_trained": [
    "without_g2",
    "with_g2"
  ],
  "results": {
    "without_g2": {
      "f1_mean": 0.91,
      "recall_mean": 0.94,
      "cv_folds": 5,
      "model_path": "model_without_g2.pkl"
    },
    "with_g2": {
      "f1_mean": 0.94,
      "recall_mean": 0.95,
      "cv_folds": 5,
      "model_path": "model_with_g2.pkl"
    }
  }
}
```

---

### Journalisation des requêtes

#### Visualisation des logs en temps réel
````
docker compose logs -f backend
````

#### Accéder au fichier app.log dans le conteneur

````
docker exec -it backend /bin/bash
cd /app/logs
````

### Ré-entrainement

Lancement des tests
````
pytest backend/tests -q
````
