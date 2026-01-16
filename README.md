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
│   ├── models/
│   │   ├── model_with_g2.pkl
│   │   └── model_without_g2.pkl
│   ├── logs/
│   │   └── app.log
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   └── retraining.py        # Retrain
│   │
│   └── mlruns/                  # MLflow
│
│
├── frontend/
│   ├── app.py                 # Interface Streamlit
│   ├── Dockerfile
│   └── requirements.txt
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
docker-compose up --build
````

Les services sont automatiquement lancés :

- backend (API)
- frontend (interface utilisateur)

### 🌐 Accès aux services

| Service               | URL                                                          |
| --------------------- | ------------------------------------------------------------ |
| Interface Streamlit   | [http://localhost:8501](http://localhost:8501)               |
| API FastAPI           | [http://localhost:8000](http://localhost:8000)               |
| Documentation Swagger | [http://localhost:8000/docs](http://localhost:8000/docs)     |
| Healthcheck           | [http://localhost:8000/health](http://localhost:8000/health) |

🔌 API — Routes disponibles
🔹 Healthcheck
````
GET /health
````

Réponse :
````
{
  "status": "ok"
}
````

🔹 Prédiction sans G2 (précoce)
````
POST /predict-without-g2
````

Payload attendu :
````
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
````
🔹 Prédiction avec G2 (complète)
````
POST /predict-with-g2
````

Payload attendu :
````
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
````
🔹 Réponse type
````
{
  "prediction": 1,
  "mode": "with_g2",
  "interpretation": "Réussite probable"
}
````
### Journalisation des requêtes

#### Visualisation des logs en temps réel
````
docker-compose logs -f backend
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
