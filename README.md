# 🎓 Prédiction de la réussite scolaire

Ce projet vise à concevoir une solution d’intelligence artificielle permettant de prédire la réussite scolaire d’un élève à partir de données sociodémographiques, comportementales et académiques, dans un cadre éthique et conforme aux bonnes pratiques de la data science.

Le projet s’appuie sur des jeux de données publics issus d’établissements scolaires portugais et s’inscrit dans le cadre de l’épreuve majeure **Atlas IA – Expert IT**.

---

## 📂 Structure du projet

.
├── data/
│ ├── student-mat.csv
│ └── student-por.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── journal-de-bord.ipynb

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
