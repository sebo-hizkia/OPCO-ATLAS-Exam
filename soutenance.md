---
marp: true
theme: default
paginate: true
header: Prédiction de la réussite scolaire
footer: Projet IA – OPCO Atlas
---

# 🎓 Prédiction de la réussite scolaire
## Projet IA – OPCO Atlas

**Nom :** Sébastien Andres
**Certification :** Expert IT – Intelligence Artificielle
**Objectif :** Concevoir, évaluer et industrialiser une solution IA responsable

---

# 🧩 Contexte & Problématique

- Données issues d’un contexte éducatif
- Objectif : **anticiper le risque d’échec scolaire**
- Aide à la décision
- Enjeux :
  - performance prédictive
  - biais socio-économiques
  - responsabilité éthique

---

# 🎯 Objectifs du projet

- Analyser les données élèves
- Identifier les **variables sensibles**
- Tester plusieurs **scénarios de modélisation**
- Comparer les modèles
- Déployer une application fonctionnelle
- Garantir la traçabilité des prédictions

---

# 📊 Données utilisées

- Jeux de données :
  - Mathématiques (`student-mat.csv`)
  - Portugais (`student-por.csv`)
- 1 044 observations après concaténation
- 34 variables initiales
- Une observation = **un élève / un cursus**

---

# 🔎 Exploration des données

- Aucune valeur manquante
- Variables de notes :
  - G1, G2, G3 fortement corrélées
- Variables socio-démographiques :
  - corrélations faibles avec G3
- Variables comportementales

---

# ⚠️ Identification des variables sensibles

- Sensibles directes :
  - sexe, âge, école, situation familiale
- Sensibles indirectes (proxies) :
  - niveau d’éducation des parents
- Comportementales :
  - consommation d’alcool, sorties

➡️ Risque de **biais socio-économiques**

---

# 🧪 Stratégie par scénarios

- **Scénario 1** : toutes les variables
- **Scénario 2** : sans variables sensibles
- **Scénario 3** : sans sensibles + sans G2
- **Scénario 4** : sans sensibles + sans G1 et G2

🎯 Comparer performance vs responsabilité

---

# ⚙️ Préparation des données

- Séparation X / y
- Pipeline sklearn :
  - standardisation des variables numériques
  - encodage one-hot des catégorielles
- Un préprocesseur par scénario
- Validation croisée (5 folds)

---

# 🤖 Modèles testés

- Régression logistique
- Random Forest
- Gradient Boosting

👉 Choix motivé par :
- interprétabilité
- robustesse
- adaptés au sujet

---

# 📈 Résultats (F1-score et Recall CV)

| Scénario | Modèle | F1-score | Recall
|--------|--------|---------|---------|
| S1 | Logistic Regression | ~0.93 | ~0.93 |
| S2 | Logistic Regression | ~0.94 | ~0.94 |
| S3 | Random Forest | ~0.92 | ~0.93 |
| S4 | Logistic Regression | ~0.87 | ~0.95 |

---

# 🧠 Analyse des résultats

- Les notes passées (G1, G2) sont très prédictives
- Retirer certaines variables sensibles :
  - peut améliorer la généralisation
- Les modèles complexes sont sensibles aux variables proxy
- La régression logistique est plus stable

---

# ✅ Choix final

- Deux modèles conservés :
  - **avec G2** → prédiction plus fiable
  - **sans G2** → prédiction plus précoce
- Variables sensibles exclues
- Bon compromis :
  - performance
  - éthique
  - explicabilité

---

# 🏗️ Industrialisation

Architecture conteneurisée :

- Backend : FastAPI
- Frontend : Streamlit
- Modèles sérialisés (`.pkl`)
- Docker Compose
- CI/CD avec GitHub Actions

---

# 🧾 Journalisation & traçabilité

- Inférence : Middleware FastAPI
- Entraînement : MLFlow

---

# 🧪 CI / CD

- CI :
  - installation
  - tests unitaires backend
- CD :
  - build image Docker
  - publication Docker Hub

---

# ⚖️ Éthique & limites

- Données éducatives sensibles
- Risque de biais indirects
- Pas d’utilisation automatique décisionnelle
- Nécessité d’un encadrement humain

---

# 🏁 Conclusion

- Solution IA fonctionnelle et responsable
- Démarche scientifique et éthique
- Comparaison des scénarios
- Industrialisation

---

# 🙏 Merci pour votre attention

## Questions ?
