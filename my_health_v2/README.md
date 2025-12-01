# DiagnoX - Assistant Médical IA

Assistant médical intelligent basé sur l'IA pour l'évaluation préliminaire de symptômes et l'analyse prédictive de santé.

## Fonctionnalités

### 1. Chat Conversationnel Intelligent
- Conversation naturelle en français avec IA (Claude AI ou GPT-4)
- Détection automatique des urgences médicales
- Questions de suivi contextuelles
- Interface chat moderne et intuitive

### 2. Analyse de Symptômes
- Machine Learning avec ensemble de 4 modèles (Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes)
- Prédiction de 12+ maladies courantes
- Confiance calculée par vote majoritaire
- Recommandations personnalisées

### 3. Analyse Prédictive
- Analyse basée sur l'historique des consultations
- Identification des patterns récurrents
- Prédiction des risques futurs
- Recommandations de suivi

### 4. Historique des Consultations
- Sauvegarde automatique dans MongoDB
- Consultation de l'historique
- Export des données (à venir)

## Technologies

### Backend
- **Flask** - Framework web Python
- **Anthropic Claude / OpenAI GPT** - IA conversationnelle
- **Scikit-learn** - Machine Learning
- **NLTK** - Traitement du langage naturel
- **MongoDB** - Base de données NoSQL

### Frontend
- **HTML5/CSS3** - Interface moderne
- **Vanilla JavaScript** - Logique frontend
- **Responsive Design** - Compatible mobile

### ML/IA
- **TF-IDF** - Vectorisation de texte
- **Ensemble Learning** - Combinaison de modèles
- **Cross-validation** - Validation robuste

## Installation

### Prérequis
- Python 3.9+
- MongoDB (local ou Atlas)
- Clé API: Anthropic Claude OU OpenAI

### Installation Rapide

```bash
# 1. Cloner le projet
git clone <votre-repo>
cd diagnox

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 4. Lancer l'application
python start.py
```

### Configuration .env

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017/

# API Conversationnelle (choisir une)
ANTHROPIC_API_KEY=sk-ant-xxxxx
# OU
OPENAI_API_KEY=sk-xxxxx
```

## Utilisation

### Démarrage Automatique

```bash
python start.py
```

Cela va:
1. Vérifier les dépendances
2. Télécharger les ressources NLTK
3. Créer la structure de dossiers
4. Lancer le serveur Flask
5. Ouvrir le frontend dans votre navigateur

### Démarrage Manuel

```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend
cd frontend
python -m http.server 8000
# Ouvrir http://localhost:8000
```

## Structure du Projet

```
diagnox/
├── backend/
│   ├── app.py                    # Serveur Flask
│   ├── models/
│   │   ├── conversational_agent.py
│   │   ├── disease_predictor.py
│   │   └── predictive_health_analyzer.py
│   ├── data/
│   │   └── training_data.json
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── api.js
│       ├── chat.js
│       └── app.js
├── start.py                      # Script de démarrage
└── README.md
```

## API Endpoints

### POST /api/chat
Envoie un message au chatbot et reçoit une réponse avec analyse éventuelle.

**Request:**
```json
{
  "message": "J'ai de la fièvre et des courbatures",
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "message": "Je comprends que vous avez...",
  "intent": "symptom_analysis",
  "symptoms": ["fièvre", "courbatures"],
  "possible_diseases": [
    {
      "disease": "grippe",
      "confidence": 85.5,
      "severity": "modéré"
    }
  ],
  "emergency": false
}
```

### POST /api/predict-health
Génère une analyse prédictive de santé.

**Request:**
```json
{
  "user_id": "user_123",
  "user_data": {
    "age": 35,
    "gender": "M",
    "lifestyle": "active"
  }
}
```

### GET /api/history/{user_id}
Récupère l'historique des consultations.

### GET /api/health
Vérifie l'état du serveur.

## Maladies Prédites

Le système peut analyser et prédire les maladies suivantes:

1. **Grippe** - Fièvre élevée, courbatures, toux sèche
2. **Rhume** - Nez bouché, éternuements, mal de gorge
3. **Angine** - Mal de gorge intense, fièvre, ganglions
4. **Gastro-entérite** - Diarrhée, vomissements, nausées
5. **Migraine** - Maux de tête intenses, photophobie
6. **Allergie** - Éternuements, yeux rouges, démangeaisons
7. **Bronchite** - Toux grasse, expectorations, fatigue
8. **Sinusite** - Douleur faciale, pression sinusale
9. **Otite** - Douleur d'oreille, fièvre
10. **Conjonctivite** - Yeux rouges, larmoiement
11. **Cystite** - Brûlures urinaires, envies fréquentes
12. **Varicelle** - Éruption vésiculeuse, démangeaisons

## Performance du Modèle

### Métriques (Cross-validation)
- **Random Forest**: ~92% accuracy
- **Gradient Boosting**: ~89% accuracy
- **Logistic Regression**: ~87% accuracy
- **Naive Bayes**: ~85% accuracy

### Ensemble Learning
Vote majoritaire des 4 modèles pour une prédiction plus robuste.

## Sécurité et Avertissements

⚠️ **IMPORTANT**

Ce système est un **outil d'assistance préliminaire uniquement**:

- ✗ NE REMPLACE PAS un diagnostic médical professionnel
- ✗ Ne doit PAS être utilisé pour des urgences
- ✓ Utiliser uniquement comme guide initial
- ✓ Toujours consulter un médecin pour confirmation

### Détection d'Urgences

Le système détecte automatiquement les situations d'urgence et recommande:
- Appel du 15 (SAMU)
- Consultation médicale immédiate
- Arrêt de l'auto-diagnostic

## Développement

### Ajouter une Nouvelle Maladie

1. **Éditer `data/training_data.json`**:
```json
{
  "name": "pneumonie",
  "symptoms": [
    "toux grasse fièvre élevée douleur thoracique essoufflement",
    "...autres variations..."
  ],
  "severity": "grave",
  "typical_duration": "7-14 jours"
}
```

2. **Réentraîner le modèle**:
```bash
curl -X POST http://localhost:5000/api/retrain
```

### Tests

```bash
# Tests unitaires (à venir)
pytest tests/

# Test manuel API
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "J'\''ai mal à la tête", "user_id": "test"}'
```

### Architecture ML

```
User Input → Preprocessing (NLTK)
           ↓
      TF-IDF Vectorization
           ↓
    ┌──────┴──────┬──────────┬─────────┐
    ↓             ↓          ↓         ↓
Random Forest  Gradient  Logistic  Naive
              Boosting  Regression Bayes
    │             │          │         │
    └──────┬──────┴──────────┴─────────┘
           ↓
    Vote Majoritaire
           ↓
    Prédiction Finale
```

## Contributions

Les contributions sont les bienvenues ! Pour contribuer:

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

## Roadmap

- [ ] Système de recommandations personnalisées
- [ ] Export PDF des consultations
- [ ] Support multilingue (anglais, espagnol)
- [ ] Application mobile (React Native)
- [ ] Intégration avec appareils connectés
- [ ] Téléconsultation intégrée
- [ ] Base de données de médicaments

## Licence

MIT License - Voir LICENSE pour plus de détails

## Auteurs

Projet DiagnoX - Assistant Médical IA

## Support

Pour toute question ou problème:
- GitHub Issues: [lien]
- Email: support@diagnox.ai

---

**Avertissement Légal**: Ce logiciel est fourni à des fins éducatives et informatives uniquement. Il ne constitue pas un avis médical professionnel. Consultez toujours un professionnel de santé qualifié pour toute question médicale.
