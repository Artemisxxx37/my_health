import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from datetime import datetime

class DiseasePredictor:
    """
    Système de prédiction de maladies multi-modèles avec ensemble learning
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),  # Unigrammes, bigrammes, trigrammes
            min_df=2,
            max_df=0.8
        )
        
        # Ensemble de modèles
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        
    def train(self, training_data=None):
        """
        Entraîne tous les modèles avec un dataset enrichi
        """
        print("Démarrage entraînement multi-modèles...")
        
        # Dataset médical enrichi (basé sur votre MEDICAL_KNOWLEDGE)
        if training_data is None:
            training_data = self._create_training_dataset()
        
        df = pd.DataFrame(training_data)
        
        print(f"Dataset: {len(df)} exemples, {df['disease'].nunique()} maladies")
        
        # Préparation des données
        X_text = df['symptoms']
        y = df['disease']
        
        # Vectorisation TF-IDF
        X_features = self.vectorizer.fit_transform(X_text)
        
        # Encodage des labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Entraînement de chaque modèle
        results = {}
        for name, model in self.models.items():
            print(f"\nEntraînement: {name}")
            
            model.fit(X_train, y_train)
            
            # Évaluation
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   Train: {train_score:.3f} | Test: {test_score:.3f} | CV: {cv_scores.mean():.3f} +/-{cv_scores.std():.3f}")
        
        # Calcul de l'importance des features (Random Forest)
        self._calculate_feature_importance()
        
        self.is_trained = True
        
        # Sauvegarde de l'historique
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'results': results
        })
        
        print(f"\nEntraînement terminé!")
        print(f"Meilleur modèle: {max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]}")
        
        return results
    
    def _create_training_dataset(self):
        """
        Crée un dataset d'entraînement enrichi avec variations
        """
        base_data = {
            'grippe': [
                'fièvre élevée toux sèche fatigue intense courbatures maux de tête frissons',
                'forte fièvre courbatures toux fatigue maux de tête',
                'fièvre courbatures toux frissons fatigue',
                'température élevée courbatures généralisées toux sèche',
                'fièvre 39 courbatures fatigue intense maux de tête',
                'syndrome grippal fièvre courbatures toux',
                'fièvre courbatures articulaires fatigue extrême',
                'fièvre haute toux sèche courbatures maux de tête intenses'
            ],
            'rhume': [
                'nez bouché éternuements mal de gorge toux légère',
                'nez qui coule éternuements gorge qui gratte',
                'rhinite éternuements nez bouché légère toux',
                'nez congestionné éternuements fréquents mal de gorge',
                'écoulement nasal éternuements gorge irritée',
                'nez bouché éternuements toux légère gorge qui gratte',
                'rhinite aiguë éternuements nez qui coule',
                'congestion nasale éternuements mal de gorge léger'
            ],
            'angine': [
                'mal de gorge intense difficulté à avaler fièvre ganglions',
                'gorge très douloureuse fièvre ganglions gonflés',
                'douleur gorge intense déglutition difficile fièvre',
                'mal de gorge sévère fièvre ganglions cervicaux',
                'pharyngite aiguë douleur gorge fièvre',
                'gorge enflammée difficulté avaler fièvre élevée',
                'amygdales gonflées mal de gorge intense fièvre',
                'douleur gorge aiguë fièvre ganglions palpables'
            ],
            'gastro-entérite': [
                'diarrhée vomissements nausées crampes abdominales',
                'diarrhée aiguë vomissements douleurs ventre',
                'vomissements diarrhée crampes abdominales nausées',
                'troubles digestifs diarrhée vomissements crampes',
                'diarrhée liquide vomissements douleurs abdominales',
                'gastro diarrhée vomissements crampes intestinales',
                'vomissements répétés diarrhée aiguë crampes ventre',
                'diarrhée vomissements déshydratation crampes abdominales'
            ],
            'migraine': [
                'maux de tête intenses sensibilité lumière nausées',
                'céphalées pulsatiles photophobie nausées',
                'mal de tête sévère lumière insupportable nausées',
                'migraine intense sensibilité au bruit nausées',
                'douleur crânienne pulsatile photophobie phonophobie',
                'céphalée intense lumière gênante nausées vomissements',
                'mal de tête unilatéral sensibilité lumière son',
                'migraine sévère aura visuelle nausées'
            ],
            'allergie': [
                'éternuements démangeaisons yeux rouges nez qui coule',
                'rhinite allergique éternuements yeux qui piquent',
                'yeux irrités éternuements nez qui coule démangeaisons',
                'allergie saisonnière éternuements larmoiement',
                'éternuements fréquents yeux rouges nez qui coule',
                'rhinite allergique éternuements démangeaisons nasales',
                'yeux qui pleurent éternuements nez bouché démangeaisons',
                'allergie pollen éternuements yeux irrités nez qui coule'
            ],
            'bronchite': [
                'toux grasse expectorations fatigue légère fièvre',
                'toux productive mucus fièvre modérée',
                'bronchite aiguë toux grasse expectorations',
                'toux avec glaires fièvre fatigue respiration sifflante',
                'toux productive fièvre légère essoufflement',
                'toux grasse persistante fatigue fièvre',
                'expectorations abondantes toux grasse fièvre',
                'bronchite toux productive difficultés respiratoires légères'
            ],
            'sinusite': [
                'douleur faciale nez bouché maux de tête pression sinusale',
                'sinusite frontale douleur front nez congestionné',
                'pression au niveau des sinus douleur faciale nez bouché',
                'douleur sinus maxillaires nez bouché maux de tête',
                'congestion nasale douleur pression faciale',
                'sinusite aiguë douleur front joues nez bouché',
                'pression sinusale douleur faciale sécrétions nasales',
                'douleur sinus nez bouché maux de tête frontaux'
            ]
        }
        
        # Générer le dataset
        training_data = []
        for disease, symptom_lists in base_data.items():
            for symptoms in symptom_lists:
                training_data.append({
                    'symptoms': symptoms,
                    'disease': disease
                })
        
        return training_data
    
    def predict(self, symptoms_text):
        """
        Prédiction avec vote majoritaire (ensemble learning)
        """
        if not self.is_trained:
            raise Exception("Modèle non entraîné")
        
        # Vectorisation
        X = self.vectorizer.transform([symptoms_text])
        
        # Prédictions de chaque modèle
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            pred_disease = self.label_encoder.inverse_transform([pred])[0]
            
            # Probabilités (si disponible)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                max_proba = np.max(proba)
                probabilities[name] = {
                    'disease': pred_disease,
                    'confidence': float(max_proba * 100)
                }
            else:
                probabilities[name] = {
                    'disease': pred_disease,
                    'confidence': 50.0
                }
            
            predictions[name] = pred_disease
        
        # Vote majoritaire
        from collections import Counter
        vote_counts = Counter(predictions.values())
        final_disease = vote_counts.most_common(1)[0][0]
        
        # Confiance moyenne
        avg_confidence = np.mean([
            p['confidence'] for p in probabilities.values() 
            if p['disease'] == final_disease
        ])
        
        return {
            'predicted_disease': final_disease,
            'confidence': round(avg_confidence, 2),
            'voting_details': dict(vote_counts),
            'model_predictions': probabilities,
            'consensus': len(set(predictions.values())) == 1  # Tous d'accord?
        }
    
    def predict_top_n(self, symptoms_text, n=3):
        """
        Retourne les N maladies les plus probables
        """
        if not self.is_trained:
            raise Exception("Modèle non entraîné")
        
        X = self.vectorizer.transform([symptoms_text])
        
        # Utiliser le modèle le plus performant (Random Forest)
        model = self.models['random_forest']
        
        probas = model.predict_proba(X)[0]
        top_indices = np.argsort(probas)[-n:][::-1]
        
        results = []
        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probas[idx] * 100)
            
            results.append({
                'disease': disease,
                'confidence': round(confidence, 2)
            })
        
        return results
    
    def _calculate_feature_importance(self):
        """
        Calcule l'importance des features avec Random Forest
        """
        rf_model = self.models['random_forest']
        feature_names = self.vectorizer.get_feature_names_out()
        importances = rf_model.feature_importances_
        
        # Top 20 features
        top_indices = np.argsort(importances)[-20:][::-1]
        
        self.feature_importance = {
            feature_names[i]: float(importances[i])
            for i in top_indices
        }
        
        print(f"\nTop 10 symptômes importants:")
        for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10], 1):
            print(f"   {i}. {feature}: {importance:.4f}")
    
    def save_model(self, filepath='models/disease_model.pkl'):
        """Sauvegarde tous les modèles et métadonnées"""
        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath='models/disease_model.pkl'):
        """Charge les modèles sauvegardés"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', [])
        self.feature_importance = model_data.get('feature_importance', {})
        
        print(f"Modèle chargé: {filepath}")
        print(f"   Dernière formation: {self.training_history[-1]['timestamp'] if self.training_history else 'N/A'}")
    
    def get_model_info(self):
        """Retourne les informations sur les modèles"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'models': list(self.models.keys()),
            'diseases': list(self.label_encoder.classes_),
            'n_features': len(self.vectorizer.get_feature_names_out()),
            'training_history': self.training_history,
            'top_features': list(self.feature_importance.items())[:10]
        }
    
    def explain_prediction(self, symptoms_text, predicted_disease):
        """
        Explique pourquoi cette maladie a été prédite
        """
        X = self.vectorizer.transform([symptoms_text])
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Récupérer les features actives
        active_features_indices = X.toarray()[0].nonzero()[0]
        active_features = feature_names[active_features_indices]
        active_values = X.toarray()[0][active_features_indices]
        
        # Trier par importance
        feature_importance_pairs = [
            (feat, val, self.feature_importance.get(feat, 0))
            for feat, val in zip(active_features, active_values)
        ]
        feature_importance_pairs.sort(key=lambda x: x[2], reverse=True)
        
        explanation = {
            'predicted_disease': predicted_disease,
            'key_symptoms_detected': [
                {'symptom': feat, 'weight': round(val, 3), 'importance': round(imp, 4)}
                for feat, val, imp in feature_importance_pairs[:5]
            ],
            'total_features_used': len(active_features)
        }
        
        return explanation


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def evaluate_model_performance(predictor, test_data):
    """
    Évalue la performance du modèle sur des données de test
    """
    if not predictor.is_trained:
        return {'error': 'Model not trained'}
    
    true_labels = []
    predictions = []
    
    for symptoms, true_disease in test_data:
        pred = predictor.predict(symptoms)
        predictions.append(pred['predicted_disease'])
        true_labels.append(true_disease)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_score': f1_score(true_labels, predictions, average='weighted')
    }
