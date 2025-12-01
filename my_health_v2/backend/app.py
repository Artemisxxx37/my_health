from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
from pymongo import MongoClient
from bson import ObjectId
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pickle

# Import des modules
import sys
sys.path.append('models')
from disease_predictor import DiseasePredictor
from conversational_agent import ConversationalAgent
from predictive_health_analyzer import PredictiveHealthAnalyzer

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Configuration MongoDB
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info()
    db = client['medical_chatbot']
    users_collection = db['users']
    consultations_collection = db['consultations']
    conversations_collection = db['conversations']
    predictions_collection = db['predictions']
    mongo_available = True
    print("MongoDB connecté avec succès")
except Exception as e:
    print(f"MongoDB non disponible: {e}")
    mongo_available = False

# Initialiser les modules
predictor = DiseasePredictor()
try:
    if os.path.exists('models/disease_model.pkl'):
        predictor.load_model()
        print("Modèle ML chargé depuis le fichier")
    else:
        print("Modèle ML non trouvé. Entraînement en cours...")
        predictor.train()
        os.makedirs('models', exist_ok=True)
        predictor.save_model()
        print("Modèle ML entraîné et sauvegardé")
except Exception as e:
    print(f"Erreur ML: {e}")
    predictor.train()
    predictor.save_model()

# Initialiser l'agent conversationnel et le prédicteur de santé
conversational_agent = ConversationalAgent()
predictive_analyzer = PredictiveHealthAnalyzer()

# Base de connaissances médicales
MEDICAL_KNOWLEDGE = {
    "grippe": {
        "symptoms": ["fièvre", "toux", "fatigue", "courbatures", "maux de tête", "frissons"],
        "severity": "modéré",
        "recommendations": [
            "Repos au lit pendant 5-7 jours",
            "Hydratation abondante (2-3 litres/jour)",
            "Paracétamol 1g toutes les 6h pour la fièvre",
            "Consulter si symptômes persistent > 7 jours",
            "Isolement pour éviter la contagion"
        ]
    },
    "rhume": {
        "symptoms": ["nez bouché", "éternuements", "mal de gorge", "toux légère"],
        "severity": "léger",
        "recommendations": [
            "Repos relatif",
            "Boissons chaudes (tisanes, bouillon)",
            "Décongestionnants nasaux (spray nasal)",
            "Consultation si aggravation ou fièvre",
            "Durée normale: 7-10 jours"
        ]
    },
    "angine": {
        "symptoms": ["mal de gorge intense", "difficulté à avaler", "fièvre", "ganglions"],
        "severity": "modéré",
        "recommendations": [
            "IMPORTANT: Consulter un médecin dans les 24h",
            "Test streptocoque nécessaire (angine bactérienne?)",
            "Antibiotiques si streptocoque confirmé",
            "Antidouleurs: paracétamol ou ibuprofène",
            "Aliments froids pour soulager (glace, yaourt)"
        ]
    },
    "gastro-entérite": {
        "symptoms": ["diarrhée", "vomissements", "nausées", "crampes abdominales", "fièvre légère"],
        "severity": "modéré",
        "recommendations": [
            "PRIORITAIRE: Réhydratation (SRO, eau, bouillon)",
            "Régime BRAT: Banane, Riz, Compote, Toast",
            "Repos complet",
            "Smecta ou Tiorfan pour la diarrhée",
            "URGENCE si: déshydratation sévère, sang dans les selles"
        ]
    },
    "migraine": {
        "symptoms": ["maux de tête intenses", "sensibilité à la lumière", "nausées", "vision troublée"],
        "severity": "modéré",
        "recommendations": [
            "Repos dans une pièce sombre et calme",
            "Antalgiques spécifiques (triptans si prescrits)",
            "Éviter les déclencheurs (stress, certains aliments)",
            "Compresse froide sur le front",
            "Consulter un neurologue si crises fréquentes (>4/mois)"
        ]
    },
    "allergie": {
        "symptoms": ["éternuements", "démangeaisons", "yeux rouges", "nez qui coule"],
        "severity": "léger",
        "recommendations": [
            "Antihistaminiques (Cétirizine, Loratadine)",
            "Éviter les allergènes identifiés",
            "Lavage nasal au sérum physiologique",
            "Aération quotidienne du logement",
            "Consultation allergologue pour bilan si symptômes chroniques"
        ]
    },
    "bronchite": {
        "symptoms": ["toux grasse", "expectorations", "fatigue", "fièvre légère", "essoufflement"],
        "severity": "modéré",
        "recommendations": [
            "Repos et hydratation",
            "Antitussifs pour toux sèche, expectorants pour toux grasse",
            "Inhalations de vapeur",
            "Consulter si fièvre persistante ou essoufflement",
            "Durée normale: 7-21 jours"
        ]
    },
    "sinusite": {
        "symptoms": ["douleur faciale", "nez bouché", "maux de tête", "pression sinusale"],
        "severity": "modéré",
        "recommendations": [
            "Lavages nasaux fréquents (sérum physiologique)",
            "Décongestionnants nasaux (max 5 jours)",
            "Antalgiques pour la douleur",
            "Inhalations de vapeur",
            "Consulter si douleur intense ou fièvre élevée"
        ]
    },
    "otite": {
        "symptoms": ["douleur oreille", "fièvre", "diminution audition", "écoulement"],
        "severity": "modéré",
        "recommendations": [
            "Consultation médicale recommandée",
            "Antalgiques pour la douleur",
            "Éviter l'eau dans l'oreille",
            "Ne pas utiliser de coton-tige",
            "Antibiotiques si otite bactérienne (sur prescription)"
        ]
    },
    "conjonctivite": {
        "symptoms": ["yeux rouges", "larmoiement", "démangeaisons", "sécrétions"],
        "severity": "léger",
        "recommendations": [
            "Lavage oculaire au sérum physiologique",
            "Collyres antiseptiques",
            "Éviter le port de lentilles",
            "Ne pas se frotter les yeux",
            "Consulter si aggravation ou douleur intense"
        ]
    },
    "cystite": {
        "symptoms": ["brûlures mictionnelles", "envies fréquentes", "douleurs bas ventre"],
        "severity": "modéré",
        "recommendations": [
            "Hydratation massive (2-3 litres/jour)",
            "Jus de canneberge",
            "Consultation médicale pour ECBU",
            "Antibiotiques sur prescription médicale",
            "Consulter rapidement si fièvre ou sang dans les urines"
        ]
    },
    "varicelle": {
        "symptoms": ["éruption cutanée", "vésicules", "démangeaisons", "fièvre"],
        "severity": "modéré",
        "recommendations": [
            "Éviter de gratter les lésions",
            "Antihistaminiques pour les démangeaisons",
            "Bains tièdes à l'amidon",
            "Paracétamol pour la fièvre (PAS d'aspirine)",
            "Isolement jusqu'à guérison complète"
        ]
    }
}

EMERGENCY_SYMPTOMS = [
    "douleur thoracique", "difficulté respiratoire", "perte de conscience",
    "convulsions", "hémorragie", "paralysie", "confusion mentale",
    "douleur abdominale sévère", "vomissement de sang", "paralysie faciale",
    "trouble de la parole", "perte de sensibilité", "crise cardiaque",
    "avc", "accident vasculaire", "saignement abondant", "essoufflement sévère"
]

class SymptomAnalyzer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('french'))
        except:
            print("WARNING: Stopwords français non disponibles")
            self.stop_words = set()
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        try:
            tokens = word_tokenize(text, language='french')
        except:
            tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens
    
    def extract_symptoms(self, text):
        tokens = self.preprocess_text(text)
        detected_symptoms = []
        
        for disease, info in MEDICAL_KNOWLEDGE.items():
            for symptom in info['symptoms']:
                symptom_tokens = self.preprocess_text(symptom)
                if any(token in tokens for token in symptom_tokens):
                    detected_symptoms.append(symptom)
        
        return list(set(detected_symptoms))
    
    def check_emergency(self, text):
        text_lower = text.lower()
        for emergency in EMERGENCY_SYMPTOMS:
            if emergency in text_lower:
                return True
        return False

analyzer = SymptomAnalyzer()

def save_conversation(user_id, message, response, intent):
    if mongo_available:
        try:
            conversation = {
                'user_id': user_id,
                'message': message,
                'response': response,
                'intent': intent,
                'timestamp': datetime.now()
            }
            conversations_collection.insert_one(conversation)
        except Exception as e:
            print(f"WARNING: Erreur sauvegarde conversation: {e}")

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint conversationnel principal avec analyse prédictive intégrée"""
    try:
        data = request.json
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')
        
        print(f"Chat request from user {user_id}: {user_message}")
        
        # Vérifier les urgences
        is_emergency = analyzer.check_emergency(user_message)
        if is_emergency:
            emergency_response = {
                'message': "URGENCE MÉDICALE DÉTECTÉE\n\n"
                          "Vos symptômes nécessitent une attention médicale IMMÉDIATE.\n\n"
                          "Appelez le 15 (SAMU) maintenant\n"
                          "Ou rendez-vous aux urgences les plus proches\n\n"
                          "ATTENTION: Ne perdez pas de temps, chaque minute compte !",
                'emergency': True,
                'intent': 'emergency',
                'severity': 'critique'
            }
            
            save_conversation(user_id, user_message, emergency_response['message'], 'emergency')
            
            if mongo_available:
                try:
                    consultation = {
                        'user_id': user_id,
                        'message': user_message,
                        'emergency': True,
                        'timestamp': datetime.now()
                    }
                    consultations_collection.insert_one(consultation)
                except Exception as e:
                    print(f"WARNING: Erreur MongoDB: {e}")
            
            return jsonify(emergency_response)
        
        # Gérer la conversation
        conversation_result = conversational_agent.handle_conversation(user_message, user_id)
        
        if conversation_result['needs_analysis']:
            return perform_symptom_analysis(user_message, user_id, conversation_result)
        else:
            save_conversation(user_id, user_message, conversation_result['response'], conversation_result['intent'])
            
            return jsonify({
                'message': conversation_result['response'],
                'intent': conversation_result['intent'],
                'conversational': True,
                'needs_analysis': False
            })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def perform_symptom_analysis(user_message, user_id, conversation_context):
    """Analyse complète avec diagnostic + prédiction"""
    try:
        # Extraire les symptômes
        detected_symptoms = analyzer.extract_symptoms(user_message)
        
        # Prédiction ML
        ml_result = None
        try:
            ml_result = predictor.predict(user_message)
            print(f"ML Prediction: {ml_result['predicted_disease']} ({ml_result['confidence']:.1f}%)")
        except Exception as e:
            print(f"WARNING: Erreur ML: {e}")
        
        if not detected_symptoms and not ml_result:
            no_symptoms_response = conversational_agent.generate_symptom_prompt()
            save_conversation(user_id, user_message, no_symptoms_response, 'no_symptoms')
            
            return jsonify({
                'message': no_symptoms_response,
                'symptoms': [],
                'needs_more_info': True
            })
        
        # Trouver les maladies correspondantes
        possible_diseases = []
        for disease, info in MEDICAL_KNOWLEDGE.items():
            matches = sum(1 for s in detected_symptoms if s in info['symptoms'])
            if matches > 0:
                confidence = (matches / len(info['symptoms'])) * 100
                possible_diseases.append({
                    'disease': disease,
                    'confidence': round(confidence, 2),
                    'severity': info['severity'],
                    'recommendations': info['recommendations'],
                    'method': 'rules'
                })
        
        # Ajouter prédiction ML
        if ml_result:
            ml_disease = ml_result['predicted_disease']
            if ml_disease in MEDICAL_KNOWLEDGE:
                existing = next((d for d in possible_diseases if d['disease'] == ml_disease), None)
                if existing:
                    existing['confidence'] = (existing['confidence'] + ml_result['confidence']) / 2
                    existing['method'] = 'hybrid'
                else:
                    possible_diseases.append({
                        'disease': ml_disease,
                        'confidence': round(ml_result['confidence'], 2),
                        'severity': MEDICAL_KNOWLEDGE[ml_disease]['severity'],
                        'recommendations': MEDICAL_KNOWLEDGE[ml_disease]['recommendations'],
                        'method': 'ml'
                    })
        
        possible_diseases.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Sauvegarder consultation
        if mongo_available:
            try:
                consultation = {
                    'user_id': user_id,
                    'symptoms': detected_symptoms,
                    'message': user_message,
                    'diagnosis': possible_diseases,
                    'ml_prediction': ml_result,
                    'timestamp': datetime.now()
                }
                consultations_collection.insert_one(consultation)
                print("Consultation sauvegardée")
            except Exception as e:
                print(f"WARNING: Erreur MongoDB: {e}")
        
        # Préparer réponse diagnostic
        if possible_diseases:
            top_disease = possible_diseases[0]
            
            response_message = conversation_context['response']
            response_message += f"ANALYSE DIAGNOSTIQUE\n"
            response_message += f"{'='*40}\n\n"
            
            if detected_symptoms:
                response_message += f"Symptômes identifiés:\n   {', '.join(detected_symptoms)}\n\n"
            
            response_message += f"Diagnostic le plus probable:\n"
            response_message += f"   -> {top_disease['disease'].upper()}\n"
            response_message += f"   -> Confiance: {top_disease['confidence']:.1f}%\n"
            response_message += f"   -> Gravité: {top_disease['severity']}\n"
            response_message += f"   -> Méthode: {'IA + Règles' if top_disease['method'] == 'hybrid' else 'IA' if top_disease['method'] == 'ml' else 'Règles'}\n\n"
            
            response_message += f"RECOMMANDATIONS:\n"
            for i, rec in enumerate(top_disease['recommendations'], 1):
                response_message += f"   {i}. {rec}\n"
            
            if len(possible_diseases) > 1:
                response_message += f"\nAutres diagnostics possibles:\n"
                for disease in possible_diseases[1:3]:
                    response_message += f"   • {disease['disease']} ({disease['confidence']:.1f}%)\n"
            
            response_message = conversational_agent.enhance_diagnosis_response(response_message, detected_symptoms)
            
            response_message += f"\n{'='*40}\n"
            response_message += "AVERTISSEMENT\n"
            response_message += "Cette analyse est préliminaire et automatisée.\n"
            response_message += "Consultez un professionnel de santé pour un diagnostic définitif.\n"
            
            save_conversation(user_id, user_message, response_message, 'diagnosis')
            
            return jsonify({
                'message': response_message,
                'symptoms': detected_symptoms,
                'possible_diseases': possible_diseases,
                'ml_prediction': ml_result,
                'emergency': False,
                'conversational': True
            })
        else:
            no_diagnosis = ("Je n'ai pas pu établir de diagnostic précis.\n\n"
                           "Consultez un médecin pour une évaluation complète.\n\n"
                           "Voulez-vous me donner plus de détails ?")
            
            save_conversation(user_id, user_message, no_diagnosis, 'no_diagnosis')
            
            return jsonify({
                'message': no_diagnosis,
                'symptoms': detected_symptoms,
                'emergency': False
            })
        
    except Exception as e:
        print(f"ERROR: Analyse error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-health', methods=['POST'])
def predict_health_risks():
    """Analyse prédictive des risques de santé"""
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')
        user_data = data.get('user_data', {})
        
        print(f"Predictive analysis for user {user_id}")
        
        if not mongo_available:
            return jsonify({
                'error': 'MongoDB requis pour l\'analyse prédictive',
                'message': 'Cette fonctionnalité nécessite une base de données active.'
            }), 503
        
        # Récupérer l'historique des consultations
        try:
            consultations = list(consultations_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).limit(50))
            
            # Nettoyer les ObjectId pour JSON
            for consultation in consultations:
                consultation['_id'] = str(consultation['_id'])
                if isinstance(consultation.get('timestamp'), datetime):
                    consultation['timestamp'] = consultation['timestamp'].isoformat()
            
        except Exception as e:
            print(f"ERROR: Failed to fetch history: {e}")
            return jsonify({
                'error': 'Impossible de récupérer l\'historique',
                'message': str(e)
            }), 500
        
        # Analyse de l'historique
        history_analysis = predictive_analyzer.analyze_consultation_history(consultations)
        
        if not history_analysis.get('sufficient_data'):
            return jsonify({
                'has_predictions': False,
                'message': "DONNÉES INSUFFISANTES\n\n"
                          "Pour une analyse prédictive fiable, j'ai besoin de plus de consultations.\n\n"
                          f"Consultations actuelles: {len(consultations)}\n"
                          "Minimum requis: 2-3 consultations\n\n"
                          "Continuez à utiliser DiagnoX et revenez dans quelques semaines !",
                'consultations_count': len(consultations),
                'minimum_required': 2
            })
        
        # Enrichir user_data avec l'analyse
        user_data['total_consultations'] = history_analysis.get('total_consultations', 0)
        user_data['avg_frequency'] = history_analysis.get('avg_consultation_frequency', 0)
        
        # Calculer les risques
        predictions = predictive_analyzer.calculate_disease_risk(user_data, history_analysis)
        
        # Générer le rapport
        report = predictive_analyzer.generate_prediction_report(predictions, user_data)
        
        # Déterminer le prochain contrôle
        next_checkup = predictive_analyzer.get_next_checkup_reminder(predictions)
        
        # Sauvegarder l'analyse prédictive
        try:
            prediction_record = {
                'user_id': user_id,
                'predictions': predictions,
                'history_analysis': history_analysis,
                'user_data': user_data,
                'report': report,
                'next_checkup': next_checkup,
                'timestamp': datetime.now()
            }
            predictions_collection.insert_one(prediction_record)
            print("Predictive analysis saved")
        except Exception as e:
            print(f"WARNING: Failed to save prediction: {e}")
        
        return jsonify({
            'has_predictions': report['has_predictions'],
            'message': report['message'],
            'predictions': predictions,
            'priority_level': report.get('priority_level'),
            'next_checkup': next_checkup,
            'history_summary': {
                'total_consultations': history_analysis.get('total_consultations'),
                'recurring_symptoms': history_analysis.get('recurring_symptoms'),
                'avg_frequency_days': history_analysis.get('avg_consultation_frequency')
            }
        })
        
    except Exception as e:
        print(f"ERROR: Predictive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_symptoms():
    """Ancien endpoint - redirige vers /api/chat"""
    return chat()

@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Récupère l'historique des consultations"""
    if not mongo_available:
        return jsonify({'error': 'MongoDB non disponible'}), 503
    
    try:
        consultations = list(consultations_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(10))
        
        for consultation in consultations:
            consultation['_id'] = str(consultation['_id'])
            consultation['timestamp'] = consultation['timestamp'].isoformat()
        
        return jsonify({'consultations': consultations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<user_id>', methods=['GET'])
def get_predictions_history(user_id):
    """Récupère l'historique des analyses prédictives"""
    if not mongo_available:
        return jsonify({'error': 'MongoDB non disponible'}), 503
    
    try:
        predictions = list(predictions_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(5))
        
        for pred in predictions:
            pred['_id'] = str(pred['_id'])
            pred['timestamp'] = pred['timestamp'].isoformat()
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<user_id>', methods=['GET'])
def get_conversations(user_id):
    """Récupère l'historique des conversations"""
    if not mongo_available:
        return jsonify({'error': 'MongoDB non disponible'}), 503
    
    try:
        conversations = list(conversations_collection.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(50))
        
        for conv in conversations:
            conv['_id'] = str(conv['_id'])
            conv['timestamp'] = conv['timestamp'].isoformat()
        
        return jsonify({'conversations': conversations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        'status': 'healthy',
        'ml_model': 'loaded' if predictor.is_trained else 'not loaded',
        'mongodb': 'connected' if mongo_available else 'disconnected',
        'conversational_agent': 'active',
        'predictive_analyzer': 'active',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Réentraîne le modèle ML"""
    try:
        predictor.train()
        predictor.save_model()
        return jsonify({
            'status': 'success',
            'message': 'Modèle réentraîné avec succès'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DIAGNOX MEDICAL AI - Backend Flask")
    print("="*60)
    print(f"Modèle ML: {'Chargé' if predictor.is_trained else 'Non chargé'}")
    print(f"MongoDB: {'Connecté' if mongo_available else 'Non connecté'}")
    print(f"Agent conversationnel: Actif")
    print(f"Analyseur prédictif: Actif")
    print(f"API disponible sur: http://localhost:5000")
    print("="*60)
    print("\nEndpoints disponibles:")
    print("   • POST /api/chat - Conversation + analyse")
    print("   • POST /api/predict-health - Analyse prédictive")
    print("   • POST /api/analyze - Analyse symptômes (legacy)")
    print("   • GET /api/history/<user_id> - Historique consultations")
    print("   • GET /api/predictions/<user_id> - Historique prédictions")
    print("   • GET /api/conversations/<user_id> - Historique conversations")
    print("   • GET /api/health - Status système")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
