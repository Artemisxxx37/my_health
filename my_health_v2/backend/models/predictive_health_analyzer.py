from datetime import datetime, timedelta
from collections import Counter
import json

class PredictiveHealthAnalyzer:
    """
    Analyseur prédictif de santé basé sur l'historique des consultations
    """
    
    def __init__(self):
        self.risk_factors = {
            'age': {
                'young': (0, 18),
                'adult': (18, 60),
                'senior': (60, 120)
            },
            'frequency_threshold': 3,  # consultations par mois
            'recurring_threshold': 2   # même symptôme X fois
        }
    
    def analyze_consultation_history(self, consultations):
        """
        Analyse l'historique des consultations pour identifier les patterns
        """
        if not consultations or len(consultations) < 2:
            return {
                'sufficient_data': False,
                'total_consultations': len(consultations)
            }
        
        # Extraire les symptômes de toutes les consultations
        all_symptoms = []
        all_diseases = []
        dates = []
        
        for consultation in consultations:
            if 'symptoms' in consultation and consultation['symptoms']:
                all_symptoms.extend(consultation['symptoms'])
            
            if 'diagnosis' in consultation and consultation['diagnosis']:
                for diag in consultation['diagnosis']:
                    all_diseases.append(diag['disease'])
            
            if 'timestamp' in consultation:
                timestamp = consultation['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                dates.append(timestamp)
        
        # Compter les occurrences
        symptom_counts = Counter(all_symptoms)
        disease_counts = Counter(all_diseases)
        
        # Calculer la fréquence de consultation
        if len(dates) > 1:
            dates_sorted = sorted(dates)
            time_span = (dates_sorted[-1] - dates_sorted[0]).days
            avg_frequency = time_span / len(dates) if len(dates) > 0 else 0
        else:
            avg_frequency = 0
        
        # Identifier les symptômes récurrents
        recurring_symptoms = {
            symptom: count 
            for symptom, count in symptom_counts.items() 
            if count >= self.risk_factors['recurring_threshold']
        }
        
        # Identifier les maladies récurrentes
        recurring_diseases = {
            disease: count 
            for disease, count in disease_counts.items() 
            if count >= self.risk_factors['recurring_threshold']
        }
        
        return {
            'sufficient_data': True,
            'total_consultations': len(consultations),
            'unique_symptoms': len(symptom_counts),
            'unique_diseases': len(disease_counts),
            'recurring_symptoms': recurring_symptoms,
            'recurring_diseases': recurring_diseases,
            'avg_consultation_frequency': avg_frequency,
            'most_common_symptoms': symptom_counts.most_common(5),
            'most_common_diseases': disease_counts.most_common(3)
        }
    
    def calculate_disease_risk(self, user_data, history_analysis):
        """
        Calcule le risque de développer certaines maladies
        """
        predictions = []
        
        # Facteurs de risque basés sur l'âge
        age = user_data.get('age', 30)
        age_risk_multiplier = self._get_age_risk_multiplier(age)
        
        # Facteurs de risque basés sur le style de vie
        lifestyle = user_data.get('lifestyle', 'active')
        lifestyle_risk_multiplier = self._get_lifestyle_risk_multiplier(lifestyle)
        
        # Analyser les maladies récurrentes
        recurring_diseases = history_analysis.get('recurring_diseases', {})
        
        for disease, count in recurring_diseases.items():
            # Calculer le score de risque
            base_risk = (count / history_analysis['total_consultations']) * 100
            adjusted_risk = base_risk * age_risk_multiplier * lifestyle_risk_multiplier
            
            # Limiter entre 0 et 100
            risk_score = min(100, max(0, adjusted_risk))
            
            # Déterminer le niveau de risque
            if risk_score >= 70:
                risk_level = 'high'
            elif risk_score >= 40:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            predictions.append({
                'disease': disease,
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'occurrences': count,
                'recommendations': self._get_recommendations(disease, risk_level)
            })
        
        # Analyser les symptômes récurrents
        recurring_symptoms = history_analysis.get('recurring_symptoms', {})
        
        if recurring_symptoms and not predictions:
            # Si pas de maladies récurrentes mais des symptômes récurrents
            most_common_symptom = max(recurring_symptoms.items(), key=lambda x: x[1])
            
            predictions.append({
                'disease': 'condition_chronique',
                'risk_score': 50.0,
                'risk_level': 'medium',
                'occurrences': most_common_symptom[1],
                'recommendations': [
                    f"Symptôme récurrent détecté: {most_common_symptom[0]}",
                    "Consultation médicale recommandée pour bilan",
                    "Tenir un journal des symptômes",
                    "Identifier les facteurs déclenchants"
                ]
            })
        
        # Trier par score de risque décroissant
        predictions.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return predictions
    
    def _get_age_risk_multiplier(self, age):
        """Retourne un multiplicateur de risque basé sur l'âge"""
        if age < 18:
            return 0.8
        elif age < 40:
            return 1.0
        elif age < 60:
            return 1.2
        else:
            return 1.5
    
    def _get_lifestyle_risk_multiplier(self, lifestyle):
        """Retourne un multiplicateur de risque basé sur le style de vie"""
        lifestyle_map = {
            'very_active': 0.8,
            'active': 1.0,
            'sedentary': 1.3
        }
        return lifestyle_map.get(lifestyle, 1.0)
    
    def _get_recommendations(self, disease, risk_level):
        """Génère des recommandations basées sur la maladie et le niveau de risque"""
        recommendations_map = {
            'grippe': {
                'high': [
                    "Vaccination antigrippale annuelle fortement recommandée",
                    "Consultation médicale pour évaluer le système immunitaire",
                    "Renforcer les mesures d'hygiène (lavage des mains)",
                    "Éviter les lieux publics en période épidémique"
                ],
                'medium': [
                    "Envisager la vaccination antigrippale",
                    "Repos adéquat et alimentation équilibrée",
                    "Hygiène des mains régulière"
                ],
                'low': [
                    "Maintenir une bonne hygiène de vie",
                    "Vaccination si personne à risque"
                ]
            },
            'allergie': {
                'high': [
                    "Consultation allergologue pour tests cutanés",
                    "Traitement de désensibilisation possible",
                    "Éviction stricte des allergènes identifiés",
                    "Avoir un plan d'action d'urgence"
                ],
                'medium': [
                    "Identifier les allergènes déclencheurs",
                    "Antihistaminiques préventifs si nécessaire",
                    "Aération régulière du domicile"
                ],
                'low': [
                    "Surveiller les symptômes saisonniers",
                    "Antihistaminiques occasionnels"
                ]
            },
            'migraine': {
                'high': [
                    "Consultation neurologique recommandée",
                    "Tenir un journal des migraines (déclencheurs)",
                    "Traitement de fond à envisager",
                    "Éviter les facteurs déclenchants connus"
                ],
                'medium': [
                    "Identifier les déclencheurs (stress, aliments)",
                    "Traitement préventif si crises fréquentes",
                    "Techniques de relaxation"
                ],
                'low': [
                    "Gérer le stress",
                    "Antalgiques dès les premiers signes"
                ]
            }
        }
        
        default_recommendations = {
            'high': [
                "Consultation médicale recommandée",
                "Suivi régulier nécessaire",
                "Maintenir un mode de vie sain"
            ],
            'medium': [
                "Surveillance des symptômes",
                "Consultation si aggravation"
            ],
            'low': [
                "Prévention et hygiène de vie"
            ]
        }
        
        disease_recs = recommendations_map.get(disease, default_recommendations)
        return disease_recs.get(risk_level, default_recommendations['medium'])
    
    def generate_prediction_report(self, predictions, user_data):
        """
        Génère un rapport textuel de l'analyse prédictive
        """
        if not predictions:
            return {
                'has_predictions': False,
                'message': "Aucune prédiction de risque identifiée pour le moment."
            }
        
        report = "ANALYSE PRÉDICTIVE DE SANTÉ\n"
        report += "="*40 + "\n\n"
        
        # Informations utilisateur
        age = user_data.get('age', 'Non spécifié')
        lifestyle = user_data.get('lifestyle', 'Non spécifié')
        
        report += f"Profil:\n"
        report += f"   Age: {age}\n"
        report += f"   Style de vie: {lifestyle}\n"
        report += f"   Consultations analysées: {user_data.get('total_consultations', 0)}\n\n"
        
        # Risques identifiés
        report += "RISQUES IDENTIFIÉS:\n\n"
        
        high_risks = [p for p in predictions if p['risk_level'] == 'high']
        medium_risks = [p for p in predictions if p['risk_level'] == 'medium']
        low_risks = [p for p in predictions if p['risk_level'] == 'low']
        
        if high_risks:
            report += "PRIORITÉ ÉLEVÉE:\n"
            for pred in high_risks:
                report += f"   • {pred['disease'].upper()}\n"
                report += f"     Score de risque: {pred['risk_score']}%\n"
                report += f"     Occurrences: {pred['occurrences']}\n"
        
        if medium_risks:
            report += "\nPRIORITÉ MOYENNE:\n"
            for pred in medium_risks:
                report += f"   • {pred['disease'].capitalize()}\n"
                report += f"     Score de risque: {pred['risk_score']}%\n"
        
        if low_risks:
            report += "\nRISQUE FAIBLE:\n"
            for pred in low_risks:
                report += f"   • {pred['disease'].capitalize()} ({pred['risk_score']}%)\n"
        
        # Recommandations principales
        report += "\n" + "="*40 + "\n"
        report += "RECOMMANDATIONS PRINCIPALES:\n\n"
        
        if high_risks:
            top_risk = high_risks[0]
            report += f"Pour {top_risk['disease']}:\n"
            for i, rec in enumerate(top_risk['recommendations'], 1):
                report += f"   {i}. {rec}\n"
        
        report += "\n" + "="*40 + "\n"
        report += "AVERTISSEMENT:\n"
        report += "Cette analyse est basée sur votre historique et des facteurs généraux.\n"
        report += "Elle ne remplace pas un avis médical professionnel.\n"
        
        # Déterminer le niveau de priorité global
        if high_risks:
            priority_level = 'high'
        elif medium_risks:
            priority_level = 'medium'
        else:
            priority_level = 'low'
        
        return {
            'has_predictions': True,
            'message': report,
            'priority_level': priority_level
        }
    
    def get_next_checkup_reminder(self, predictions):
        """
        Détermine quand le prochain contrôle médical est recommandé
        """
        if not predictions:
            return "Prochain contrôle: Dans 6 mois pour un bilan de routine"
        
        high_risks = [p for p in predictions if p['risk_level'] == 'high']
        medium_risks = [p for p in predictions if p['risk_level'] == 'medium']
        
        if high_risks:
            return "Prochain contrôle: URGENT - Consultation médicale dans les 2 semaines"
        elif medium_risks:
            return "Prochain contrôle: Dans 1-2 mois pour suivi"
        else:
            return "Prochain contrôle: Dans 3-6 mois pour surveillance"
