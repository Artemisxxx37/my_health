"""
Agent conversationnel complet avec OpenAI - Version complète et corrigée
"""
import os
import json
import re
import random
from datetime import datetime
from openai import OpenAI

class ConversationalAgentOpenAI:
    """Agent conversationnel utilisant OpenAI GPT-4"""
    
    def __init__(self):
        """Initialise l'agent conversationnel avec OpenAI"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            print("⚠️ WARNING: OPENAI_API_KEY non définie dans les variables d'environnement")
            print("💡 Définissez-la avec: export OPENAI_API_KEY='votre-clé'")
            print("🔄 Mode dégradé activé (réponses prédéfinies)")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("✅ Client OpenAI initialisé avec succès")
            except Exception as e:
                print(f"❌ Erreur initialisation OpenAI: {e}")
                self.client = None
        
        self.conversation_history = {}
        
        self.system_prompt = """Tu es DiagnoX, un assistant médical IA expert et empathique.

**TON RÔLE:**
- Collecter les symptômes de manière conversationnelle et naturelle
- Poser des questions de clarification intelligentes
- Identifier les urgences médicales
- Préparer les données pour l'analyse prédictive

**RÈGLES STRICTES:**
1. NE JAMAIS donner de diagnostic définitif toi-même
2. TOUJOURS recommander une consultation médicale en cas de doute
3. IDENTIFIER les urgences (douleur thoracique, AVC, etc.) et orienter vers le 15/SAMU
4. Être empathique, rassurant mais professionnel
5. Poser des questions ciblées sur:
   - Durée des symptômes
   - Intensité (échelle 1-10)
   - Facteurs déclenchants
   - Symptômes associés
   - Antécédents médicaux pertinents

**FORMAT DE RÉPONSE:**
- Conversationnel et humain
- Questions une par une (pas de liste à puces)
- Reformuler pour confirmer la compréhension
- Être concis (2-4 phrases maximum)

**URGENCES À DÉTECTER:**
- Douleur thoracique intense
- Difficulté respiratoire sévère
- Perte de conscience
- Signes d'AVC (paralysie faciale, trouble de la parole)
- Hémorragie importante
- Convulsions
- Douleur abdominale aiguë et intense

**QUAND LANCER L'ANALYSE:**
- Quand l'utilisateur a décrit au moins 2-3 symptômes clairs
- Quand l'utilisateur demande explicitement un diagnostic
- Quand tu as assez d'informations pour une première évaluation

Réponds de manière naturelle et empathique."""
    
    def handle_conversation(self, user_message, user_id, conversation_context=None):
        """
        Gère la conversation avec contexte et détection intelligente
        
        Args:
            user_message: Message de l'utilisateur
            user_id: Identifiant unique de l'utilisateur
            conversation_context: Contexte additionnel (optionnel)
        
        Returns:
            dict: {
                'response': str,
                'intent': str,
                'needs_analysis': bool,
                'emergency': bool,
                'confidence': float,
                'collected_info': dict
            }
        """
        try:
            # Mode dégradé si pas d'API
            if not self.client:
                return self._fallback_response(user_message)
            
            # Récupérer ou initialiser l'historique
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            history = self.conversation_history[user_id]
            
            # Construire les messages avec contexte
            messages = self._build_messages(history, user_message, conversation_context)
            
            # Appel API OpenAI avec la syntaxe correcte
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Ou "gpt-4" pour plus de qualité
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extraire la réponse
            ai_response = response.choices[0].message.content
            
            # Sauvegarder dans l'historique
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": ai_response})
            
            # Limiter l'historique à 10 derniers messages
            if len(history) > 10:
                self.conversation_history[user_id] = history[-10:]
            
            # Analyser la réponse pour détecter les signaux
            analysis = self._analyze_response(ai_response, user_message)
            
            return {
                'response': ai_response,
                'intent': analysis['intent'],
                'needs_analysis': analysis['needs_analysis'],
                'emergency': analysis['emergency'],
                'confidence': analysis['confidence'],
                'collected_info': analysis['collected_info']
            }
            
        except Exception as e:
            print(f"ERROR: Erreur API conversationnelle: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response(user_message)
    
    def _build_messages(self, history, user_message, context):
        """Construit les messages avec contexte enrichi"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Ajouter l'historique
        messages.extend(history)
        
        # Ajouter contexte si disponible
        if context:
            context_msg = f"\n\n[CONTEXTE ADDITIONNEL: {json.dumps(context, ensure_ascii=False)}]"
            user_message += context_msg
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _analyze_response(self, ai_response, user_message):
        """
        Analyse la réponse de l'IA pour extraire les signaux de décision
        """
        analysis = {
            'intent': 'conversation',
            'needs_analysis': False,
            'emergency': False,
            'confidence': 0.5,
            'collected_info': {}
        }
        
        # Détection d'urgence
        emergency_keywords = [
            'urgence', 'samu', '15', 'appeler immédiatement', 
            'urgences', 'danger', 'grave', 'critique', '🚨'
        ]
        if any(kw in ai_response.lower() for kw in emergency_keywords):
            analysis['emergency'] = True
            analysis['intent'] = 'emergency'
            return analysis
        
        # Détection de demande d'analyse dans le message utilisateur
        analysis_triggers = [
            'analyser', 'diagnostic', 'évaluer', 'prédire',
            'que pensez-vous', 'quel est le problème', 'c\'est quoi',
            'qu\'est-ce que j\'ai', 'aide-moi', 'analyse mes symptômes'
        ]
        
        # Compter les symptômes mentionnés
        symptom_count = self._count_symptoms(user_message)
        
        # Vérifier si l'IA dit qu'elle va analyser
        ai_will_analyze = any(phrase in ai_response.lower() for phrase in [
            'vais analyser', 'procéder à l\'analyse', 
            'analyser vos symptômes', 'faire une évaluation',
            'regarder vos symptômes'
        ])
        
        # Décision d'analyse basée sur plusieurs facteurs
        if ai_will_analyze:
            analysis['needs_analysis'] = True
            analysis['confidence'] = 0.9
            analysis['intent'] = 'ready_for_analysis'
        elif symptom_count >= 3:
            analysis['needs_analysis'] = True
            analysis['confidence'] = min(0.9, 0.5 + (symptom_count * 0.1))
            analysis['intent'] = 'symptom_analysis'
        elif any(trigger in user_message.lower() for trigger in analysis_triggers):
            analysis['needs_analysis'] = True
            analysis['confidence'] = 0.7
            analysis['intent'] = 'diagnosis_request'
        
        # Extraire informations collectées
        analysis['collected_info'] = self._extract_medical_info(user_message)
        
        return analysis
    
    def _count_symptoms(self, text):
        """Compte les symptômes mentionnés dans le texte"""
        common_symptoms = [
            'fièvre', 'toux', 'douleur', 'fatigue', 'nausée', 'vomissement',
            'diarrhée', 'maux de tête', 'vertige', 'étourdissement',
            'essoufflement', 'palpitation', 'frisson', 'sueur',
            'mal de gorge', 'nez bouché', 'éternuement', 'courbature',
            'crampe', 'gonflement', 'rougeur', 'démangeaison',
            'mal de ventre', 'brûlure', 'picotement', 'engourdissement'
        ]
        
        text_lower = text.lower()
        count = sum(1 for symptom in common_symptoms if symptom in text_lower)
        return count
    
    def _extract_medical_info(self, text):
        """Extrait des informations médicales structurées du texte"""
        info = {}
        
        # Extraction de la durée
        duration_patterns = [
            (r'depuis (\d+) jours?', 'days'),
            (r'(\d+) heures?', 'hours'),
            (r'depuis hier', 'yesterday'),
            (r'ce matin', 'this_morning'),
            (r'cette nuit', 'last_night'),
            (r'depuis (\d+) semaines?', 'weeks')
        ]
        
        for pattern, key in duration_patterns:
            match = re.search(pattern, text.lower())
            if match:
                info['duration'] = match.group(0)
                break
        
        # Extraction d'intensité
        if any(word in text.lower() for word in ['intense', 'fort', 'sévère', 'terrible', 'insupportable']):
            info['severity'] = 'high'
        elif any(word in text.lower() for word in ['léger', 'faible', 'peu', 'modéré']):
            info['severity'] = 'low'
        else:
            info['severity'] = 'medium'
        
        # Extraction de température si mentionnée
        temp_match = re.search(r'(\d{2}(?:\.\d)?)[°\s]*(?:c|celsius)?', text.lower())
        if temp_match:
            info['temperature'] = temp_match.group(1)
        
        return info
    
    def _fallback_response(self, user_message):
        """Réponse de secours si l'API n'est pas disponible"""
        user_lower = user_message.lower()
        symptom_count = self._count_symptoms(user_message)
        
        # Détecter les salutations
        greetings = ['bonjour', 'salut', 'hello', 'hey', 'bonsoir', 'coucou']
        if any(greeting in user_lower for greeting in greetings):
            return {
                'response': "Bonjour ! Je suis DiagnoX, votre assistant médical IA.\n\n"
                           "Comment puis-je vous aider aujourd'hui ? N'hésitez pas à me décrire vos symptômes.",
                'intent': 'greeting',
                'needs_analysis': False,
                'emergency': False,
                'confidence': 0.3,
                'collected_info': {}
            }
        
        # Si plusieurs symptômes détectés
        if symptom_count >= 2:
            return {
                'response': "Je comprends que vous ressentez plusieurs symptômes. "
                           "Pouvez-vous me préciser depuis combien de temps et quelle est l'intensité ?",
                'intent': 'symptom_collection',
                'needs_analysis': symptom_count >= 3,
                'emergency': False,
                'confidence': 0.6,
                'collected_info': self._extract_medical_info(user_message)
            }
        
        # Réponse par défaut
        return {
            'response': "Je suis là pour vous aider avec vos questions de santé.\n\n"
                       "Pouvez-vous me décrire en détail ce que vous ressentez ? "
                       "Plus vous êtes précis, mieux je pourrai vous aider.",
            'intent': 'clarification',
            'needs_analysis': False,
            'emergency': False,
            'confidence': 0.3,
            'collected_info': {}
        }
    
    def generate_symptom_prompt(self):
        """Génère une question ciblée pour collecter plus de symptômes"""
        prompts = [
            "Pouvez-vous me décrire plus précisément vos symptômes ? Par exemple, depuis quand les ressentez-vous ?",
            "Pour mieux vous aider, j'aurais besoin de savoir : quelle est l'intensité de vos symptômes sur une échelle de 1 à 10 ?",
            "Avez-vous d'autres symptômes associés que vous n'avez pas encore mentionnés ?",
            "Ces symptômes sont-ils constants ou intermittents ?",
            "Y a-t-il des facteurs qui aggravent ou soulagent vos symptômes ?",
            "Avez-vous de la fièvre ? Si oui, quelle température ?"
        ]
        
        return random.choice(prompts)
    
    def enhance_diagnosis_response(self, base_response, symptoms):
        """Enrichit la réponse de diagnostic avec un ton conversationnel et empathique"""
        if not self.client:
            # Mode dégradé : ajouter une phrase empathique simple
            return ("Je comprends que ces symptômes vous inquiètent. "
                   "Voici mon analyse :\n\n" + base_response)
        
        try:
            # Demander à l'IA d'améliorer la réponse
            enhancement_prompt = f"""Améliore cette réponse de diagnostic médical pour la rendre plus empathique et claire, 
sans changer les informations médicales.

Réponse originale:
{base_response}

Symptômes du patient: {', '.join(symptoms)}

Fournis une version améliorée qui:
1. Commence par UNE phrase empathique courte (1 phrase seulement)
2. Garde TOUTES les informations médicales exactement comme elles sont
3. Reste professionnelle et rassurante
4. Ne rajoute PAS de conclusion ou de phrase finale

Retourne UNIQUEMENT la phrase empathique suivie du diagnostic complet."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.8,
                max_tokens=600
            )
            
            enhanced = response.choices[0].message.content
            return enhanced
            
        except Exception as e:
            print(f"⚠️ WARNING: Erreur amélioration réponse: {e}")
            return base_response
    
    def clear_history(self, user_id):
        """Efface l'historique de conversation d'un utilisateur"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            return True
        return False
    
    def get_conversation_summary(self, user_id):
        """Génère un résumé de la conversation pour l'utilisateur"""
        if user_id not in self.conversation_history:
            return None
        
        history = self.conversation_history[user_id]
        
        # Extraire tous les messages utilisateur
        user_messages = [msg['content'] for msg in history if msg['role'] == 'user']
        
        # Compter les symptômes totaux mentionnés
        all_symptoms = []
        for msg in user_messages:
            symptoms = self._count_symptoms(msg)
            if symptoms > 0:
                all_symptoms.append(msg)
        
        return {
            'total_messages': len(user_messages),
            'symptom_messages': len(all_symptoms),
            'collected_info': self._extract_medical_info(' '.join(user_messages))
        }


# Alias pour compatibilité
ConversationalAgent = ConversationalAgentOpenAI