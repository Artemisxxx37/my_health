"""
data_generator.py
Générateur de dataset pour DiagnoX
Crée un CSV avec symptômes et diagnostics
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Base de connaissances: Maladie -> Symptômes
DISEASE_SYMPTOMS = {
    "Grippe": {
        "symptoms": ["fièvre", "toux", "fatigue", "courbatures", "mal_tete", "frissons"],
        "severity": "modéré"
    },
    "Rhume": {
        "symptoms": ["nez_bouche", "toux_legere", "mal_gorge", "fatigue"],
        "severity": "léger"
    },
    "Gastro": {
        "symptoms": ["nausée", "vomissement", "diarrhée", "crampes_abdominales"],
        "severity": "modéré"
    },
    "Angine": {
        "symptoms": ["mal_gorge_intense", "fièvre", "ganglions", "difficulté_avaler"],
        "severity": "modéré"
    },
    "Migraine": {
        "symptoms": ["mal_tete_intense", "sensibilité_lumière", "nausée", "vision_troublée"],
        "severity": "modéré"
    },
    "Allergie": {
        "symptoms": ["éternuements", "démangeaisons", "yeux_rouges", "nez_qui_coule"],
        "severity": "léger"
    },
    "Bronchite": {
        "symptoms": ["toux", "fièvre", "essoufflement", "poitrine_douloureuse"],
        "severity": "grave"
    },
    "Diabète": {
        "symptoms": ["fatigue", "soif_excessive", "mictions_fréquentes", "perte_poids"],
        "severity": "modéré"
    },
    "Hypertension": {
        "symptoms": ["mal_tete", "vertiges", "essoufflement", "douleur_poitrine"],
        "severity": "modéré"
    },
    "Pneumonie": {
        "symptoms": ["toux", "fièvre", "essoufflement", "douleur_poitrine", "mal_tete"],
        "severity": "grave"
    },
    "Arthrite": {
        "symptoms": ["douleurs_articulations", "gonflement", "raideur", "fatigue"],
        "severity": "modéré"
    },
    "Sinusite": {
        "symptoms": ["mal_tete", "nez_bouche", "congestion_nasale", "mal_gorge"],
        "severity": "léger"
    },
}

# Tous les symptômes possibles
ALL_SYMPTOMS = sorted(list(set([s for disease_info in DISEASE_SYMPTOMS.values() 
                                 for s in disease_info["symptoms"]])))


def create_symptom_vector(disease_symptoms, all_symptoms):
    """
    Crée un vecteur de symptômes (0/1) pour l'entraînement ML
    
    Args:
        disease_symptoms: liste des symptômes de la maladie
        all_symptoms: liste de tous les symptômes possibles
        
    Returns:
        dict avec symptômes encodés
    """
    return {symptom: 1 if symptom in disease_symptoms else 0 
            for symptom in all_symptoms}


def generate_training_data(output_file='data/training_data.csv', variations=True):
    """
    Génère le dataset d'entraînement
    
    Args:
        output_file: chemin du fichier CSV à créer
        variations: si True, crée des variations (cas léger, cas grave, etc.)
        
    Returns:
        pandas DataFrame
    """
    
    rows = []
    
    print("📊 Génération du dataset d'entraînement...")
    print(f"   Maladies: {len(DISEASE_SYMPTOMS)}")
    print(f"   Symptômes uniques: {len(ALL_SYMPTOMS)}")
    print()
    
    # 1. Pour chaque maladie, créer les exemples
    for disease_name, disease_info in DISEASE_SYMPTOMS.items():
        symptoms = disease_info["symptoms"]
        
        # Cas 1: Tous les symptômes (cas typique)
        print(f"   ✓ {disease_name} - cas complet")
        row = create_symptom_vector(symptoms, ALL_SYMPTOMS)
        row['disease'] = disease_name
        row['severity'] = disease_info["severity"]
        rows.append(row)
        
        # Cas 2: Symptômes légers (75% des symptômes)
        if variations and len(symptoms) > 2:
            light_symptoms = symptoms[:len(symptoms)-1]
            row = create_symptom_vector(light_symptoms, ALL_SYMPTOMS)
            row['disease'] = disease_name
            row['severity'] = "léger"
            rows.append(row)
        
        # Cas 3: Symptômes graves (tous les symptômes + 1 symptôme aléatoire)
        if variations and len(ALL_SYMPTOMS) > len(symptoms):
            other_symptoms = [s for s in ALL_SYMPTOMS if s not in symptoms]
            severe_symptoms = symptoms + [np.random.choice(other_symptoms, 1)[0]]
            row = create_symptom_vector(severe_symptoms, ALL_SYMPTOMS)
            row['disease'] = disease_name
            row['severity'] = "grave"
            rows.append(row)
        
        # Cas 4: Symptômes minimes (2-3 symptômes principaux)
        if variations and len(symptoms) > 2:
            minimal_symptoms = symptoms[:2]
            row = create_symptom_vector(minimal_symptoms, ALL_SYMPTOMS)
            row['disease'] = disease_name
            row['severity'] = "léger"
            rows.append(row)
    
    # Créer DataFrame
    df = pd.DataFrame(rows)
    
    # Créer le dossier s'il existe pas
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', 
                exist_ok=True)
    
    # Sauvegarder
    df.to_csv(output_file, index=False)
    
    print()
    print("✅ Dataset créé avec succès!")
    print(f"   Fichier: {output_file}")
    print(f"   Exemples: {len(df)}")
    print(f"   Maladies: {df['disease'].nunique()}")
    print(f"   Symptômes: {len(ALL_SYMPTOMS)}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df


def load_training_data(filepath='data/training_data.csv'):
    """
    Charge le dataset d'entraînement
    
    Args:
        filepath: chemin du fichier CSV
        
    Returns:
        pandas DataFrame
    """
    if not os.path.exists(filepath):
        print(f"⚠️  Fichier {filepath} introuvable. Génération...")
        return generate_training_data(filepath)
    
    df = pd.read_csv(filepath)
    print(f"✅ Dataset chargé: {len(df)} exemples")
    return df


def get_disease_info(disease_name):
    """
    Récupère les infos d'une maladie
    
    Args:
        disease_name: nom de la maladie
        
    Returns:
        dict avec symptômes et sévérité
    """
    return DISEASE_SYMPTOMS.get(disease_name, None)


def get_all_symptoms():
    """Retourne la liste de tous les symptômes"""
    return ALL_SYMPTOMS


def get_all_diseases():
    """Retourne la liste de toutes les maladies"""
    return list(DISEASE_SYMPTOMS.keys())


def add_disease(disease_name, symptoms, severity="modéré"):
    """
    Ajoute une nouvelle maladie au dataset
    
    Args:
        disease_name: nom de la maladie
        symptoms: liste des symptômes
        severity: sévérité (léger, modéré, grave)
    """
    DISEASE_SYMPTOMS[disease_name] = {
        "symptoms": symptoms,
        "severity": severity
    }
    print(f"✅ Maladie ajoutée: {disease_name}")


def export_dataset_info(output_file='data/dataset_info.txt'):
    """
    Exporte les infos du dataset dans un fichier texte
    
    Args:
        output_file: chemin du fichier
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DIAGNOX - DATASET INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Maladies: {len(DISEASE_SYMPTOMS)}\n")
        f.write(f"Symptômes uniques: {len(ALL_SYMPTOMS)}\n\n")
        
        f.write("MALADIES ET SYMPTÔMES:\n")
        f.write("-" * 60 + "\n")
        
        for disease_name, disease_info in DISEASE_SYMPTOMS.items():
            f.write(f"\n{disease_name} ({disease_info['severity']})\n")
            f.write(f"  Symptômes: {', '.join(disease_info['symptoms'])}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("TOUS LES SYMPTÔMES:\n")
        f.write("-" * 60 + "\n")
        for i, symptom in enumerate(ALL_SYMPTOMS, 1):
            f.write(f"{i:2d}. {symptom}\n")
    
    print(f"✅ Info dataset exportée: {output_file}")


if __name__ == "__main__":
    # Script de test
    print("\n" + "="*60)
    print("🏥 DIAGNOX - Data Generator")
    print("="*60 + "\n")
    
    # Générer dataset
    df = generate_training_data()
    
    # Afficher aperçu
    print("\n📋 Aperçu du dataset:")
    print(df.head())
    
    print("\n📊 Statistiques:")
    print(f"   Shape: {df.shape}")
    print(f"   Colonnes: {list(df.columns)[:5]}... (+ {len(df.columns) - 5} autres)")
    print(f"   Distributions:")
    print(df['disease'].value_counts())
    
    # Exporter infos
    export_dataset_info()
    
    print("\n✅ Prêt pour l'entraînement ML!")
