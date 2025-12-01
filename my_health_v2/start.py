import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path



def check_nltk_data():
    """Télécharge les données NLTK si nécessaire"""
    print("\nVérification des ressources NLTK...")
    
    import nltk
    
    resources = ['punkt', 'stopwords']
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            print(f"   OK: {resource}")
        except LookupError:
            print(f"   Téléchargement: {resource}")
            nltk.download(resource, quiet=True)
    
    print("✓ Ressources NLTK prêtes")

def check_env_file():
    """Vérifie que le fichier .env existe"""
    print("\n Vérification de la configuration...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            print("   ATTENTION: Fichier .env manquant")
            print("   Création depuis .env.example...")
            env_example.read_text()
            with open('.env', 'w') as f:
                f.write(env_example.read_text())
            print("\n   IMPORTANT: Modifiez le fichier .env avec vos clés API")
            print("   - ANTHROPIC_API_KEY ou OPENAI_API_KEY")
            print("   - MONGO_URI (optionnel)")
            
            response = input("\n   Voulez-vous continuer sans clé API? (o/n): ")
            if response.lower() != 'o':
                print("\n   Arrêt. Configurez .env puis relancez.")
                sys.exit(0)
        else:
            print("     Aucun fichier de configuration trouvé")
    else:
        print("    Fichier .env trouvé")
    
    # Charger les variables d'environnement
    from dotenv import load_dotenv
    load_dotenv()
    
    # Vérifier les clés API
    has_anthropic = bool(os.getenv('ANTHROPIC_API_KEY'))
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    
    if not (has_anthropic or has_openai):
        print("\n   AVERTISSEMENT: Aucune clé API configurée")
        print("   Le mode conversationnel sera limité")
    else:
        api_type = "Claude AI" if has_anthropic else "OpenAI GPT-4"
        print(f"    API configurée: {api_type}")

def check_directories():
    """Crée les dossiers nécessaires"""
    print("\n Vérification de la structure...")
    
    directories = ['models', 'data', 'logs', 'frontend/css', 'frontend/js']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Créé: {directory}")
        else:
            print(f"   OK: {directory}")
    
    print("Structure de dossiers prête")

def start_backend():
    """Lance le serveur Flask"""
    print("\n Démarrage du serveur backend...")
    print("   URL: http://localhost:5000")
    print("   Logs: Voir la console ci-dessous")
    print("\n" + "="*60 + "\n")
    
    # Changer vers le répertoire backend si nécessaire
    if Path('backend').exists():
        os.chdir('backend')
    
    # Lancer Flask
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n Arrêt du serveur")
    except Exception as e:
        print(f"\nErreur lors du démarrage: {e}")
        sys.exit(1)

def open_frontend():
    """Ouvre le frontend dans le navigateur"""
    time.sleep(2)  # Attendre que le serveur démarre
    
    frontend_path = Path('frontend/index.html')
    
    if frontend_path.exists():
        url = f'file://{frontend_path.absolute()}'
        print(f"\n Ouverture du frontend: {url}")
        webbrowser.open(url)
    else:
        print("\n  Frontend non trouvé. Accédez manuellement à:")
        print("   http://localhost:5000/")

def main():
    """Point d'entrée principal"""
    try:
        
        # Vérifications
        check_nltk_data()
        check_env_file()
        check_directories()
        
        # Ouvrir le frontend dans un thread séparé
        import threading
        frontend_thread = threading.Thread(target=open_frontend, daemon=True)
        frontend_thread.start()
        
        # Lancer le backend (bloquant)
        start_backend()
        
    except KeyboardInterrupt:
        print("\n\n Arrêt de DiagnoX")
        sys.exit(0)
    except Exception as e:
        print(f"\nErreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
