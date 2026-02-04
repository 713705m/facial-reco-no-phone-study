#!/bin/bash
# Script d'installation du Détecteur de Distraction

echo "==================================="
echo "Installation du Détecteur de Distraction"
echo "==================================="

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé"
    exit 1
fi

echo "Python trouvé: $(python3 --version)"

# Créer environnement virtuel
echo ""
echo "Création de l'environnement virtuel..."
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate

# Installer les dépendances
echo ""
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==================================="
echo "Installation terminée!"
echo "==================================="
echo ""
echo "Pour lancer le détecteur:"
echo "  1. source venv/bin/activate"
echo "  2. python main.py"
echo ""
echo "N'oublie pas d'ajouter des memes dans le dossier memes/ !"
echo ""
