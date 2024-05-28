# Système de Détection d'Ordinateur et de Souris

## Description
Ce projet utilise OpenCV et le modèle de détection d'objets YOLO pour détecter des ordinateurs et des souris sur des images ou des flux vidéo. Le code est écrit en Python et utilise une approche de programmation orientée objet.

## Installation
1. Clonez le dépôt GitHub.
2. Créez et activez un environnement virtuel:
    ```sh
    python3 -m venv myenv
    source myenv/bin/activate
    ```
3. Installez les dépendances:
    ```sh
    pip install -r requirements.txt
    ```
4. Téléchargez les fichiers de configuration YOLO (`yolov3.cfg`, `yolov3.weights`, `coco.names`) et placez-les dans le dossier `yolo`.

## Utilisation
1. Mettez votre clé API dans le fichier `config.env`.
2. Exécutez le script principal:
    ```sh
    python main.py
    ```

## Structure du Projet
- `detection.py`: Contient la classe `ObjectDetector` pour la détection d'objets.
- `video_stream.py`: Contient la classe `VideoStream` pour gérer le flux vidéo.
- `main.py`: Point d'entrée du projet. Intègre OpenCV et YOLO pour traiter et afficher le flux vidéo.
- `requirements.txt`: Liste des dépendances.
- `README.md`: Documentation du projet.

## Auteur
Votre Nom
