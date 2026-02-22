# Détecteur de Distraction Téléphone

Un système de Computer Vision en temps réel qui détecte quand tu regardes ton téléphone au lieu de travailler, et t'alerte avec un son + des memes motivants.

```bash
source venv/bin/activate && python main.py
```

## Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Théorie et Concepts](#théorie-et-concepts)
  - [1. Détection de visage avec MediaPipe](#1-détection-de-visage-avec-mediapipe)
  - [2. Estimation de la pose de tête (Head Pose Estimation)](#2-estimation-de-la-pose-de-tête-head-pose-estimation)
  - [3. Détection d'objets avec YOLO](#3-détection-dobjets-avec-yolo)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Structure du projet](#structure-du-projet)
- [Plus-values CV](#plus-values-cv)
- [Axes d'amélioration](#axes-damélioration)

---

## Fonctionnalités

- **Détection de visage en temps réel** - 468 points de repère faciaux via MediaPipe
- **Estimation de la direction du regard** - Calcul des angles de rotation de la tête (pitch/yaw/roll)
- **Détection de téléphone** - YOLOv8 pré-entraîné sur COCO dataset
- **Système d'alerte** - Son + affichage de memes aléatoires
- **Calibration personnalisée** - Ajuste les seuils à ton setup
- **Logging des statistiques** - Suivi de tes distractions en JSON

---

## Théorie et Concepts

### 1. Détection de visage avec MediaPipe

#### Qu'est-ce que MediaPipe ?

**MediaPipe** est un framework open-source développé par Google pour construire des pipelines de machine learning multimodaux. Il fournit des modèles pré-entraînés pour diverses tâches de vision par ordinateur.

#### Face Landmarker

Le modèle **Face Landmarker** détecte 468 points de repère (landmarks) sur un visage en temps réel :

```
Points clés utilisés dans ce projet :
- Point 1   : Bout du nez
- Point 152 : Menton
- Point 33  : Coin externe œil gauche
- Point 263 : Coin externe œil droit
- Point 61  : Coin gauche de la bouche
- Point 291 : Coin droit de la bouche
```

#### Pourquoi 468 points ?

Le maillage facial (Face Mesh) fournit une représentation dense du visage permettant :
- Une estimation précise de la pose 3D
- Le suivi des expressions faciales
- La détection du regard (gaze estimation)

```
Coordonnées retournées (normalisées 0-1) :
- x : position horizontale
- y : position verticale
- z : profondeur (distance relative à la caméra)
```

---

### 2. Estimation de la pose de tête (Head Pose Estimation)

#### Le problème PnP (Perspective-n-Point)

L'estimation de pose consiste à trouver la **rotation** et la **translation** d'un objet 3D à partir de ses projections 2D dans une image.

```
Données d'entrée :
- Points 3D de référence (modèle de visage générique en mm)
- Points 2D correspondants (landmarks détectés en pixels)
- Paramètres intrinsèques de la caméra

Données de sortie :
- Vecteur de rotation (3 angles)
- Vecteur de translation (position 3D)
```

#### Modèle 3D de référence

On utilise un modèle anthropométrique standard d'un visage humain :

```python
MODEL_POINTS = [
    (0.0, 0.0, 0.0),           # Nez - origine
    (0.0, -330.0, -65.0),      # Menton
    (-225.0, 170.0, -135.0),   # Œil gauche
    (225.0, 170.0, -135.0),    # Œil droit
    (-150.0, -150.0, -125.0),  # Bouche gauche
    (150.0, -150.0, -125.0)    # Bouche droite
]
# Distances en millimètres
```

#### Algorithme solvePnP

OpenCV implémente plusieurs algorithmes pour résoudre le problème PnP :

```python
success, rotation_vec, translation_vec = cv2.solvePnP(
    objectPoints=MODEL_POINTS,    # Points 3D du modèle
    imagePoints=image_points,     # Points 2D détectés
    cameraMatrix=camera_matrix,   # Matrice intrinsèque
    distCoeffs=dist_coeffs,       # Coefficients de distorsion
    flags=cv2.SOLVEPNP_ITERATIVE  # Méthode itérative
)
```

#### Angles d'Euler (Pitch, Yaw, Roll)

La rotation 3D est décomposée en 3 angles :

```
PITCH (tangage) : Rotation autour de l'axe X
  → Positif = regarde vers le haut
  → Négatif = regarde vers le bas (téléphone!)

YAW (lacet) : Rotation autour de l'axe Y
  → Positif = tourne à droite
  → Négatif = tourne à gauche

ROLL (roulis) : Rotation autour de l'axe Z
  → Inclinaison latérale de la tête
```

```
        Y (Yaw)
        ↑
        |
        |
        +------→ X (Pitch)
       /
      /
     ↓
    Z (Roll)
```

#### Extraction des angles depuis la matrice de rotation

```python
# Convertir vecteur de rotation → matrice 3x3
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

# Extraire les angles d'Euler
pitch = arctan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
yaw   = arctan2(R[1,0], R[0,0])
roll  = arctan2(R[2,1], R[2,2])
```

---

### 3. Détection d'objets avec YOLO

#### Qu'est-ce que YOLO ?

**YOLO** (You Only Look Once) est une famille de modèles de détection d'objets en temps réel. Contrairement aux méthodes traditionnelles qui scannent l'image en plusieurs passes, YOLO analyse l'image entière en une seule passe.

#### Architecture YOLOv8

```
Image (640x480)
      ↓
┌─────────────────┐
│   Backbone      │  ← Extraction de features (CSPDarknet)
│   (CNN)         │
└────────┬────────┘
         ↓
┌─────────────────┐
│      Neck       │  ← Fusion multi-échelle (PANet)
│   (FPN + PAN)   │
└────────┬────────┘
         ↓
┌─────────────────┐
│      Head       │  ← Prédiction (classes + bbox)
│   (Détection)   │
└─────────────────┘
         ↓
    Sorties :
    - Bounding boxes (x, y, w, h)
    - Classes (80 classes COCO)
    - Scores de confiance
```

#### Dataset COCO

YOLOv8 est pré-entraîné sur **COCO** (Common Objects in Context) qui contient 80 classes d'objets :

```
Classe 67 : "cell phone" (téléphone portable)

Autres classes utiles :
- 0  : person
- 63 : laptop
- 66 : keyboard
- 72 : tv
```

#### Non-Maximum Suppression (NMS)

Quand plusieurs bounding boxes détectent le même objet, NMS garde uniquement la meilleure :

```
Avant NMS :           Après NMS :
┌──────────┐         ┌──────────┐
│ ┌──────┐ │         │          │
│ │ 0.92 │ │    →    │   0.92   │
│ └──────┘ │         │          │
│   0.87   │         └──────────┘
└──────────┘
```

---

## Installation

### Prérequis

- Python 3.10+
- Webcam
- macOS / Linux / Windows

### Installation rapide

```bash
cd ~/Documents/projet-reco-faciale

# Créer l'environnement virtuel et installer les dépendances
./setup.sh

# Ou manuellement :
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| opencv-python | ≥4.8.0 | Capture vidéo, affichage |
| mediapipe | ≥0.10.0 | Détection visage (468 landmarks) |
| ultralytics | ≥8.0.0 | YOLOv8 (détection téléphone) |
| numpy | ≥1.24.0 | Calculs matriciels |
| pygame | ≥2.5.0 | Lecture audio |
| PyYAML | ≥6.0 | Configuration |

---

## Utilisation

### Lancement basique

```bash
source venv/bin/activate
python main.py
```

### Options

```bash
python main.py              # Mode normal
python main.py --calibrate  # Lance la calibration au démarrage
python main.py --stats      # Affiche les statistiques hebdomadaires
```

### Contrôles (dans la fenêtre)

| Touche | Action |
|--------|--------|
| `q` | Quitter et sauvegarder les stats |
| `c` | Lancer la calibration |
| `r` | Rafraîchir la liste des memes |

### Ajouter des memes

Place tes images dans le dossier `memes/` :

```bash
cp ~/Downloads/meme1.jpg memes/
cp ~/Downloads/stop_scrolling.png memes/
```

Formats supportés : `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

---

## Configuration

Le fichier `config/settings.yaml` permet de personnaliser le comportement :

```yaml
detection:
  pitch_threshold: -15    # Seuil de détection (degrés)
                          # Plus négatif = tête plus penchée
  min_duration: 2.0       # Secondes avant alerte
  alert_cooldown: 10.0    # Secondes entre alertes

yolo:
  enabled: true           # Activer détection téléphone
  confidence: 0.5         # Confiance minimum (0-1)

camera:
  index: 0                # Index webcam (0 = défaut)
  width: 640
  height: 480
  fps: 30

alert:
  volume: 0.7             # Volume relatif (0-1)
  num_images: 4           # Nombre de memes affichés
  display_duration: 5     # Durée affichage (secondes)

logging:
  enabled: true           # Sauvegarder les stats
```

### Calibration

La calibration mesure tes angles de tête typiques pour ajuster `pitch_threshold` :

1. Appuie sur `c` dans l'application
2. Regarde ton écran normalement (3 secondes)
3. Regarde ton téléphone (3 secondes)
4. Note le seuil suggéré et modifie `config/settings.yaml`

---

## Structure du projet

```
projet-reco-faciale/
│
├── main.py                    # Point d'entrée principal
├── setup.sh                   # Script d'installation
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
│
├── config/
│   ├── settings.yaml          # Configuration utilisateur
│   └── face_landmarker.task   # Modèle MediaPipe (téléchargé)
│
├── src/
│   ├── __init__.py
│   ├── head_pose.py           # Estimation pose 3D de la tête
│   ├── phone_detector.py      # Détection téléphone (YOLO)
│   ├── alert_system.py        # Alertes son + images
│   └── stats_logger.py        # Logging statistiques
│
├── memes/                     # Tes images/memes d'alerte
│   └── .gitkeep
│
├── logs/                      # Statistiques JSON par session
│
└── venv/                      # Environnement virtuel Python
```

### Flux de données

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Webcam    │───→│  MediaPipe   │───→│  Head Pose  │
│  (OpenCV)   │    │ Face Lander  │    │  Estimation │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│   YOLOv8    │───→│   Phone      │           │
│  Detector   │    │  Detected?   │           │
└─────────────┘    └──────┬───────┘           │
                          │                   │
                          ▼                   ▼
                   ┌──────────────────────────────┐
                   │     Logique de décision      │
                   │  pitch < seuil OU téléphone  │
                   └──────────────┬───────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌────────────┐              ┌────────────┐
             │   Alerte   │              │   Stats    │
             │ Son + Meme │              │  Logging   │
             └────────────┘              └────────────┘
```

---


## Ressources

### Papiers et documentation

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [OpenCV solvePnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Head Pose Estimation (learnopencv.com)](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)

### Datasets pour fine-tuning

- [COCO Dataset](https://cocodataset.org/) - Détection d'objets
- [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) - Head pose estimation
- [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) - Gaze estimation

---

## Licence

Projet personnel à but éducatif.


