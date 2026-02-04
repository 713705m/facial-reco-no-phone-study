#!/usr/bin/env python3
"""
Détecteur de Distraction Téléphone
===================================
Active ta webcam et t'alerte quand tu regardes ton téléphone au lieu de travailler.

Usage:
    python main.py              # Mode normal
    python main.py --calibrate  # Mode calibration
    python main.py --stats      # Voir les statistiques
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# MediaPipe Tasks API (nouvelle version)
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.head_pose import get_head_pose, draw_pose_info
from src.phone_detector import PhoneDetector
from src.alert_system import AlertSystem
from src.stats_logger import StatsLogger


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Charge la configuration depuis le fichier YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_face_landmarker_model():
    """Télécharge le modèle Face Landmarker si nécessaire."""
    import urllib.request
    model_path = Path(__file__).parent / "config" / "face_landmarker.task"

    if not model_path.exists():
        print("Téléchargement du modèle Face Landmarker...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Modèle téléchargé!")

    return str(model_path)


class FaceLandmarksWrapper:
    """Wrapper pour adapter les landmarks MediaPipe Tasks au format attendu par head_pose."""

    def __init__(self, landmarks_list):
        self.landmark = landmarks_list


class DistractionDetector:
    """Classe principale du détecteur de distraction."""

    def __init__(self, config: dict):
        self.config = config

        # Télécharger le modèle si nécessaire
        model_path = download_face_landmarker_model()

        # MediaPipe Face Landmarker (nouvelle API)
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Détecteur de téléphone (YOLO)
        self.phone_detector = None
        if config['yolo']['enabled']:
            self.phone_detector = PhoneDetector(
                confidence=config['yolo']['confidence'],
                phone_class_id=config['yolo']['phone_class_id']
            )

        # Système d'alerte
        project_root = Path(__file__).parent
        self.alert_system = AlertSystem(
            memes_dir=str(project_root / "memes"),
            volume=config['alert']['volume'],
            num_images=config['alert']['num_images'],
            display_duration=config['alert']['display_duration']
        )

        # Logger de stats
        self.stats = StatsLogger(
            log_dir=str(project_root / "logs"),
            enabled=config['logging']['enabled']
        )

        # État de détection
        self.distraction_start_time = None
        self.last_alert_time = 0
        self.is_distracted = False

    def run(self):
        """Boucle principale de détection."""
        cam_config = self.config['camera']
        cap = cv2.VideoCapture(cam_config['index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
        cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])

        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra")
            return

        print("\n" + "=" * 50)
        print("DÉTECTEUR DE DISTRACTION ACTIF")
        print("=" * 50)
        print("Appuie sur 'q' pour quitter")
        print("Appuie sur 'c' pour calibrer")
        print("Appuie sur 'r' pour rafraîchir les memes")
        print("=" * 50 + "\n")

        # Charger YOLO en lazy loading au premier besoin
        yolo_loaded = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Miroir
                h, w = frame.shape[:2]

                # Convertir en format MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Incrémenter le timestamp
                self.frame_timestamp_ms += 33  # ~30fps

                # Détection visage avec MediaPipe
                results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

                pitch, yaw, roll = 0, 0, 0
                face_detected = False

                if results.face_landmarks and len(results.face_landmarks) > 0:
                    face_detected = True
                    # Wrapper pour adapter au format attendu par head_pose
                    landmarks = FaceLandmarksWrapper(results.face_landmarks[0])

                    # Calculer la pose de la tête
                    pitch, yaw, roll = get_head_pose(landmarks, w, h)

                # Détection téléphone avec YOLO
                phone_detected = False
                if self.phone_detector and face_detected:
                    if not yolo_loaded:
                        self.phone_detector.load()
                        yolo_loaded = True

                    phones = self.phone_detector.detect(frame)
                    phone_detected = len(phones) > 0
                    self.phone_detector.draw_detections(frame, phones)

                # Logique de détection de distraction
                current_time = time.time()
                detection_config = self.config['detection']

                # Critères: tête penchée vers le bas OU téléphone détecté
                looking_down = pitch < detection_config['pitch_threshold']
                is_currently_distracted = looking_down or phone_detected

                if is_currently_distracted:
                    if self.distraction_start_time is None:
                        self.distraction_start_time = current_time
                        self.stats.start_distraction(pitch, phone_detected)

                    # Vérifier si la distraction dure assez longtemps
                    distraction_duration = current_time - self.distraction_start_time
                    cooldown_ok = (current_time - self.last_alert_time) > detection_config['alert_cooldown']

                    if distraction_duration >= detection_config['min_duration'] and cooldown_ok:
                        if not self.alert_system.is_alert_active():
                            print(f"ALERTE! Distraction détectée (pitch={pitch:.1f}°, phone={phone_detected})")
                            self.alert_system.trigger_alert()
                            self.last_alert_time = current_time
                            self.is_distracted = True
                else:
                    if self.distraction_start_time is not None:
                        self.stats.end_distraction()
                    self.distraction_start_time = None
                    self.is_distracted = False

                # Affichage
                draw_pose_info(frame, pitch, yaw, roll, self.is_distracted)

                if phone_detected:
                    cv2.putText(frame, "TELEPHONE DETECTE!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if not face_detected:
                    cv2.putText(frame, "Pas de visage detecte", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

                cv2.imshow("Distraction Detector", frame)

                # Afficher les memes si une alerte est déclenchée (doit être dans le main thread)
                self.alert_system.show_images_if_needed()

                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._calibrate(cap)
                elif key == ord('r'):
                    self.alert_system.refresh_images()
                    print("Liste des memes rafraîchie!")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_landmarker.close()
            self.stats.save()
            self.stats.print_summary()

    def _calibrate(self, cap):
        """Mode calibration pour ajuster les seuils."""
        print("\n=== MODE CALIBRATION ===")
        print("Regarde ton écran normalement pendant 3 secondes...")
        time.sleep(1)

        pitches_normal = []
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp_ms += 33
            results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmarks = FaceLandmarksWrapper(results.face_landmarks[0])
                p, _, _ = get_head_pose(landmarks, w, h)
                pitches_normal.append(p)
            cv2.waitKey(33)

        print("Maintenant regarde ton téléphone pendant 3 secondes...")
        time.sleep(1)

        pitches_phone = []
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp_ms += 33
            results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmarks = FaceLandmarksWrapper(results.face_landmarks[0])
                p, _, _ = get_head_pose(landmarks, w, h)
                pitches_phone.append(p)
            cv2.waitKey(33)

        if pitches_normal and pitches_phone:
            avg_normal = sum(pitches_normal) / len(pitches_normal)
            avg_phone = sum(pitches_phone) / len(pitches_phone)
            suggested_threshold = (avg_normal + avg_phone) / 2

            print(f"\nRésultats calibration:")
            print(f"  Pitch moyen (écran): {avg_normal:.1f}°")
            print(f"  Pitch moyen (téléphone): {avg_phone:.1f}°")
            print(f"  Seuil suggéré: {suggested_threshold:.1f}°")
            print(f"\nModifie 'pitch_threshold' dans config/settings.yaml")
        else:
            print("Calibration échouée - visage non détecté")

        print("=== FIN CALIBRATION ===\n")


def show_stats():
    """Affiche les statistiques historiques."""
    project_root = Path(__file__).parent
    logger = StatsLogger(str(project_root / "logs"))
    logger.print_weekly_report()


def main():
    parser = argparse.ArgumentParser(description="Détecteur de distraction téléphone")
    parser.add_argument('--calibrate', action='store_true', help="Lancer la calibration au démarrage")
    parser.add_argument('--stats', action='store_true', help="Afficher les statistiques")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    config = load_config()
    detector = DistractionDetector(config)

    if args.calibrate:
        print("Mode calibration activé - sera lancé au démarrage")

    detector.run()


if __name__ == "__main__":
    main()
