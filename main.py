#!/usr/bin/env python3
"""
Detecteur de Distraction Telephone
===================================
Active ta webcam et t'alerte quand tu regardes ton telephone au lieu de travailler.

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

from src.head_pose import get_head_pose
from src.phone_detector import PhoneDetector
from src.alert_system import AlertSystem
from src.stats_logger import StatsLogger
from src.ui_renderer import ModernUI


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Charge la configuration depuis le fichier YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_face_landmarker_model():
    """Telecharge le modele Face Landmarker si necessaire."""
    import urllib.request
    model_path = Path(__file__).parent / "config" / "face_landmarker.task"

    if not model_path.exists():
        print("Telechargement du modele Face Landmarker...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)

    return str(model_path)



class DistractionDetector:
    """Classe principale du detecteur de distraction."""

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
            output_facial_transformation_matrixes=True
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
        self.last_focus_alert_time = 0
        self.is_distracted = False
        self.session_distractions = 0

        # Interface moderne
        self.ui = ModernUI(
            width=config['camera']['width'],
            height=config['camera']['height']
        )

    def run(self):
        """Boucle principale de détection."""
        cam_config = self.config['camera']
        cap = cv2.VideoCapture(cam_config['index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config['height'])
        cap.set(cv2.CAP_PROP_FPS, cam_config['fps'])

        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la camera")
            return

        print("\n" + "=" * 50)
        print("   Focus Guardian - ACTIF")
        print("=" * 50)
        print("  [Q] Quitter")
        print("  [C] Calibrer la position focus")
        print("=" * 50)

        # Afficher l'etat de calibration
        if self.config['detection'].get('calibrated', False):
            ph = self.config['detection'].get('pitch_max_haut', 0)
            pb = self.config['detection'].get('pitch_max_bas', 0)
            yg = self.config['detection'].get('yaw_max_gauche', 0)
            yd = self.config['detection'].get('yaw_max_droite', 0)
            print(f"  Calibre : OUI")
            print(f"  Zone : [{pb:.0f} a {ph:.0f}] x [{yg:.0f} a {yd:.0f}]")
        else:
            print(f"  Calibre : NON (appuie sur [C])")

        print(f"  Alarme après : {self.config['detection']['min_duration']}s")
        print("=" * 50)
        print(f"  Memes focus: {len(self.alert_system.images_focus)}")
        print(f"  Memes telephone: {len(self.alert_system.images_telephone)}")
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

                if results.face_landmarks and len(results.face_landmarks) > 0 and results.facial_transformation_matrixes:
                    face_detected = True
                    transform_matrix = np.array(results.facial_transformation_matrixes[0])
                    pitch, yaw, roll = get_head_pose(transform_matrix)

                # Détection téléphone avec YOLO (indépendant du visage)
                phone_detected = False
                if self.phone_detector:
                    if not yolo_loaded:
                        self.phone_detector.load()
                        yolo_loaded = True

                    phones = self.phone_detector.detect(frame)
                    phone_detected = len(phones) > 0
                    self.phone_detector.draw_detections(frame, phones)

                # Logique de détection de distraction
                current_time = time.time()
                detection_config = self.config['detection']

                # Verifier si hors de la zone focus calibree (4 limites)
                out_of_focus_zone = False
                if face_detected and detection_config.get('calibrated', False):
                    # Recuperer les 4 limites calibrees (valeurs brutes)
                    pitch_haut = detection_config.get('pitch_max_haut', 20)
                    pitch_bas = detection_config.get('pitch_max_bas', -20)
                    yaw_gauche = detection_config.get('yaw_max_gauche', -30)
                    yaw_droite = detection_config.get('yaw_max_droite', 30)

                    # Calculer min/max automatiquement (peu importe l'ordre de calibration)
                    MARGE = 10.0
                    pitch_min = min(pitch_haut, pitch_bas) - MARGE
                    pitch_max = max(pitch_haut, pitch_bas) + MARGE
                    yaw_min = min(yaw_gauche, yaw_droite) - MARGE
                    yaw_max = max(yaw_gauche, yaw_droite) + MARGE

                    # DANS la zone focus si : pitch_min <= pitch <= pitch_max ET yaw_min <= yaw <= yaw_max
                    # HORS zone si en dehors de ces limites
                    hors_pitch = pitch < pitch_min or pitch > pitch_max
                    hors_yaw = yaw < yaw_min or yaw > yaw_max
                    out_of_focus_zone = hors_pitch or hors_yaw
                elif face_detected:
                    # Pas calibre : utiliser des seuils par defaut
                    out_of_focus_zone = pitch < -20 or pitch > 20 or abs(yaw) > 30

                is_currently_distracted = out_of_focus_zone or phone_detected

                distraction_duration = 0
                alert_type = None
                focus_cooldown_ok = (current_time - self.last_focus_alert_time) > detection_config['alert_cooldown']

                # ALARME TELEPHONE : IMMEDIATE, sans cooldown
                if phone_detected and not self.alert_system.is_alert_active():
                    alert_type = "telephone"
                    print(f"ALERTE TELEPHONE DETECTE!")
                    self.alert_system.trigger_alert(alert_type)
                    self.is_distracted = True
                    self.session_distractions += 1
                    self.stats.start_distraction(pitch, phone_detected)

                # ALARME FOCUS : apres min_duration secondes hors zone
                elif out_of_focus_zone and not phone_detected:
                    if self.distraction_start_time is None:
                        self.distraction_start_time = current_time
                        self.stats.start_distraction(pitch, False)

                    distraction_duration = current_time - self.distraction_start_time

                    if distraction_duration >= detection_config['min_duration'] and focus_cooldown_ok:
                        if not self.alert_system.is_alert_active():
                            alert_type = "focus"
                            print(f"ALERTE FOCUS! Hors zone (pitch={pitch:.1f}, yaw={yaw:.1f})")
                            self.alert_system.trigger_alert(alert_type)
                            self.last_focus_alert_time = current_time
                            self.is_distracted = True
                            self.session_distractions += 1

                # Pas de distraction
                elif not out_of_focus_zone and not phone_detected:
                    if self.distraction_start_time is not None:
                        self.stats.end_distraction()
                    self.distraction_start_time = None
                    self.is_distracted = False
                    alert_type = None

                # Préparer les données pour l'interface
                ui_data = {
                    'is_distracted': self.is_distracted,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'phone_detected': phone_detected,
                    'face_detected': face_detected,
                    'distraction_duration': distraction_duration,
                    'alert_threshold': detection_config['min_duration'],
                    'session_distractions': self.session_distractions,
                    'alert_type': alert_type,
                }

                # Créer l'interface moderne
                display_frame = self.ui.create_frame(frame, ui_data)
                cv2.imshow("Focus Guardian", display_frame)

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

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_landmarker.close()
            self.stats.save()
            self.stats.print_summary()

    def _calibrate(self, cap):
        """Mode calibration - capture les 4 limites de la zone focus."""
        positions = [
            ("HAUT", "Regarde le HAUT de ton ecran", "pitch_max_haut", (0, 255, 255)),
            ("BAS", "Regarde le BAS de ton ecran/clavier", "pitch_max_bas", (255, 255, 0)),
            ("GAUCHE", "Tourne la tete a GAUCHE (max)", "yaw_max_gauche", (255, 0, 255)),
            ("DROITE", "Tourne la tete a DROITE (max)", "yaw_max_droite", (0, 255, 0)),
        ]

        calibration_data = {}

        print("\n" + "=" * 50)
        print("   CALIBRATION - ZONE DE FOCUS")
        print("=" * 50)
        print("  Tu vas definir les 4 limites de ta zone de travail")
        print("  Pour chaque position, appuie sur [ESPACE] pour capturer")
        print("  Appuie sur [Q] ou [ESC] pour annuler")
        print("=" * 50 + "\n")

        for pos_name, instruction, config_key, color in positions:
            print(f"\n  >>> Position {pos_name}")
            print(f"      {instruction}")

            # Attendre que l'utilisateur appuie sur ESPACE
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Détecter le visage pour afficher les angles actuels
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self.frame_timestamp_ms += 33
                results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

                current_pitch, current_yaw = 0, 0
                if results.face_landmarks and len(results.face_landmarks) > 0 and results.facial_transformation_matrixes:
                    transform_matrix = np.array(results.facial_transformation_matrixes[0])
                    current_pitch, current_yaw, _ = get_head_pose(transform_matrix)

                # Afficher les instructions sur la vidéo
                cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
                cv2.putText(frame, f"CALIBRATION: {pos_name}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, instruction, (20, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Afficher les angles actuels
                cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1)
                cv2.putText(frame, f"Inclinaison: {current_pitch:.1f}", (20, h-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Rotation: {current_yaw:.1f}", (250, h-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "[ESPACE] Capturer", (w-200, h-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Focus Guardian", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # ESPACE - capturer
                    # Capturer sur 1 seconde (~30 frames) et prendre la mediane
                    measurements = []
                    for _ in range(30):  # ~1 sec a 30fps
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        self.frame_timestamp_ms += 33
                        results = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
                        if results.face_landmarks and len(results.face_landmarks) > 0 and results.facial_transformation_matrixes:
                            transform_matrix = np.array(results.facial_transformation_matrixes[0])
                            p, y, _ = get_head_pose(transform_matrix)
                            measurements.append((p, y))
                        cv2.waitKey(33)

                    if measurements:
                        # Mediane au lieu de moyenne (robuste aux outliers)
                        pitches = sorted([m[0] for m in measurements])
                        yaws = sorted([m[1] for m in measurements])
                        median_pitch = pitches[len(pitches) // 2]
                        median_yaw = yaws[len(yaws) // 2]

                        # Stocker la bonne valeur selon la position
                        if "pitch" in config_key:
                            calibration_data[config_key] = round(median_pitch, 1)
                            print(f"      Capture (mediane sur {len(measurements)} mesures): {median_pitch:.1f}")
                        else:
                            calibration_data[config_key] = round(median_yaw, 1)
                            print(f"      Capture (mediane sur {len(measurements)} mesures): {median_yaw:.1f}")
                        break
                    else:
                        print("      Visage non detecte, reessaie !")

                elif key == ord('q') or key == 27:  # Q ou ESC
                    print("\n  Calibration annulee.")
                    return

        # Sauvegarder les valeurs brutes (la marge sera appliquee lors de la detection)
        if len(calibration_data) == 4:
            self._save_calibration_4pos(calibration_data)

            print("\n" + "=" * 50)
            print("   CALIBRATION REUSSIE !")
            print("=" * 50)
            print("  Zone de focus definie (valeurs brutes) :")
            print(f"    HAUT    : {calibration_data['pitch_max_haut']:.1f}")
            print(f"    BAS     : {calibration_data['pitch_max_bas']:.1f}")
            print(f"    GAUCHE  : {calibration_data['yaw_max_gauche']:.1f}")
            print(f"    DROITE  : {calibration_data['yaw_max_droite']:.1f}")
            print(f"\n  Marge de tolerance : 5 degres")
            print(f"  Alarme apres : {self.config['detection']['min_duration']}s hors zone")
            print("=" * 50 + "\n")

            # Mettre a jour la config en memoire
            self.config['detection'].update(calibration_data)
            self.config['detection']['calibrated'] = True
        else:
            print("\n  Calibration incomplete !")

    def _save_calibration_4pos(self, data: dict):
        """Sauvegarde les 4 positions de calibration."""
        config_path = Path(__file__).parent / "config" / "settings.yaml"
        try:
            with open(config_path, 'r') as f:
                content = f.read()

            import re
            content = re.sub(r'pitch_max_haut:.*', f"pitch_max_haut: {data['pitch_max_haut']}", content)
            content = re.sub(r'pitch_max_bas:.*', f"pitch_max_bas: {data['pitch_max_bas']}", content)
            content = re.sub(r'yaw_max_gauche:.*', f"yaw_max_gauche: {data['yaw_max_gauche']}", content)
            content = re.sub(r'yaw_max_droite:.*', f"yaw_max_droite: {data['yaw_max_droite']}", content)
            content = re.sub(r'calibrated:.*', 'calibrated: true', content)

            with open(config_path, 'w') as f:
                f.write(content)

            print("  Config sauvegardee !")
        except Exception as e:
            print(f"  Erreur sauvegarde: {e}")


def show_stats():
    """Affiche les statistiques historiques."""
    project_root = Path(__file__).parent
    logger = StatsLogger(str(project_root / "logs"))
    logger.print_weekly_report()


def main():
    parser = argparse.ArgumentParser(description="Detecteur de distraction telephone")
    parser.add_argument('--calibrate', action='store_true', help="Lancer la calibration au demarrage")
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
