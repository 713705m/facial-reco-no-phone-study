"""
Estimation de la pose de la tete via la matrice de transformation MediaPipe FaceLandmarker.

Comment ca marche :
------------------
MediaPipe FaceLandmarker detecte 478 landmarks sur le visage et calcule en interne
une matrice de transformation 4x4 qui decrit exactement comment le visage est oriente
dans l'espace 3D par rapport a la camera.

Cette matrice transforme du "canonical face space" vers le "camera space" :
  - Canonical face space : le visage dans une position neutre standard (modele 3D interne)
  - Camera space : le systeme de coordonnees de la camera (X droite, Y bas, Z vers l'ecran)

La matrice 4x4 a la forme :
  [ R  | t ]   R = matrice de rotation 3x3
  [ 0  | 1 ]   t = vecteur de translation 3x1

On extrait R et on calcule les angles d'Euler (ZYX convention) :
  - yaw  : rotation autour de Y = tourner la tete gauche/droite
  - pitch : rotation autour de X = incliner la tete haut/bas
  - roll  : rotation autour de Z = pencher la tete sur le cote

Pourquoi c'est mieux que solvePnP avec 6 points :
  - MediaPipe utilise 478 landmarks -> beaucoup plus stable
  - Son modele 3D est ajuste pour chaque visage -> pas de biais "visage generique"
  - La matrice est directement issue du pipeline de reconstruction 3D de MediaPipe
"""
import numpy as np
import cv2


def get_head_pose(transform_matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Calcule les angles de rotation depuis la matrice 4x4 de MediaPipe.

    Args:
        transform_matrix: Matrice 4x4 numpy (canonical face space -> camera space)

    Returns:
        Tuple (pitch, yaw, roll) en degres
        - pitch: + = regard vers le haut, - = vers le bas
        - yaw: tourne gauche/droite
        - roll: inclinaison laterale
    """
    R = transform_matrix[:3, :3]

    # Decomposition ZYX des angles d'Euler depuis la matrice de rotation
    # sy = norme de la premiere colonne (indicateur de gimbal lock si proche de 0)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        # Yaw : rotation autour de Y = tournement gauche/droite
        yaw = np.arctan2(-R[2, 0], sy)

        # Pitch : rotation autour de X = regard haut/bas
        # Avec la matrice MediaPipe, x_rot est proche de 0° face camera
        # Le signe est inverse : x_rot positif = nez vers le bas, negatif = nez vers le haut
        # On inverse pour avoir pitch positif = regard vers le haut
        pitch = -np.arctan2(R[2, 1], R[2, 2])

        # Roll : rotation autour de Z = inclinaison laterale
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock (tete pointant directement vers le haut ou le bas)
        x_rot = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.sign(x_rot) * (np.abs(x_rot) - np.pi)
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0

    # Conversion en float Python (evite les float32 numpy non serialisables en JSON)
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def draw_pose_info(frame, pitch: float, yaw: float, roll: float, is_distracted: bool):
    """Affiche les informations de pose sur l'image."""
    color = (0, 0, 255) if is_distracted else (0, 255, 0)
    status = "DISTRACTED!" if is_distracted else "Focused"

    cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Status: {status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
