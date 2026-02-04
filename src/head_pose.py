"""
Estimation de la pose de la tête à partir des landmarks MediaPipe.
Calcule pitch (haut/bas), yaw (gauche/droite), roll (inclinaison).
"""
import numpy as np
import cv2


# Points 3D de référence d'un visage générique (en mm)
# Basé sur un modèle anthropométrique standard
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nez (tip)
    (0.0, -330.0, -65.0),        # Menton
    (-225.0, 170.0, -135.0),     # Coin œil gauche
    (225.0, 170.0, -135.0),      # Coin œil droit
    (-150.0, -150.0, -125.0),    # Coin bouche gauche
    (150.0, -150.0, -125.0)      # Coin bouche droit
], dtype=np.float64)

# Indices des landmarks MediaPipe correspondants
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def get_head_pose(landmarks, frame_width: int, frame_height: int) -> tuple[float, float, float]:
    """
    Calcule les angles de rotation de la tête.

    Args:
        landmarks: Wrapper contenant landmarks MediaPipe (liste d'objets avec x, y, z)
        frame_width: Largeur de l'image
        frame_height: Hauteur de l'image

    Returns:
        Tuple (pitch, yaw, roll) en degrés
        - pitch: positif = regarde vers le haut, négatif = vers le bas
        - yaw: positif = tourne à droite, négatif = à gauche
        - roll: inclinaison latérale de la tête
    """
    # Extraire les points 2D des landmarks
    image_points = []
    landmark_list = landmarks.landmark

    for idx in LANDMARK_INDICES:
        lm = landmark_list[idx]
        # MediaPipe Tasks retourne des coordonnées normalisées (0-1)
        x = lm.x * frame_width
        y = lm.y * frame_height
        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    # Matrice de caméra approximative
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Pas de distorsion
    dist_coeffs = np.zeros((4, 1))

    # Résoudre PnP pour trouver la rotation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convertir vecteur de rotation en matrice
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extraire les angles d'Euler
    # Utilise la décomposition de la matrice de rotation
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    if sy > 1e-6:
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        roll = 0

    # Convertir en degrés
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll


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
