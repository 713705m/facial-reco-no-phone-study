"""
Détection de téléphone avec YOLOv8.
Utilise le modèle pré-entraîné sur COCO qui inclut la classe 'cell phone'.
"""
from ultralytics import YOLO
import numpy as np


class PhoneDetector:
    """Détecteur de téléphone basé sur YOLOv8."""

    def __init__(self, confidence: float = 0.5, phone_class_id: int = 67):
        """
        Args:
            confidence: Seuil de confiance minimum
            phone_class_id: ID de la classe téléphone dans COCO (67)
        """
        self.confidence = confidence
        self.phone_class_id = phone_class_id
        self.model = None
        self._loaded = False

    def load(self):
        """Charge le modèle YOLOv8 (lazy loading)."""
        if not self._loaded:
            print("Chargement du modèle YOLO...")
            # yolov8n = version nano, rapide et légère
            self.model = YOLO('yolov8n.pt')
            self._loaded = True
            print("Modèle YOLO chargé!")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Détecte les téléphones dans l'image.

        Args:
            frame: Image BGR OpenCV

        Returns:
            Liste de détections avec bbox et confiance
        """
        if not self._loaded:
            self.load()

        results = self.model(frame, verbose=False)[0]
        phones = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == self.phone_class_id and conf >= self.confidence:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                phones.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf
                })

        return phones

    def draw_detections(self, frame: np.ndarray, detections: list[dict]):
        """Dessine les bbox des téléphones détectés."""
        import cv2
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
