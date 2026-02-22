"""
Interface utilisateur moderne pour le detecteur de distraction.
Design minimaliste avec polices modernes et textes en francais.
"""
import cv2
import numpy as np
import time
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class ModernUI:
    """Renderer d'interface moderne avec polices custom."""

    # Couleurs (BGR pour OpenCV)
    COLORS = {
        'bg_dark': (20, 20, 25),
        'bg_card': (35, 35, 40),
        'accent_green': (100, 220, 120),
        'accent_red': (80, 80, 255),
        'accent_orange': (60, 165, 255),
        'accent_blue': (255, 180, 100),
        'accent_purple': (200, 100, 180),
        'text_white': (255, 255, 255),
        'text_gray': (150, 150, 150),
        'text_dark': (80, 80, 80),
    }

    # Couleurs RGB pour PIL
    COLORS_RGB = {k: (v[2], v[1], v[0]) for k, v in COLORS.items()}

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.animation_time = 0
        self.pulse_phase = 0

        # Charger les polices système
        self._load_fonts()

    def _load_fonts(self):
        """Charge les polices systeme."""
        # Polices macOS par defaut
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]

        self.font_regular = None
        self.font_bold = None

        for path in font_paths:
            if Path(path).exists():
                try:
                    self.font_regular = ImageFont.truetype(path, 16)
                    self.font_bold = ImageFont.truetype(path, 20)
                    self.font_title = ImageFont.truetype(path, 28)
                    self.font_large = ImageFont.truetype(path, 48)
                    break
                except Exception:
                    continue

        # Fallback si aucune police trouvee
        if self.font_regular is None:
            self.font_regular = ImageFont.load_default()
            self.font_bold = self.font_regular
            self.font_title = self.font_regular
            self.font_large = self.font_regular

    def _draw_text_pil(self, canvas: np.ndarray, text: str, pos: tuple,
                       color_rgb: tuple, font) -> np.ndarray:
        """Dessine du texte avec PIL pour de meilleures polices."""
        # Convertir OpenCV (BGR) -> PIL (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=color_rgb)
        # Reconvertir PIL -> OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def create_frame(self, video_frame: np.ndarray, data: dict) -> np.ndarray:
        """
        Cree une frame complete avec l'interface moderne.

        Args:
            video_frame: Image de la webcam
            data: Dictionnaire avec les donnees a afficher
        """
        video_h, video_w = video_frame.shape[:2]

        # Creer le canvas principal
        canvas_w = video_w + 300
        canvas_h = video_h + 100
        canvas = np.full((canvas_h, canvas_w, 3), self.COLORS['bg_dark'], dtype=np.uint8)

        # Ajouter la video avec cadre
        self._draw_video_panel(canvas, video_frame, data)

        # Panel de statut à droite
        canvas = self._draw_status_panel(canvas, video_w, data)

        # Barre inférieure
        canvas = self._draw_bottom_bar(canvas, video_h, data)

        # Overlay d'alerte si distrait
        if data.get('is_distracted', False):
            canvas = self._draw_alert_overlay(canvas, video_frame, data)

        self.animation_time += 0.05
        return canvas

    def _draw_video_panel(self, canvas: np.ndarray, video: np.ndarray, data: dict):
        """Dessine le panel video avec cadre."""
        h, w = video.shape[:2]

        # Couleur du cadre selon le type d'alerte
        alert_type = data.get('alert_type', None)
        if alert_type == 'telephone':
            border_color = self.COLORS['accent_orange']
            border_thickness = 4
        elif alert_type == 'focus' or data.get('is_distracted', False):
            border_color = self.COLORS['accent_red']
            border_thickness = 4
        elif not data.get('face_detected', True):
            border_color = self.COLORS['text_gray']
            border_thickness = 2
        else:
            border_color = self.COLORS['accent_green']
            border_thickness = 2

        # Placer la vidéo
        x_offset, y_offset = 15, 15
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = video

        # Cadre
        cv2.rectangle(canvas, (x_offset-2, y_offset-2),
                     (x_offset+w+2, y_offset+h+2), border_color, border_thickness)

        # Effet de glow si alerte
        if data.get('is_distracted', False):
            self._add_glow_effect(canvas, x_offset, y_offset, w, h, border_color)

    def _add_glow_effect(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, color: tuple):
        """Ajoute un effet de lueur pulsante."""
        pulse = abs(math.sin(self.animation_time * 3)) * 0.5 + 0.5
        alpha = 0.3 * pulse

        overlay = canvas.copy()
        for i in range(3):
            offset = (i + 1) * 3
            cv2.rectangle(overlay, (x - offset, y - offset),
                         (x + w + offset, y + h + offset), color, 2)

        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    def _draw_status_panel(self, canvas: np.ndarray, x_start: int, data: dict) -> np.ndarray:
        """Dessine le panel de statut a droite."""
        panel_x = x_start + 30
        panel_y = 15
        panel_w = 255
        panel_h = 450

        # Fond du panel
        cv2.rectangle(canvas, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     self.COLORS['bg_card'], -1)
        cv2.rectangle(canvas, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     self.COLORS['text_dark'], 1)

        # Titre
        canvas = self._draw_text_pil(canvas, "STATUT", (panel_x + 20, panel_y + 15),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        # Indicateur principal
        status_y = panel_y + 55
        alert_type = data.get('alert_type', None)

        if alert_type == 'telephone':
            status_text = "TELEPHONE !"
            status_color = self.COLORS['accent_orange']
            status_color_rgb = self.COLORS_RGB['accent_orange']
        elif alert_type == 'focus' or data.get('is_distracted', False):
            status_text = "DISTRAIT !"
            status_color = self.COLORS['accent_red']
            status_color_rgb = self.COLORS_RGB['accent_red']
        elif not data.get('face_detected', True):
            status_text = "PAS DE VISAGE"
            status_color = self.COLORS['text_gray']
            status_color_rgb = self.COLORS_RGB['text_gray']
        else:
            status_text = "CONCENTRE"
            status_color = self.COLORS['accent_green']
            status_color_rgb = self.COLORS_RGB['accent_green']

        # Cercle d'etat avec animation
        circle_x = panel_x + 35
        circle_y = status_y + 25
        radius = 18

        if data.get('is_distracted', False):
            pulse = abs(math.sin(self.animation_time * 4))
            animated_radius = int(radius + pulse * 5)
            cv2.circle(canvas, (circle_x, circle_y), animated_radius, status_color, -1)
            cv2.circle(canvas, (circle_x, circle_y), animated_radius + 3, status_color, 2)
        else:
            cv2.circle(canvas, (circle_x, circle_y), radius, status_color, -1)

        canvas = self._draw_text_pil(canvas, status_text, (circle_x + 30, circle_y - 12),
                                     status_color_rgb, self.font_bold)

        # Metriques
        metrics_y = status_y + 75

        # Pitch
        canvas = self._draw_text_pil(canvas, "INCLINAISON", (panel_x + 20, metrics_y),
                                     self.COLORS_RGB['text_gray'], self.font_regular)
        pitch_color = self._get_pitch_color_rgb(data.get('pitch', 0))
        canvas = self._draw_text_pil(canvas, f"{data.get('pitch', 0):.1f}°",
                                     (panel_x + 20, metrics_y + 22), pitch_color, self.font_bold)
        cv2.line(canvas, (panel_x + 20, metrics_y + 50),
                (panel_x + 120, metrics_y + 50), self.COLORS['text_dark'], 1)

        # Yaw
        canvas = self._draw_text_pil(canvas, "ROTATION", (panel_x + 20, metrics_y + 65),
                                     self.COLORS_RGB['text_gray'], self.font_regular)
        canvas = self._draw_text_pil(canvas, f"{data.get('yaw', 0):.1f}°",
                                     (panel_x + 20, metrics_y + 87),
                                     self.COLORS_RGB['accent_blue'], self.font_bold)
        cv2.line(canvas, (panel_x + 20, metrics_y + 115),
                (panel_x + 120, metrics_y + 115), self.COLORS['text_dark'], 1)

        # Detection telephone
        phone_y = metrics_y + 135
        phone_detected = data.get('phone_detected', False)
        canvas = self._draw_text_pil(canvas, "TELEPHONE", (panel_x + 20, phone_y),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        phone_status = "DETECTE !" if phone_detected else "NON DETECTE"
        phone_color = self.COLORS_RGB['accent_orange'] if phone_detected else self.COLORS_RGB['accent_green']
        canvas = self._draw_text_pil(canvas, phone_status, (panel_x + 20, phone_y + 22),
                                     phone_color, self.font_bold)

        # Barre de progression
        if data.get('distraction_duration', 0) > 0:
            canvas = self._draw_distraction_bar(canvas, panel_x + 20, phone_y + 60,
                                                panel_w - 40, data)

        # Instructions
        help_y = panel_y + panel_h - 95
        canvas = self._draw_text_pil(canvas, "CONTROLES", (panel_x + 20, help_y),
                                     self.COLORS_RGB['text_gray'], self.font_regular)
        canvas = self._draw_text_pil(canvas, "[Q] Quitter", (panel_x + 20, help_y + 22),
                                     self.COLORS_RGB['text_gray'], self.font_regular)
        canvas = self._draw_text_pil(canvas, "[C] Calibrer", (panel_x + 20, help_y + 42),
                                     self.COLORS_RGB['text_gray'], self.font_regular)
        canvas = self._draw_text_pil(canvas, "[R] Rafraichir memes", (panel_x + 20, help_y + 62),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        return canvas

    def _draw_distraction_bar(self, canvas: np.ndarray, x: int, y: int,
                              width: int, data: dict) -> np.ndarray:
        """Dessine une barre de progression avant l'alerte (focus seulement)."""
        duration = data.get('distraction_duration', 0)
        threshold = data.get('alert_threshold', 2.0)
        progress = min(duration / threshold, 1.0)

        canvas = self._draw_text_pil(canvas, "ALERTE DANS", (x, y),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        # Fond de la barre
        bar_y = y + 25
        bar_h = 8
        cv2.rectangle(canvas, (x, bar_y), (x + width, bar_y + bar_h),
                     self.COLORS['text_dark'], -1)

        # Progression avec gradient
        fill_w = int(width * progress)
        if fill_w > 0:
            r = int(80 + 175 * progress)
            g = int(220 - 140 * progress)
            b = int(120 - 40 * progress)
            cv2.rectangle(canvas, (x, bar_y), (x + fill_w, bar_y + bar_h),
                         (b, g, r), -1)

        # Temps restant
        remaining = max(0, threshold - duration)
        canvas = self._draw_text_pil(canvas, f"{remaining:.1f}s", (x + width - 35, y),
                                     self.COLORS_RGB['text_white'], self.font_regular)

        return canvas

    def _draw_bottom_bar(self, canvas: np.ndarray, y_start: int, data: dict) -> np.ndarray:
        """Dessine la barre inferieure."""
        bar_y = y_start + 30
        bar_h = 55

        # Fond
        cv2.rectangle(canvas, (15, bar_y), (canvas.shape[1] - 15, bar_y + bar_h),
                     self.COLORS['bg_card'], -1)

        # Titre
        canvas = self._draw_text_pil(canvas, "GARDIEN DE FOCUS", (30, bar_y + 10),
                                     self.COLORS_RGB['accent_blue'], self.font_bold)

        # Heure
        from datetime import datetime
        time_str = datetime.now().strftime("%H:%M:%S")
        canvas = self._draw_text_pil(canvas, time_str, (30, bar_y + 32),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        # Stats de session
        session_distractions = data.get('session_distractions', 0)
        canvas = self._draw_text_pil(canvas, f"Distractions : {session_distractions}",
                                     (220, bar_y + 22),
                                     self.COLORS_RGB['text_gray'], self.font_regular)

        return canvas

    def _draw_alert_overlay(self, canvas: np.ndarray, video: np.ndarray, data: dict) -> np.ndarray:
        """Dessine un overlay d'alerte."""
        h, w = video.shape[:2]
        x_offset, y_offset = 15, 15

        alert_type = data.get('alert_type', 'focus')

        # Couleur selon le type
        if alert_type == 'telephone':
            overlay_color = self.COLORS['accent_orange']
        else:
            overlay_color = self.COLORS['accent_red']

        # Overlay semi-transparent
        pulse = abs(math.sin(self.animation_time * 5)) * 0.3 + 0.1
        colored_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        colored_overlay[:, :] = overlay_color

        video_region = canvas[y_offset:y_offset+h, x_offset:x_offset+w]
        blended = cv2.addWeighted(video_region, 1 - pulse, colored_overlay, pulse, 0)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = blended

        return canvas

    def _get_pitch_color_rgb(self, pitch: float) -> tuple:
        """Retourne une couleur RGB basee sur l'angle pitch."""
        if pitch < -20:
            return self.COLORS_RGB['accent_red']
        elif pitch < -10:
            return self.COLORS_RGB['accent_orange']
        else:
            return self.COLORS_RGB['accent_green']


def create_modern_meme_display(images: list, message: str = "RETOURNE BOSSER !",
                               alert_type: str = "focus") -> np.ndarray:
    """
    Cree un affichage moderne pour les memes d'alerte.

    Args:
        images: Liste d'images OpenCV (BGR)
        message: Message à afficher
        alert_type: 'focus' ou 'telephone'
    """
    # Couleur du header selon le type
    if alert_type == 'telephone':
        header_color = (60, 165, 255)  # Orange BGR
        header_color_rgb = (255, 165, 60)  # Orange RGB
    else:
        header_color = (80, 80, 255)  # Rouge BGR
        header_color_rgb = (255, 80, 80)  # Rouge RGB

    if not images:
        canvas = np.full((400, 600, 3), (25, 25, 30), dtype=np.uint8)
        cv2.putText(canvas, "Ajoute des memes dans le dossier !", (50, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        return canvas

    # Dimensions
    thumb_size = 280
    padding = 20
    cols = min(2, len(images))
    rows = (len(images) + 1) // 2

    canvas_w = cols * thumb_size + (cols + 1) * padding
    canvas_h = rows * thumb_size + (rows + 1) * padding + 80

    # Canvas
    canvas = np.full((canvas_h, canvas_w, 3), (25, 25, 30), dtype=np.uint8)

    # Header avec gradient
    header_h = 60
    for y in range(header_h):
        alpha = y / header_h
        color = (int(40 + 20 * alpha), int(40 + 20 * alpha), int(50 + 20 * alpha))
        cv2.line(canvas, (0, y), (canvas_w, y), color, 1)

    # Titre avec PIL
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
        font = None
        for path in font_paths:
            if Path(path).exists():
                font = ImageFont.truetype(path, 32)
                break
        if font:
            img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((padding, 12), message, font=font, fill=header_color_rgb)
            canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(canvas, message, (padding, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, header_color, 2)
    except Exception:
        cv2.putText(canvas, message, (padding, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, header_color, 2)

    # Placer les images
    for idx, img in enumerate(images[:4]):
        row = idx // 2
        col = idx % 2

        x = padding + col * (thumb_size + padding)
        y = header_h + padding + row * (thumb_size + padding)

        resized = cv2.resize(img, (thumb_size, thumb_size))
        cv2.rectangle(canvas, (x - 2, y - 2),
                     (x + thumb_size + 2, y + thumb_size + 2),
                     (60, 60, 70), 2)
        canvas[y:y + thumb_size, x:x + thumb_size] = resized

    return canvas
