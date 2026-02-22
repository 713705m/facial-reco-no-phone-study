"""
Systeme d'alerte : son + affichage d'images aleatoires.
Supporte 2 types d'alertes : focus (distraction) et telephone.
"""
import random
import threading
from pathlib import Path

import cv2
import numpy as np

# Import conditionnel pour pygame (son)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class AlertSystem:
    """Gere les alertes sonores et visuelles."""

    def __init__(self, memes_dir: str, volume: float = 0.7,
                 num_images: int = 1, display_duration: float = 5.0):
        """
        Args:
            memes_dir: Dossier racine du projet (pour trouver memes-focus et memes-telephone)
            volume: Volume du son (0.0 a 1.0)
            num_images: Nombre d'images a afficher
            display_duration: Duree d'affichage en secondes
        """
        self.project_root = Path(memes_dir).parent
        self.memes_focus_dir = self.project_root / "memes-focus"
        self.memes_telephone_dir = self.project_root / "memes-telephone"
        self.volume = volume
        self.num_images = num_images
        self.display_duration = display_duration
        self._alert_active = False
        self._alert_thread = None
        self._should_show_images = False
        self._current_alert_type = None

        # Initialiser pygame pour le son
        if PYGAME_AVAILABLE:
            pygame.mixer.init()

        # Charger les images disponibles
        self.images_focus = self._load_image_list(self.memes_focus_dir)
        self.images_telephone = self._load_image_list(self.memes_telephone_dir)

    def _load_image_list(self, directory: Path) -> list[Path]:
        """Charge la liste des images dans un dossier."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = []

        if directory.exists():
            for f in directory.iterdir():
                if f.suffix.lower() in valid_extensions:
                    images.append(f)

        return images

    def refresh_images(self):
        """Rafraichit la liste des images."""
        self.images_focus = self._load_image_list(self.memes_focus_dir)
        self.images_telephone = self._load_image_list(self.memes_telephone_dir)
        print(f"Memes rafraichis: {len(self.images_focus)} focus, {len(self.images_telephone)} telephone")

    def play_alert_sound(self, alert_type: str = "focus"):
        """Joue un son d'alerte selon le type."""
        if not PYGAME_AVAILABLE:
            print('\a')
            return

        # Chercher un fichier son specifique au type
        if alert_type == "telephone":
            sound_file = self.project_root / 'config' / 'alert_telephone.wav'
        else:
            sound_file = self.project_root / 'config' / 'alert_focus.wav'

        # Fallback sur alert.wav generique
        if not sound_file.exists():
            sound_file = self.project_root / 'config' / 'alert.wav'

        if sound_file.exists():
            try:
                pygame.mixer.music.load(str(sound_file))
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Erreur son: {e}")
                self._generate_beep(alert_type)
        else:
            self._generate_beep(alert_type)

    def _generate_beep(self, alert_type: str = "focus"):
        """Genere un beep synthetique selon le type."""
        if not PYGAME_AVAILABLE:
            print('\a')
            return

        sample_rate = 44100

        if alert_type == "telephone":
            # Son plus aigu et rapide pour telephone
            duration = 0.15
            frequency = 1200
            # Double beep
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave1 = np.sin(2 * np.pi * frequency * t) * 0.3
            silence = np.zeros(int(sample_rate * 0.05))
            wave = np.concatenate([wave1, silence, wave1])
        else:
            # Son standard pour focus
            duration = 0.3
            frequency = 800
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t) * 0.3

        wave = (wave * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))

        sound = pygame.sndarray.make_sound(stereo)
        sound.set_volume(self.volume)
        sound.play()

    def show_random_images(self, alert_type: str = "focus"):
        """Affiche des images aleatoires selon le type d'alerte."""
        # Selectionner le bon dossier de memes
        if alert_type == "telephone":
            image_files = self.images_telephone
        else:
            image_files = self.images_focus

        if not image_files:
            folder_name = "memes-telephone" if alert_type == "telephone" else "memes-focus"
            print(f"Aucune image dans le dossier {folder_name}/")
            return

        # Selectionner 1 image aleatoire
        selected_path = random.choice(image_files)
        img = cv2.imread(str(selected_path))
        if img is None:
            return

        # Redimensionner pour un affichage correct
        max_size = 500
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Afficher
        window_name = "ALERTE"
        cv2.imshow(window_name, img)
        cv2.waitKey(int(self.display_duration * 1000))
        cv2.destroyWindow(window_name)

    def trigger_alert(self, alert_type: str = "focus"):
        """Declenche une alerte selon le type (focus ou telephone)."""
        if self._alert_active:
            return

        self._alert_active = True
        self._should_show_images = True
        self._current_alert_type = alert_type

        def _sound_worker():
            self.play_alert_sound(alert_type)

        self._alert_thread = threading.Thread(target=_sound_worker)
        self._alert_thread.start()

    def should_show_images(self) -> bool:
        """Verifie si des images doivent etre affichees."""
        return self._should_show_images

    def show_images_if_needed(self):
        """Affiche les images si necessaire (a appeler depuis le main thread)."""
        if self._should_show_images:
            self._should_show_images = False
            self.show_random_images(self._current_alert_type or "focus")
            self._alert_active = False

    def get_current_alert_type(self) -> str:
        """Retourne le type d'alerte actuel."""
        return self._current_alert_type

    def is_alert_active(self) -> bool:
        """Verifie si une alerte est en cours."""
        return self._alert_active
