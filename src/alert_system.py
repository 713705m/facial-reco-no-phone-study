"""
Système d'alerte : son + affichage d'images aléatoires.
"""
import os
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
    """Gère les alertes sonores et visuelles."""

    def __init__(self, memes_dir: str, volume: float = 0.7,
                 num_images: int = 4, display_duration: float = 5.0):
        """
        Args:
            memes_dir: Dossier contenant les images/memes
            volume: Volume du son (0.0 à 1.0)
            num_images: Nombre d'images à afficher
            display_duration: Durée d'affichage en secondes
        """
        self.memes_dir = Path(memes_dir)
        self.volume = volume
        self.num_images = num_images
        self.display_duration = display_duration
        self._alert_active = False
        self._alert_thread = None

        # Initialiser pygame pour le son
        if PYGAME_AVAILABLE:
            pygame.mixer.init()

        # Charger les images disponibles
        self.image_files = self._load_image_list()

    def _load_image_list(self) -> list[Path]:
        """Charge la liste des images dans le dossier memes."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        images = []

        if self.memes_dir.exists():
            for f in self.memes_dir.iterdir():
                if f.suffix.lower() in valid_extensions:
                    images.append(f)

        return images

    def refresh_images(self):
        """Rafraîchit la liste des images (si tu en ajoutes pendant l'exécution)."""
        self.image_files = self._load_image_list()

    def play_alert_sound(self):
        """Joue un son d'alerte."""
        if not PYGAME_AVAILABLE:
            # Fallback : beep système
            print('\a')  # ASCII bell
            return

        # Chercher un fichier son dans le dossier config
        sound_file = Path(__file__).parent.parent / 'config' / 'alert.wav'

        if sound_file.exists():
            try:
                pygame.mixer.music.load(str(sound_file))
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Erreur son: {e}")
                print('\a')
        else:
            # Générer un beep simple
            self._generate_beep()

    def _generate_beep(self):
        """Génère un beep synthétique."""
        if not PYGAME_AVAILABLE:
            print('\a')
            return

        # Créer un son simple (onde sinusoïdale)
        sample_rate = 44100
        duration = 0.3
        frequency = 800

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t) * 0.3

        # Convertir en format pygame
        wave = (wave * 32767).astype(np.int16)
        stereo = np.column_stack((wave, wave))

        sound = pygame.sndarray.make_sound(stereo)
        sound.set_volume(self.volume)
        sound.play()

    def show_random_images(self):
        """Affiche des images aléatoires dans une fenêtre."""
        if not self.image_files:
            print("Aucune image dans le dossier memes/")
            print("Ajoute des images pour voir des memes!")
            return

        # Sélectionner des images aléatoires
        selected = random.sample(
            self.image_files,
            min(self.num_images, len(self.image_files))
        )

        # Charger et redimensionner les images
        images = []
        target_size = (300, 300)

        for img_path in selected:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)

        if not images:
            return

        # Créer une grille 2x2
        rows = []
        for i in range(0, len(images), 2):
            if i + 1 < len(images):
                row = np.hstack([images[i], images[i + 1]])
            else:
                # Ajouter une image noire si nombre impair
                black = np.zeros_like(images[i])
                row = np.hstack([images[i], black])
            rows.append(row)

        if len(rows) == 1:
            grid = rows[0]
        else:
            grid = np.vstack(rows[:2])

        # Ajouter un message
        cv2.putText(grid, "ARRETE DE SCROLLER!", (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(grid, "Retourne bosser!", (50, grid.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Afficher
        cv2.imshow("ALERTE DISTRACTION", grid)
        cv2.waitKey(int(self.display_duration * 1000))
        cv2.destroyWindow("ALERTE DISTRACTION")

    def trigger_alert(self):
        """Déclenche une alerte complète (son en arrière-plan, images à afficher dans le main thread)."""
        if self._alert_active:
            return

        self._alert_active = True
        self._should_show_images = True

        def _sound_worker():
            self.play_alert_sound()

        self._alert_thread = threading.Thread(target=_sound_worker)
        self._alert_thread.start()

    def should_show_images(self) -> bool:
        """Vérifie si des images doivent être affichées (à appeler depuis le main thread)."""
        return getattr(self, '_should_show_images', False)

    def show_images_if_needed(self):
        """Affiche les images si nécessaire (à appeler depuis le main thread)."""
        if self.should_show_images():
            self._should_show_images = False
            self.show_random_images()
            self._alert_active = False

    def is_alert_active(self) -> bool:
        """Vérifie si une alerte est en cours."""
        return self._alert_active
