"""
Logging des statistiques de distraction.
Permet de suivre tes progres et generer des rapports.
"""
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DistractionEvent:
    """Un evenement de distraction."""
    timestamp: str
    duration: float  # secondes
    pitch_angle: float
    phone_detected: bool


@dataclass
class SessionStats:
    """Statistiques d'une session."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    total_distractions: int
    total_distraction_time: float
    average_distraction_duration: float
    events: list[dict]


class StatsLogger:
    """Logger de statistiques de distraction."""

    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Args:
            log_dir: Dossier pour les fichiers de log
            enabled: Activer/desactiver le logging
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.log_dir.mkdir(exist_ok=True)

        # Session actuelle
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        self.events: list[DistractionEvent] = []

        # Tracking en cours
        self._distraction_start: Optional[datetime] = None
        self._current_pitch: float = 0.0
        self._phone_detected: bool = False

    def start_distraction(self, pitch: float, phone_detected: bool):
        """Marque le début d'une distraction."""
        if self._distraction_start is None:
            self._distraction_start = datetime.now()
            self._current_pitch = pitch
            self._phone_detected = phone_detected

    def end_distraction(self):
        """Marque la fin d'une distraction et l'enregistre."""
        if self._distraction_start is not None:
            duration = (datetime.now() - self._distraction_start).total_seconds()

            event = DistractionEvent(
                timestamp=self._distraction_start.isoformat(),
                duration=duration,
                pitch_angle=self._current_pitch,
                phone_detected=self._phone_detected
            )
            self.events.append(event)

            self._distraction_start = None

    def get_session_stats(self) -> SessionStats:
        """Calcule les statistiques de la session."""
        total_time = sum(e.duration for e in self.events)
        avg_duration = total_time / len(self.events) if self.events else 0

        return SessionStats(
            session_id=self.session_id,
            start_time=self.session_start.isoformat(),
            end_time=datetime.now().isoformat(),
            total_distractions=len(self.events),
            total_distraction_time=total_time,
            average_distraction_duration=avg_duration,
            events=[asdict(e) for e in self.events]
        )

    def save(self):
        """Sauvegarde les statistiques dans un fichier JSON."""
        if not self.enabled:
            return

        stats = self.get_session_stats()
        filename = self.log_dir / f"session_{self.session_id}.json"

        with open(filename, 'w') as f:
            json.dump(asdict(stats), f, indent=2)

        print(f"Stats sauvegardees: {filename}")

    def print_summary(self):
        """Affiche un resume de la session."""
        stats = self.get_session_stats()

        print("\n" + "=" * 50)
        print("RESUME DE LA SESSION")
        print("=" * 50)
        print(f"Duree totale: {self._format_duration((datetime.now() - self.session_start).total_seconds())}")
        print(f"Nombre de distractions: {stats.total_distractions}")
        print(f"Temps total distrait: {self._format_duration(stats.total_distraction_time)}")
        print(f"Duree moyenne: {stats.average_distraction_duration:.1f}s")
        print("=" * 50)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Formate une duree en minutes:secondes."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"

    def load_history(self, days: int = 7) -> list[SessionStats]:
        """Charge l'historique des sessions recentes."""
        sessions = []

        for f in sorted(self.log_dir.glob("session_*.json"), reverse=True):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    # Verifier si assez recent
                    start = datetime.fromisoformat(data['start_time'])
                    if (datetime.now() - start).days <= days:
                        sessions.append(SessionStats(**data))
            except (json.JSONDecodeError, KeyError):
                continue

        return sessions

    def print_weekly_report(self):
        """Affiche un rapport hebdomadaire."""
        sessions = self.load_history(days=7)

        if not sessions:
            print("Aucune donnee cette semaine.")
            return

        total_distractions = sum(s.total_distractions for s in sessions)
        total_time = sum(s.total_distraction_time for s in sessions)

        print("\n" + "=" * 50)
        print("RAPPORT HEBDOMADAIRE")
        print("=" * 50)
        print(f"Sessions: {len(sessions)}")
        print(f"Total distractions: {total_distractions}")
        print(f"Temps perdu: {self._format_duration(total_time)}")
        print("=" * 50)
