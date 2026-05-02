from pathlib import Path
from datetime import datetime


class AssetManager:
    """Manages all file paths for a generation session."""

    def __init__(self, base_dir: str = "output", session_id: str | None = None):
        _session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_dir = Path(base_dir) / _session_id
        self._images_dir = _session_dir / "images"
        self._audio_dir = _session_dir / "audio"
        self._session_dir = _session_dir

        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, scene_number: int) -> str:
        return str(self._images_dir / f"scene_{scene_number:02d}.png")

    def clip_path(self, scene_number: int) -> str:
        return str(self._images_dir / f"scene_{scene_number:02d}.mp4")

    def audio_path(self) -> str:
        return str(self._audio_dir / "narration.mp3")

    def output_path(self, filename: str = "final_video.mp4") -> str:
        return str(self._session_dir / filename)
