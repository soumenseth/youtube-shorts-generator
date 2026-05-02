from models.video_types import TTS_VOICES
from services.tts_backends.base import TTSBackend


class TTSService:
    """Delegates audio generation to an injected backend (OpenAI TTS, ElevenLabs, etc.)."""

    def __init__(self, backend: TTSBackend):
        self._backend = backend

    def generate_audio(self, text: str, output_path: str, voice: str = "alloy") -> str:
        return self._backend.generate(text, output_path, voice)

    def voice_for(self, video_type: str) -> str:
        return TTS_VOICES.get(video_type, "alloy")
