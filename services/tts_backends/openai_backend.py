from pathlib import Path
import openai
from services.tts_backends.base import TTSBackend


class OpenAITTSBackend(TTSBackend):
    def __init__(self, client: openai.OpenAI):
        self._client = client

    def generate(self, text: str, output_path: str, voice: str = "alloy") -> str:
        response = self._client.audio.speech.create(model="tts-1", voice=voice, input=text)
        Path(output_path).write_bytes(response.read())
        return output_path
