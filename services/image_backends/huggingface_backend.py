import requests
from pathlib import Path
from services.image_backends.base import ImageGenerationStrategy


class HuggingFaceFluxBackend(ImageGenerationStrategy):
    _API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

    def __init__(self, hf_token: str):
        self._token = hf_token

    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str:
        headers = {"Authorization": f"Bearer {self._token}"}
        response = requests.post(self._API_URL, headers=headers, json={"inputs": prompt}, timeout=120)
        response.raise_for_status()
        Path(output_path).write_bytes(response.content)
        return output_path
