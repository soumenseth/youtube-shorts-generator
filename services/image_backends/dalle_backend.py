import requests
from pathlib import Path
import openai
from services.image_backends.base import ImageGenerationStrategy


class DalleBackend(ImageGenerationStrategy):
    def __init__(self, client: openai.OpenAI):
        self._client = client

    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str:
        response = self._client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img_bytes = requests.get(image_url, timeout=30).content
        Path(output_path).write_bytes(img_bytes)
        return output_path
