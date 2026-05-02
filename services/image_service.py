from services.image_backends.base import ImageGenerationStrategy


class ImageService:
    """Delegates image generation to an injected strategy (DALL-E, HF FLUX, etc.)."""

    def __init__(self, strategy: ImageGenerationStrategy):
        self._strategy = strategy

    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str:
        return self._strategy.generate(prompt, output_path, size)
