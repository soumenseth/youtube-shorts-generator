from services.image_backends.base import ImageGenerationStrategy


class FallbackImageBackend(ImageGenerationStrategy):
    """Tries primary backend; falls back to secondary on any exception."""

    def __init__(self, primary: ImageGenerationStrategy, fallback: ImageGenerationStrategy):
        self._primary = primary
        self._fallback = fallback

    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str:
        try:
            return self._primary.generate(prompt, output_path, size)
        except Exception as exc:
            print(f"  Primary image backend failed ({exc}), trying fallback...")
            return self._fallback.generate(prompt, output_path, size)
