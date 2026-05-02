from abc import ABC, abstractmethod


class ImageGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str: ...
