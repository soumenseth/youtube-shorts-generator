from typing import Protocol, runtime_checkable


@runtime_checkable
class IImageGenerator(Protocol):
    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str: ...
