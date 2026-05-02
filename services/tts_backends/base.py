from abc import ABC, abstractmethod


class TTSBackend(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str, voice: str = "alloy") -> str: ...
