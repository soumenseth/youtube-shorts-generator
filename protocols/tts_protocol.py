from typing import Protocol, runtime_checkable


@runtime_checkable
class ITTSService(Protocol):
    def generate_audio(self, text: str, output_path: str, voice: str = "alloy") -> str: ...
    def voice_for(self, video_type: str) -> str: ...
