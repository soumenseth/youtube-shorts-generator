from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class ILLMService(Protocol):
    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Any: ...
