import json
from abc import ABC, abstractmethod
from langfuse import observe, get_client
from protocols.llm_protocol import ILLMService


class BaseAgent(ABC):
    """Runs an Anthropic tool-use agentic loop until end_turn."""

    def __init__(self, llm_service: ILLMService, name: str):
        self._llm = llm_service
        self._name = name

    @abstractmethod
    def _handle_tool(self, name: str, inputs: dict) -> dict:
        """Handle a single tool call. Subclasses implement their tool routing here."""

    @observe()
    def _run_tool_loop(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> tuple[list[dict], str]:
        """Execute Claude tool-use loop. Returns (updated_messages, final_text)."""
        get_client().update_current_span(name=f"{self._name}-loop")
        while True:
            response = self._llm.complete(messages, system=system, tools=tools, max_tokens=max_tokens)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                return messages, final_text

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._handle_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return messages, ""
