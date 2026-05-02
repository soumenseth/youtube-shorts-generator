import os
import anthropic
from langfuse import observe, get_client


class LLMService:
    """Anthropic Claude wrapper for all agent reasoning tasks."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-6"

    @observe(as_type="generation")
    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list | None = None,
        max_tokens: int = 4096,
    ) -> anthropic.types.Message:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        response = self.client.messages.create(**kwargs)
        get_client().update_current_generation(
            model=self.model,
            usage_details={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )
        return response
