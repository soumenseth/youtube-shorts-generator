from models.script_model import ScriptAnalysis
from agents.base_agent import BaseAgent
from protocols.llm_protocol import ILLMService
from langfuse import observe, get_client

_TOOLS = [
    {
        "name": "classify_script",
        "description": "Classify the script and return a complete analysis for YouTube video production.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_type": {
                    "type": "string",
                    "enum": ["educational", "storytelling", "how_to", "entertainment",
                             "motivational", "product_review", "comedy", "news", "tutorial"],
                },
                "visual_style": {
                    "type": "string",
                    "enum": ["cinematic", "minimal", "vibrant", "dark_moody",
                             "bright_cheerful", "professional", "animated"],
                },
                "video_format": {
                    "type": "string",
                    "enum": ["shorts", "standard"],
                    "description": "shorts = vertical 9:16 (recommended for <60s), standard = horizontal 16:9",
                },
                "target_audience": {"type": "string"},
                "tone": {"type": "string"},
                "key_themes": {"type": "array", "items": {"type": "string"}},
                "estimated_duration_seconds": {
                    "type": "integer",
                    "description": "15–60 seconds for shorts, up to 600 for standard",
                },
                "hook_suggestion": {
                    "type": "string",
                    "description": "Compelling first line that makes viewers stop scrolling",
                },
                "title_suggestion": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"},
            },
            "required": [
                "video_type", "visual_style", "video_format", "target_audience",
                "tone", "key_themes", "estimated_duration_seconds",
                "hook_suggestion", "title_suggestion", "tags", "reasoning",
            ],
        },
    }
]

_SYSTEM = """You are an expert YouTube content strategist with a track record of viral videos.
Analyze the script and classify it for optimal YouTube production.
Consider: video type that best fits the content, visual style that maximises engagement,
target audience, viral potential, and whether it should be a Short or standard video.
Always call classify_script to return your analysis."""


class ScriptAnalyzerAgent(BaseAgent):
    def __init__(self, llm_service: ILLMService):
        super().__init__(llm_service, "ScriptAnalyzer")
        self._result: dict = {}

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "classify_script":
            self._result = inputs
            return {"status": "ok"}
        return {"error": f"Unknown tool: {name}"}

    @observe()
    def analyze(self, script: str) -> ScriptAnalysis:
        get_client().update_current_span(input={"script_preview": script[:300]})
        self._result = {}

        messages = [{"role": "user", "content": f"Analyze this script:\n\n{script}"}]
        self._run_tool_loop(messages, _SYSTEM, _TOOLS)

        if not self._result:
            raise RuntimeError("ScriptAnalyzerAgent: no classification returned")

        result = ScriptAnalysis(**self._result)
        get_client().update_current_span(
            output={
                "video_type": result.video_type.value,
                "visual_style": result.visual_style.value,
                "video_format": result.video_format.value,
                "estimated_duration_seconds": result.estimated_duration_seconds,
            },
        )
        return result
