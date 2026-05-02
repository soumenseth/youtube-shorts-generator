from models.script_model import ScriptAnalysis
from models.video_plan import VideoPlan
from models.video_types import DALLE_SIZES
from agents.base_agent import BaseAgent
from protocols.llm_protocol import ILLMService
from langfuse import observe, get_client

_TOOLS = [
    {
        "name": "create_video_plan",
        "description": "Create a complete scene-by-scene YouTube video production plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "hook": {"type": "string", "description": "Attention-grabbing opening line"},
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_number": {"type": "integer"},
                            "narration": {"type": "string", "description": "Exact text for TTS narration"},
                            "visual_prompt": {
                                "type": "string",
                                "description": "Detailed DALL-E prompt — no text/words in the image, rich visual detail",
                            },
                            "duration_seconds": {"type": "integer"},
                            "text_overlay": {
                                "type": "string",
                                "description": "Short bold caption shown on screen (optional)",
                            },
                            "transition": {
                                "type": "string",
                                "enum": ["fade", "cut", "slide"],
                            },
                        },
                        "required": ["scene_number", "narration", "visual_prompt", "duration_seconds"],
                    },
                },
                "background_music_style": {"type": "string"},
                "call_to_action": {"type": "string"},
                "thumbnail_concept": {"type": "string"},
            },
            "required": [
                "title", "description", "hook", "scenes",
                "background_music_style", "call_to_action", "thumbnail_concept",
            ],
        },
    }
]


class VideoPlanner(BaseAgent):
    def __init__(self, llm_service: ILLMService):
        super().__init__(llm_service, "VideoPlanner")
        self._plan_data: dict = {}

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "create_video_plan":
            self._plan_data = inputs
            return {"status": "ok"}
        return {"error": f"Unknown tool: {name}"}

    @observe()
    def plan(self, script: str, analysis: ScriptAnalysis) -> VideoPlan:
        self._plan_data = {}

        size = DALLE_SIZES.get(analysis.video_format, "1024x1792")
        orientation = "vertical 9:16" if "1792" in size else "horizontal 16:9"

        system = f"""You are an expert YouTube Short video director and storyboard artist.
Create a compelling, scene-by-scene video plan that maximises watch time and engagement.

Format: {analysis.video_format} — {orientation} ({size} images)
Type: {analysis.video_type}  |  Style: {analysis.visual_style}
Audience: {analysis.target_audience}  |  Tone: {analysis.tone}
Duration target: {analysis.estimated_duration_seconds}s
Key themes: {', '.join(analysis.key_themes)}
Hook to use: {analysis.hook_suggestion}

Visual prompt rules:
- Rich, photorealistic detail — lighting, mood, composition, colours
- Style must match: {analysis.visual_style}
- NO text, letters, or words inside any image
- Each scene visually distinct to maintain viewer interest

Call create_video_plan to return the plan."""

        messages = [{"role": "user", "content": f"Plan a video for this script:\n\n{script}"}]
        self._run_tool_loop(messages, system, _TOOLS, max_tokens=6000)

        if not self._plan_data:
            raise RuntimeError("VideoPlanner: no plan returned")

        total = sum(s.get("duration_seconds", 5) for s in self._plan_data.get("scenes", []))
        result = VideoPlan(
            **self._plan_data,
            video_type=analysis.video_type,
            visual_style=analysis.visual_style,
            video_format=analysis.video_format,
            target_audience=analysis.target_audience,
            total_duration_seconds=total,
        )
        get_client().update_current_span(
            output={
                "title": result.title,
                "scenes": len(result.scenes),
                "total_duration_seconds": result.total_duration_seconds,
            },
        )
        return result
