from agents.base_agent import BaseAgent
from protocols.pipeline_protocol import IVideoProductionPipeline
from models.script_model import ScriptAnalysis
from models.video_plan import VideoPlan
from langfuse import observe, get_client

_TOOLS = [
    {
        "name": "analyze_script",
        "description": (
            "Step 1: Analyze the script to determine video type, visual style, "
            "target audience, format, and hook. Run this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string"},
            },
            "required": ["script"],
        },
    },
    {
        "name": "plan_video",
        "description": (
            "Step 2: Create a scene-by-scene production plan. "
            "Pass the analysis object from analyze_script."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string"},
                "analysis": {"type": "object", "description": "Result from analyze_script"},
            },
            "required": ["script", "analysis"],
        },
    },
    {
        "name": "review_and_approve_plan",
        "description": (
            "Step 3: Run 3 rounds of quality review to ensure YouTube hit potential. "
            "Pass the plan object from plan_video."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string"},
                "plan": {"type": "object", "description": "Result from plan_video"},
            },
            "required": ["script", "plan"],
        },
    },
    {
        "name": "generate_and_assemble_video",
        "description": (
            "Step 4: Generate all assets (TTS audio, DALL-E images) and assemble the final MP4. "
            "Pass the approved plan from review_and_approve_plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object", "description": "Approved plan from review step"},
            },
            "required": ["plan"],
        },
    },
]

_SYSTEM = """You are the video production orchestrator. Coordinate the AI team to produce
a high-quality YouTube video from the given script.

You MUST call all four tools in order:
  1. analyze_script      — understand the script
  2. plan_video          — create the production plan
  3. review_and_approve_plan  — quality-check 3 times (mandatory)
  4. generate_and_assemble_video — produce the final video

Do not skip any step. Pass outputs from each step as inputs to the next."""


class OrchestratorAgent(BaseAgent):
    def __init__(self, pipeline: IVideoProductionPipeline):
        # PipelineManager.complete() satisfies ILLMService structurally
        super().__init__(pipeline, "Orchestrator")
        self._pipeline = pipeline
        self._output_path: str = ""

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "analyze_script":
            analysis = self._pipeline.analyze_script(inputs["script"])
            print(f"[Orchestrator] ✓ Script analyzed: {analysis.video_type} | {analysis.visual_style} | {analysis.video_format}")
            get_client().update_current_span(
                metadata={
                    "video_type": analysis.video_type.value,
                    "visual_style": analysis.visual_style.value,
                    "video_format": analysis.video_format.value,
                    "target_audience": analysis.target_audience,
                },
            )
            return analysis.model_dump()

        if name == "plan_video":
            analysis = ScriptAnalysis(**inputs["analysis"])
            plan = self._pipeline.plan_video(inputs["script"], analysis)
            print(f"[Orchestrator] ✓ Video planned: {len(plan.scenes)} scenes, {plan.total_duration_seconds}s")
            return plan.model_dump()

        if name == "review_and_approve_plan":
            plan = VideoPlan(**inputs["plan"])
            print("[Orchestrator] ↻ Running 3-round quality review...")
            approved, reviews = self._pipeline.review_plan(inputs["script"], plan)
            scores = [r.overall_score for r in reviews]
            print(f"[Orchestrator] ✓ Review complete. Scores: {scores}")
            return approved.model_dump()

        if name == "generate_and_assemble_video":
            plan = VideoPlan(**inputs["plan"])
            result = self._pipeline.generate_video(plan, self._output_path)
            print(f"[Orchestrator] ✓ Video generated: {result}")
            get_client().update_current_span(output={"video_path": result})
            return {"output_path": result, "status": "success"}

        return {"error": f"Unknown tool: {name}"}

    @observe(name="video-generation")
    def run(self, script: str) -> str:
        self._output_path = self._pipeline.output_path()
        get_client().update_current_span(
            input={"script_preview": script[:500]},
            metadata={"script_length": len(script)},
        )
        print("[Orchestrator] Starting video generation pipeline\n")
        messages = [{"role": "user", "content": f"Create a YouTube video from this script:\n\n{script}"}]
        _, final_text = self._run_tool_loop(messages, _SYSTEM, _TOOLS, max_tokens=8192)
        if final_text:
            print(f"\n[Orchestrator] {final_text}")
        return self._output_path
