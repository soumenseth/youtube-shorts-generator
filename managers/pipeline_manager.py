from langfuse import observe, get_client
from protocols.llm_protocol import ILLMService
from protocols.tts_protocol import ITTSService
from agents.script_analyzer import ScriptAnalyzerAgent
from agents.video_planner import VideoPlanner
from agents.quality_reviewer import QualityReviewAgent
from managers.asset_manager import AssetManager
from models.script_model import ScriptAnalysis
from models.video_plan import VideoPlan, QualityReview
from services.video_generation_pipeline import VideoGenerationPipeline


class PipelineManager:
    """Facade over the agent and service layer. All internals are private."""

    def __init__(
        self,
        llm: ILLMService,
        tts: ITTSService,
        video_pipeline: VideoGenerationPipeline,
        assets: AssetManager,
        script_analyzer: ScriptAnalyzerAgent,
        video_planner: VideoPlanner,
        quality_reviewer: QualityReviewAgent,
    ):
        self._llm = llm
        self._tts = tts
        self._video_pipeline = video_pipeline
        self._assets = assets
        self._script_analyzer = script_analyzer
        self._video_planner = video_planner
        self._quality_reviewer = quality_reviewer

    # ── Public interface (used by OrchestratorAgent via IVideoProductionPipeline) ──

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 8192,
    ):
        return self._llm.complete(messages, system=system, tools=tools, max_tokens=max_tokens)

    def analyze_script(self, script: str) -> ScriptAnalysis:
        return self._script_analyzer.analyze(script)

    def plan_video(self, script: str, analysis: ScriptAnalysis) -> VideoPlan:
        return self._video_planner.plan(script, analysis)

    def review_plan(
        self, script: str, plan: VideoPlan
    ) -> tuple[VideoPlan, list[QualityReview]]:
        return self._quality_reviewer.run_full_review(script, plan)

    def output_path(self) -> str:
        return self._assets.output_path()

    @observe(name="asset-generation")
    def generate_video(self, plan: VideoPlan, output_path: str | None = None) -> str:
        get_client().update_current_span(
            metadata={
                "video_type": plan.video_type.value,
                "video_format": plan.video_format.value,
                "scenes": len(plan.scenes),
                "total_duration_seconds": plan.total_duration_seconds,
            },
        )

        # 1. TTS narration
        print("\n[Pipeline] Generating narration audio...")
        narration = " ".join(s.narration for s in plan.scenes)
        voice = self._tts.voice_for(plan.video_type.value)
        audio = self._assets.audio_path()
        self._tts.generate_audio(narration, audio, voice=voice)
        print(f"  Audio → {audio}")

        # 2. Render scenes + assemble via VideoGenerationPipeline
        result = self._video_pipeline.generate(plan, audio, self._assets)
        return result
