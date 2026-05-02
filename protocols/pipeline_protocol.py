from typing import Protocol, Any, runtime_checkable
from models.script_model import ScriptAnalysis
from models.video_plan import VideoPlan, QualityReview


@runtime_checkable
class IVideoProductionPipeline(Protocol):
    """Narrow interface that OrchestratorAgent depends on — only what it actually uses."""

    def complete(
        self,
        messages: list[dict],
        system: str | None,
        tools: list[dict] | None,
        max_tokens: int,
    ) -> Any: ...

    def analyze_script(self, script: str) -> ScriptAnalysis: ...

    def plan_video(self, script: str, analysis: ScriptAnalysis) -> VideoPlan: ...

    def review_plan(
        self, script: str, plan: VideoPlan
    ) -> tuple[VideoPlan, list[QualityReview]]: ...

    def output_path(self) -> str: ...

    def generate_video(self, plan: VideoPlan, output_path: str | None) -> str: ...
