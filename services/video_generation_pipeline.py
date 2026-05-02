from langfuse import observe, get_client
from models.video_plan import VideoPlan
from managers.asset_manager import AssetManager
from services.scene_renderer import SceneRenderer
from services.video_assembler import VideoAssembler


class VideoGenerationPipeline:
    """Orchestrates per-scene rendering and final assembly. Backend-agnostic."""

    def __init__(self, renderer: SceneRenderer, assembler: VideoAssembler):
        self._renderer = renderer
        self._assembler = assembler

    @observe(name="video-generation-pipeline")
    def generate(self, plan: VideoPlan, audio_path: str, assets: AssetManager) -> str:
        get_client().update_current_span(
            metadata={
                "scenes": len(plan.scenes),
                "renderer": type(self._renderer).__name__,
            }
        )

        print(f"[VideoGenPipeline] Rendering {len(plan.scenes)} scenes via {type(self._renderer).__name__}...")
        clips = []
        for scene in plan.scenes:
            print(f"  Scene {scene.scene_number}: {scene.visual_prompt[:60]}...")
            clip = self._renderer.render(scene, plan, assets.clip_path(scene.scene_number))
            clips.append(clip)
        print(f"  {len(clips)} clips rendered")

        print("[VideoGenPipeline] Assembling final video...")
        output = assets.output_path()
        try:
            result = self._assembler.assemble(clips, audio_path, output)
        finally:
            for clip in clips:
                clip.close()

        print(f"  Video → {result}")
        return result
