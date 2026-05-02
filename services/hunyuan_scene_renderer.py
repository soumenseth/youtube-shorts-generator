from moviepy.editor import VideoFileClip, VideoClip
from models.video_plan import VideoPlan, Scene
from models.video_types import VIDEO_DIMENSIONS
from services.scene_renderer import SceneRenderer, SceneRenderError
from services.hunyuan_client import HunyuanClient


class HunyuanSceneRenderer(SceneRenderer):
    """Renders each scene as an AI-generated video clip via HunyuanVideo on RunPod Serverless."""

    def __init__(self, client: HunyuanClient):
        self._client = client

    def render(self, scene: Scene, plan: VideoPlan, intermediate_path: str) -> VideoClip:
        dims = VIDEO_DIMENSIONS[plan.video_format]
        prompt = f"{scene.visual_prompt}, {plan.visual_style.value} style, cinematic motion, high quality"
        num_frames = scene.duration_seconds * 24  # 24 fps

        try:
            mp4_path = self._client.generate_clip(
                prompt=prompt,
                width=dims[0],
                height=dims[1],
                num_frames=num_frames,
                output_path=intermediate_path,
            )
        except Exception as exc:
            raise SceneRenderError(
                f"HunyuanVideo generation failed for scene {scene.scene_number}: {exc}"
            ) from exc

        return VideoFileClip(mp4_path)
