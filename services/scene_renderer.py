import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from moviepy.editor import ImageClip, VideoClip
from models.video_plan import VideoPlan, Scene
from models.video_types import VIDEO_DIMENSIONS, DALLE_SIZES
from protocols.image_protocol import IImageGenerator
from services.image_prep import resize_crop, add_overlay


class SceneRenderError(Exception):
    """Raised when a scene renderer fails to produce a clip."""


class SceneRenderer(ABC):
    @abstractmethod
    def render(
        self,
        scene: Scene,
        plan: VideoPlan,
        intermediate_path: str,
    ) -> VideoClip:
        """
        Render one scene to an in-memory MoviePy VideoClip.
        Duration must match scene.duration_seconds (±1s).
        Raises SceneRenderError on failure.
        """


class SlideshowSceneRenderer(SceneRenderer):
    """Renders each scene as a static DALL-E image with optional text overlay."""

    def __init__(self, image_generator: IImageGenerator):
        self._gen = image_generator

    def render(self, scene: Scene, plan: VideoPlan, intermediate_path: str) -> VideoClip:
        dims = VIDEO_DIMENSIONS[plan.video_format]
        size = DALLE_SIZES.get(plan.video_format, "1024x1792")
        prompt = f"{scene.visual_prompt}, {plan.visual_style.value} style, ultra detailed, professional"

        img_path = intermediate_path.replace(".mp4", ".png")
        try:
            self._gen.generate(prompt, img_path, size)
        except Exception as exc:
            raise SceneRenderError(f"Image generation failed for scene {scene.scene_number}: {exc}") from exc

        with self._prepared_image(img_path, dims, scene.text_overlay) as prep:
            clip = ImageClip(prep).set_duration(scene.duration_seconds)
            clip = clip.fadein(0.3).fadeout(0.3)

        return clip

    @contextmanager
    def _prepared_image(
        self,
        image_path: str,
        target: tuple[int, int],
        text_overlay: str | None,
    ):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            img = Image.open(image_path).convert("RGB")
            img = resize_crop(img, target)
            if text_overlay:
                img = add_overlay(img, text_overlay)
            img.save(tmp_path)
            yield tmp_path
        finally:
            Path(tmp_path).unlink(missing_ok=True)
