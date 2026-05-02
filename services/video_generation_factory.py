from enum import Enum
from typing import Callable
from protocols.image_protocol import IImageGenerator
from services.scene_renderer import SceneRenderer, SlideshowSceneRenderer
from services.hunyuan_scene_renderer import HunyuanSceneRenderer
from services.hunyuan_client import HunyuanClient
from services.video_assembler import VideoAssembler
from services.video_generation_pipeline import VideoGenerationPipeline


class VideoBackend(str, Enum):
    SLIDESHOW = "slideshow"
    HUNYUAN = "hunyuan"


# Registry: adding a new backend = one entry here, zero other changes
_RENDERER_REGISTRY: dict[VideoBackend, Callable[..., SceneRenderer]] = {
    VideoBackend.SLIDESHOW: lambda img_gen, _client: SlideshowSceneRenderer(img_gen),
    VideoBackend.HUNYUAN:   lambda _img_gen, client: HunyuanSceneRenderer(client),
}


class VideoGenerationFactory:
    """
    Typed static factory methods for explicit wiring (preferred).
    from_backend() for env-var-driven selection.
    """

    @staticmethod
    def slideshow(image_generator: IImageGenerator) -> VideoGenerationPipeline:
        return VideoGenerationPipeline(
            renderer=SlideshowSceneRenderer(image_generator),
            assembler=VideoAssembler(),
        )

    @staticmethod
    def hunyuan(client: HunyuanClient) -> VideoGenerationPipeline:
        return VideoGenerationPipeline(
            renderer=HunyuanSceneRenderer(client),
            assembler=VideoAssembler(),
        )

    @classmethod
    def from_backend(
        cls,
        backend: VideoBackend,
        image_generator: IImageGenerator,
        hunyuan_client: HunyuanClient | None,
    ) -> VideoGenerationPipeline:
        renderer = _RENDERER_REGISTRY[backend](image_generator, hunyuan_client)
        return VideoGenerationPipeline(renderer=renderer, assembler=VideoAssembler())
