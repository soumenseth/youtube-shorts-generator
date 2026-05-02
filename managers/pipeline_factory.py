import os
import openai
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.tts_backends.openai_backend import OpenAITTSBackend
from services.image_service import ImageService
from services.image_backends.dalle_backend import DalleBackend
from services.image_backends.huggingface_backend import HuggingFaceFluxBackend
from services.image_backends.fallback_backend import FallbackImageBackend
from services.video_generation_factory import VideoGenerationFactory, VideoBackend
from services.hunyuan_client import HunyuanClient
from agents.script_analyzer import ScriptAnalyzerAgent
from agents.video_planner import VideoPlanner
from agents.quality_reviewer import QualityReviewAgent
from managers.asset_manager import AssetManager
from managers.pipeline_manager import PipelineManager


class PipelineFactory:
    @staticmethod
    def production(session_id: str | None = None) -> PipelineManager:
        oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        llm = LLMService()

        hf_token = os.getenv("HF_TOKEN")
        dalle = DalleBackend(oai)
        image_strategy = (
            FallbackImageBackend(dalle, HuggingFaceFluxBackend(hf_token))
            if hf_token
            else dalle
        )
        images = ImageService(image_strategy)
        tts = TTSService(OpenAITTSBackend(oai))
        assets = AssetManager(session_id=session_id)

        backend = VideoBackend(os.getenv("VIDEO_BACKEND", VideoBackend.SLIDESHOW))
        if backend == VideoBackend.HUNYUAN:
            hunyuan = HunyuanClient(
                api_key=os.environ["RUNPOD_API_KEY"],
                endpoint_id=os.environ["HUNYUAN_ENDPOINT_ID"],
            )
            video_pipeline = VideoGenerationFactory.hunyuan(hunyuan)
        else:
            video_pipeline = VideoGenerationFactory.slideshow(images)

        return PipelineManager(
            llm=llm,
            tts=tts,
            video_pipeline=video_pipeline,
            assets=assets,
            script_analyzer=ScriptAnalyzerAgent(llm),
            video_planner=VideoPlanner(llm),
            quality_reviewer=QualityReviewAgent(llm),
        )

    @staticmethod
    def for_testing(
        llm_stub,
        image_stub,
        tts_stub,
        base_dir: str = "/tmp/test",
    ) -> PipelineManager:
        video_pipeline = VideoGenerationFactory.slideshow(image_stub)
        return PipelineManager(
            llm=llm_stub,
            tts=tts_stub,
            video_pipeline=video_pipeline,
            assets=AssetManager(base_dir=base_dir),
            script_analyzer=ScriptAnalyzerAgent(llm_stub),
            video_planner=VideoPlanner(llm_stub),
            quality_reviewer=QualityReviewAgent(llm_stub),
        )
