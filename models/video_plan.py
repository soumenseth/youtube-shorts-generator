from __future__ import annotations
from pydantic import BaseModel, Field
from .video_types import VideoType, VisualStyle, VideoFormat


class Scene(BaseModel):
    scene_number: int
    narration: str = Field(description="Exact text to be spoken by TTS")
    visual_prompt: str = Field(description="Detailed DALL-E image generation prompt, no text in image")
    duration_seconds: int = Field(ge=2, le=30)
    text_overlay: str | None = Field(default=None, description="Short bold text shown on screen")
    transition: str = Field(default="fade", description="fade | cut | slide")


class VideoPlan(BaseModel):
    title: str
    description: str
    video_type: VideoType
    visual_style: VisualStyle
    video_format: VideoFormat
    target_audience: str
    hook: str = Field(description="The very first line to grab attention")
    scenes: list[Scene]
    background_music_style: str
    total_duration_seconds: int
    call_to_action: str
    thumbnail_concept: str


class QualityReview(BaseModel):
    round_number: int = Field(ge=1, le=3)
    overall_score: int = Field(ge=0, le=100)
    hook_score: int = Field(ge=0, le=100)
    content_value_score: int = Field(ge=0, le=100)
    retention_score: int = Field(ge=0, le=100)
    youtube_optimization_score: int = Field(ge=0, le=100)
    strengths: list[str]
    weaknesses: list[str]
    specific_improvements: list[str]
    reviewer_notes: str
    approved: bool
    revised_plan: VideoPlan | None = None
