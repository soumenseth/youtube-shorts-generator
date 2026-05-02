from pydantic import BaseModel, Field
from .video_types import VideoType, VisualStyle, VideoFormat


class ScriptAnalysis(BaseModel):
    video_type: VideoType
    visual_style: VisualStyle
    video_format: VideoFormat = VideoFormat.SHORTS
    target_audience: str
    tone: str
    key_themes: list[str]
    estimated_duration_seconds: int = Field(ge=15, le=60)
    hook_suggestion: str = Field(description="Compelling opening line for first 3 seconds")
    title_suggestion: str
    tags: list[str]
    reasoning: str
