from enum import Enum


class VideoType(str, Enum):
    EDUCATIONAL = "educational"
    STORYTELLING = "storytelling"
    HOW_TO = "how_to"
    ENTERTAINMENT = "entertainment"
    MOTIVATIONAL = "motivational"
    PRODUCT_REVIEW = "product_review"
    COMEDY = "comedy"
    NEWS = "news"
    TUTORIAL = "tutorial"


class VisualStyle(str, Enum):
    CINEMATIC = "cinematic"
    MINIMAL = "minimal"
    VIBRANT = "vibrant"
    DARK_MOODY = "dark_moody"
    BRIGHT_CHEERFUL = "bright_cheerful"
    PROFESSIONAL = "professional"
    ANIMATED = "animated"


class VideoFormat(str, Enum):
    SHORTS = "shorts"       # 1080x1920 vertical 9:16
    STANDARD = "standard"   # 1920x1080 horizontal 16:9


VIDEO_DIMENSIONS: dict[VideoFormat, tuple[int, int]] = {
    VideoFormat.SHORTS: (1080, 1920),
    VideoFormat.STANDARD: (1920, 1080),
}

# DALL-E 3 supported sizes closest to each format
DALLE_SIZES: dict[VideoFormat, str] = {
    VideoFormat.SHORTS: "1024x1792",
    VideoFormat.STANDARD: "1792x1024",
}

TTS_VOICES: dict[str, str] = {
    "educational": "nova",
    "motivational": "onyx",
    "storytelling": "shimmer",
    "how_to": "alloy",
    "tutorial": "alloy",
    "comedy": "fable",
    "entertainment": "echo",
    "news": "onyx",
    "product_review": "nova",
}
