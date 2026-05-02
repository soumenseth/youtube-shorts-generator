import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from models.video_plan import VideoPlan
from models.video_types import VIDEO_DIMENSIONS
from services.image_prep import resize_crop, add_overlay


class VideoAssemblyService:
    """Assembles scene images + narration audio into final MP4 via MoviePy."""

    def assemble(
        self,
        plan: VideoPlan,
        image_paths: list[str],
        audio_path: str,
        output_path: str,
    ) -> str:
        dims = VIDEO_DIMENSIONS[plan.video_format]

        clips = []
        for scene, img_path in zip(plan.scenes, image_paths):
            with self._prepared_image(img_path, dims, scene.text_overlay) as prep:
                clip = ImageClip(prep).set_duration(scene.duration_seconds)
                clip = clip.fadein(0.3).fadeout(0.3)
                clips.append(clip)

        video = concatenate_videoclips(clips, method="compose")

        if os.path.exists(audio_path):
            audio = AudioFileClip(audio_path)
            if audio.duration > video.duration:
                audio = audio.subclip(0, video.duration)
            video = video.set_audio(audio)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        return output_path

    @contextmanager
    def _prepared_image(
        self,
        image_path: str,
        target: tuple[int, int],
        text_overlay: str | None,
    ):
        """Context manager: yields a prepared temp image path, cleans up on exit."""
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
