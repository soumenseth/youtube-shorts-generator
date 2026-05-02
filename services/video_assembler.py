import os
from pathlib import Path
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips


class VideoAssembler:
    """Concatenates rendered VideoClip objects with narration audio into a final MP4."""

    def assemble(
        self,
        clips: list[VideoClip],
        audio_path: str,
        output_path: str,
    ) -> str:
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
