import base64
import time
import requests
from pathlib import Path


class HunyuanClient:
    """
    Wraps the RunPod Serverless API for HunyuanVideo inference.
    Compatible with the ashleykleynhans/hunyuan-video worker image.

    Worker input reference:
      https://github.com/ashleykleynhans/hunyuan-video-docker
    """

    def __init__(self, api_key: str, endpoint_id: str):
        if not endpoint_id:
            raise ValueError("HUNYUAN_ENDPOINT_ID is required when using the hunyuan backend.")
        self._endpoint_id = endpoint_id
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_clip(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        output_path: str,
    ) -> str:
        job_id = self._submit(prompt, width, height, num_frames)
        output = self._poll(job_id)
        return self._save(output, output_path)

    # ── private ──────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"https://api.runpod.io/v2/{self._endpoint_id}/{path}"

    def _submit(self, prompt: str, width: int, height: int, num_frames: int) -> str:
        payload = {
            "input": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": 30,
                "guidance_scale": 6.0,
                "flow_shift": 7.0,
                "embedded_guidance_scale": 6.0,
                "fps": 24,
                "seed": -1,       # -1 = random
            }
        }
        resp = requests.post(self._url("run"), json=payload, headers=self._headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return a job ID: {data}")
        return job_id

    def _poll(self, job_id: str, timeout: int = 1800, interval: int = 15) -> dict:
        """Poll until COMPLETED; returns the output dict."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(self._url(f"status/{job_id}"), headers=self._headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status == "COMPLETED":
                output = data.get("output")
                if not output:
                    raise RuntimeError(f"Job {job_id} completed but output is empty: {data}")
                return output

            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                error = data.get("error", "no error details")
                raise RuntimeError(f"RunPod job {job_id} ended with status {status}: {error}")

            # IN_QUEUE or IN_PROGRESS — keep polling
            time.sleep(interval)

        raise TimeoutError(f"RunPod job {job_id} did not complete within {timeout}s")

    def _save(self, output: dict, output_path: str) -> str:
        """Save the video from output dict (URL or base64) to output_path."""
        # Worker may return a URL or base64-encoded video
        video_url = output.get("video_url") or output.get("url")
        if video_url:
            resp = requests.get(video_url, timeout=300, stream=True)
            resp.raise_for_status()
            Path(output_path).write_bytes(resp.content)
            return output_path

        video_b64 = output.get("video") or output.get("video_base64")
        if video_b64:
            Path(output_path).write_bytes(base64.b64decode(video_b64))
            return output_path

        raise RuntimeError(
            f"HunyuanVideo output has no 'video_url' or 'video' field. "
            f"Keys returned: {list(output.keys())}"
        )
