import time
import requests
from pathlib import Path


class HunyuanClient:
    """
    Wraps the RunPod Serverless API for HunyuanVideo inference.
    Uses serverless (pay-per-second) rather than always-on pods — correct
    for on-demand per-clip generation with unpredictable request rates.
    """

    _BASE = "https://api.runpod.io/v2/{endpoint_id}"

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
        video_url = self._poll(job_id)
        return self._download(video_url, output_path)

    # ── private ──────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._BASE.format(endpoint_id=self._endpoint_id)}/{path}"

    def _submit(self, prompt: str, width: int, height: int, num_frames: int) -> str:
        payload = {
            "input": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": 50,
                "guidance_scale": 7.0,
            }
        }
        resp = requests.post(self._url("run"), json=payload, headers=self._headers, timeout=30)
        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return a job ID: {resp.json()}")
        return job_id

    def _poll(self, job_id: str, timeout: int = 600, interval: int = 10) -> str:
        """Poll until job is COMPLETED, return video download URL."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(self._url(f"status/{job_id}"), headers=self._headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "COMPLETED":
                output = data.get("output", {})
                url = output.get("video_url") or output.get("url")
                if not url:
                    raise RuntimeError(f"Job completed but no video_url in output: {output}")
                return url
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"RunPod job {job_id} ended with status: {status}")
            time.sleep(interval)
        raise TimeoutError(f"RunPod job {job_id} did not complete within {timeout}s")

    def _download(self, url: str, output_path: str) -> str:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)
        return output_path
