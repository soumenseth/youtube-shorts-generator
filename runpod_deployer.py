import os
import requests
import time


class RunPodModelDeployer:
    def __init__(self, api_key: str | None = None):
        resolved_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not resolved_key:
            raise ValueError("RUNPOD_API_KEY not set. Add it to .env or pass it directly.")
        self.api_key = resolved_key
        self.base_url = "https://api.runpod.io/v2"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def deploy_model(self, model_path: str, pod_type: str = "secure", gpu_type: str = "NVIDIA_A100") -> str:
        url = f"{self.base_url}/pods"
        payload = {
            "name": "MyModelPod",
            "image": model_path,
            "gpuTypeId": gpu_type,
            "podType": pod_type,
        }
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json().get("id")

    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> bool:
        url = f"{self.base_url}/pods/{pod_id}"
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            if resp.json().get("status") == "RUNNING":
                return True
            time.sleep(5)
        return False

    def run_inference(self, pod_id: str, input_data: dict) -> dict:
        url = f"{self.base_url}/pods/{pod_id}/run"
        response = requests.post(url, json=input_data, headers=self.headers)
        response.raise_for_status()
        return response.json()
