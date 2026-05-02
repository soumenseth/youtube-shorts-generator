"""
Deploy HunyuanVideo on RunPod Serverless and write HUNYUAN_ENDPOINT_ID to .env.

Usage:
    uv run deploy_hunyuan.py

What this does:
  1. Creates a 100 GB network volume in US-TX-3 for model weights
     (models download automatically on first worker start — takes ~25 min)
  2. Creates a serverless pod template using the community HunyuanVideo worker image
  3. Creates the serverless endpoint (H100 SXM / A100 SXM, 0–3 workers)
  4. Writes HUNYUAN_ENDPOINT_ID to .env

GPU requirement: H100 80GB or A100 80GB — minimum for HunyuanVideo inference.
Estimated cost: ~$0.02–0.06 per generated clip (serverless pay-per-second).
"""

import os
import sys
import requests
from dotenv import load_dotenv, set_key

load_dotenv()

_GQL = "https://api.runpod.io/graphql"

# Community-maintained RunPod worker for HunyuanVideo.
# Source: https://github.com/ashleykleynhans/hunyuan-video-docker
_IMAGE = "ashleykleynhans/hunyuan-video:latest"

# RunPod GPU pool IDs — AMPERE_80 = A100 80 GB, ADA_80_PRO = RTX 6000 Ada 80 GB.
# Both meet HunyuanVideo's 80 GB VRAM requirement.
_GPU_IDS = "AMPERE_80,ADA_80_PRO"

_ENV_FILE = ".env"


def _gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        f"{_GQL}?api_key={api_key}",
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
    return payload["data"]


def create_network_volume(api_key: str, name: str = "hunyuan-models") -> str:
    """Create a 100 GB network volume. Returns volume ID."""
    print("  Creating network volume (100 GB) for model weights...")
    mutation = """
    mutation($input: CreateNetworkVolumeInput!) {
        createNetworkVolume(input: $input) { id name size dataCenterId }
    }
    """
    data = _gql(api_key, mutation, {
        "input": {
            "name": name,
            "size": 100,
            "dataCenterId": "US-TX-3",
        }
    })
    vol = data["createNetworkVolume"]
    print(f"  Network volume created: {vol['id']} ({vol['size']} GB, {vol['dataCenterId']})")
    return vol["id"]


def create_template(api_key: str, hf_token: str | None = None) -> str:
    """Create a serverless pod template using the HunyuanVideo worker image. Returns template ID."""
    print(f"  Creating pod template with image: {_IMAGE}")
    env = [
        {"key": "MODEL_BASE_PATH", "value": "/runpod-volume/models"},
        {"key": "HF_HOME",         "value": "/runpod-volume/huggingface"},
        {"key": "TRANSFORMERS_CACHE", "value": "/runpod-volume/huggingface"},
    ]
    if hf_token:
        env.append({"key": "HF_TOKEN", "value": hf_token})

    mutation = """
    mutation($input: SaveTemplateInput!) {
        saveTemplate(input: $input) { id name imageName isServerless }
    }
    """
    data = _gql(api_key, mutation, {
        "input": {
            "name": "HunyuanVideo Worker",
            "imageName": _IMAGE,
            "isServerless": True,
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "volumeMountPath": "/runpod-volume",
            "env": env,
            "dockerArgs": "",
            "ports": "",
        }
    })
    tmpl = data["saveTemplate"]
    print(f"  Template created: {tmpl['id']} ({tmpl['name']})")
    return tmpl["id"]


def create_endpoint(api_key: str, template_id: str, network_volume_id: str) -> dict:
    """Create the serverless endpoint. Returns endpoint dict with id."""
    print("  Creating serverless endpoint...")
    mutation = """
    mutation($input: EndpointInput!) {
        saveEndpoint(input: $input) { id name gpuIds workersMin workersMax idleTimeout }
    }
    """
    data = _gql(api_key, mutation, {
        "input": {
            "name": "hunyuan-video",
            "templateId": template_id,
            "gpuIds": _GPU_IDS,
            "networkVolumeId": network_volume_id,
            "locations": "US",
            "workersMin": 0,
            "workersMax": 3,
            "idleTimeout": 60,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    })
    ep = data["saveEndpoint"]
    print(f"  Endpoint created: {ep['id']} ({ep['name']})")
    return ep


def main() -> None:
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN") or None

    print("\n=== Deploying HunyuanVideo on RunPod Serverless ===\n")

    print("[1/3] Network volume")
    volume_id = create_network_volume(api_key)

    print("\n[2/3] Pod template")
    template_id = create_template(api_key, hf_token)

    print("\n[3/3] Serverless endpoint")
    endpoint = create_endpoint(api_key, template_id, volume_id)

    endpoint_id = endpoint["id"]

    # Write to .env
    set_key(_ENV_FILE, "HUNYUAN_ENDPOINT_ID", endpoint_id)
    set_key(_ENV_FILE, "VIDEO_BACKEND", "hunyuan")
    print(f"\nWrote to .env:")
    print(f"  HUNYUAN_ENDPOINT_ID={endpoint_id}")
    print(f"  VIDEO_BACKEND=hunyuan")

    print(f"""
=== Deployment complete ===

Endpoint ID : {endpoint_id}
GPU         : H100 SXM / A100 SXM (80 GB)
Workers     : 0 min → 3 max (scale-to-zero)
Idle timeout: 60 s

IMPORTANT — first cold start:
  The worker will download HunyuanVideo model weights (~80 GB) from
  HuggingFace into the network volume on the very first request.
  This takes approximately 20–30 minutes. Subsequent starts reuse
  the cached weights and take ~2–5 minutes.

To switch back to slideshow:
  Set VIDEO_BACKEND=slideshow in .env
""")


if __name__ == "__main__":
    main()
