# YouTube Shorts Generator

An agentic AI pipeline that turns a text script into a fully produced YouTube video — narration audio, scene images, and a final MP4 — with zero manual intervention.

## How it works

The pipeline is driven by four Claude-powered agents that coordinate via an Anthropic tool-use loop:

```
Script
  │
  ▼
ScriptAnalyzerAgent  ── classifies video type, visual style, format, audience
  │
  ▼
VideoPlanner         ── generates scene-by-scene storyboard with narration + visual prompts
  │
  ▼
QualityReviewAgent   ── 3 rounds of scoring (hook, retention, YouTube algorithm fit)
  │                     revises plan if score < 75/100
  ▼
OrchestratorAgent    ── coordinates the above, then triggers asset generation
  │
  ▼
VideoGenerationPipeline
  ├── TTS narration  (OpenAI tts-1)
  ├── Scene images   (DALL-E 3 → HuggingFace FLUX.1-schnell fallback)
  └── MP4 assembly   (MoviePy)
```


## Project structure

```
├── agents/               # AI agents (base, script analyzer, planner, reviewer, orchestrator)
├── managers/             # PipelineManager (facade), AssetManager, PipelineFactory (DI root)
├── models/               # Pydantic models (ScriptAnalysis, VideoPlan, QualityReview)
├── protocols/            # Structural typing protocols (ILLMService, IImageGenerator, …)
├── services/
│   ├── image_backends/   # DALL-E, HuggingFace FLUX, fallback wrapper
│   ├── tts_backends/     # OpenAI TTS
│   ├── scene_renderer.py # SceneRenderer ABC + SlideshowSceneRenderer
│   ├── hunyuan_*.py      # HunyuanVideo renderer + RunPod client
│   ├── video_assembler.py
│   └── video_generation_factory.py  # Strategy registry (OCP-compliant)
├── design/               # Architecture docs and SOLID assessment
├── main.py
└── pyproject.toml
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- [ffmpeg](https://ffmpeg.org/download.html) on your PATH (required by MoviePy)
- API keys for Anthropic and OpenAI

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/soumenseth/youtube-shorts-generator.git
cd youtube-shorts-generator
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment variables**

Create a `.env` file in the project root:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional — enables HuggingFace FLUX as DALL-E fallback
HF_TOKEN=hf_...

# Optional — Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional — switch to HunyuanVideo backend (default: slideshow)
VIDEO_BACKEND=slideshow        # or: hunyuan
RUNPOD_API_KEY=...             # required if VIDEO_BACKEND=hunyuan
HUNYUAN_ENDPOINT_ID=...        # required if VIDEO_BACKEND=hunyuan
```

## Running

**From a script file:**

```bash
uv run main.py my_script.txt
```

**From a string directly:**

```bash
uv run main.py "Did you know honey never expires? Archaeologists found 3000-year-old honey in Egyptian tombs that was still perfectly edible..."
```

The pipeline prints progress as it runs:

```
Script loaded (218 chars)

[Orchestrator] Starting video generation pipeline
[Orchestrator] → analyze_script
[Orchestrator] ✓ Script analyzed: educational | cinematic | shorts
[Orchestrator] → plan_video
[Orchestrator] ✓ Video planned: 5 scenes, 45s
[Orchestrator] → review_and_approve_plan

  [Review 1/3] Focus: hook quality ...
  Score: 82/100 [████████░░]  Approved: True
  ...

[Orchestrator] → generate_and_assemble_video

[Pipeline] Generating narration audio...
[Pipeline] Generating scene images...
[VideoGenPipeline] Rendering 5 scenes via SlideshowSceneRenderer...
[VideoGenPipeline] Assembling final video...

==================================================
Done!  Video saved to: output/20260502_143021/final_video.mp4
==================================================
```

Output is saved under `output/<session_timestamp>/`:

```
output/20260502_143021/
├── images/
│   ├── scene_01.png
│   ├── scene_02.png
│   └── ...
├── audio/
│   └── narration.mp3
└── final_video.mp4
```

## Video backends

| Backend | Description | Env var |
|---|---|---|
| `slideshow` (default) | Static DALL-E 3 images with pan/fade | `VIDEO_BACKEND=slideshow` |
| `hunyuan` | AI-generated motion clips via HunyuanVideo on RunPod Serverless | `VIDEO_BACKEND=hunyuan` |

Adding a new backend requires only a new `SceneRenderer` subclass and one entry in the registry — no changes to the pipeline or agents.

## Observability

All agents and LLM calls are traced with [Langfuse](https://langfuse.com). Set the three `LANGFUSE_*` env vars to enable. Quality scores are recorded per review round so you can track plan improvement over time.
