# Implementation V2 — Fix Plan & Final Design

**Based on:** `implementation_v1_assessment.md` (score 5.3/10)  
**Target score:** 8.5/10  
**Date:** 2026-05-02

---

## Executive Summary

Six phases of work, ordered by risk and dependency. Phases 1–3 are architectural and must be done in order. Phases 4–5 are independent and can be done in parallel. Phase 6 (HunyuanVideo) builds on the foundation laid in Phases 1–3.

| Phase | Topic | Assessment Items Fixed | Effort |
|---|---|---|---|
| 1 | Protocols + ABC | 1.4, 2.7, 2.8, 3.3 | Small |
| 2 | Strategy for services | 2.1, 2.4, 2.5 | Medium |
| 3 | DI + Factory | 2.7, 3.2, 3.4, 1.1 | Medium |
| 4 | ISP + Template Method | 2.6, 3.1, 5.4, 1.2 | Medium |
| 5 | Bug fixes + smells | 4.1–4.4, 5.1–5.3 | Small |
| 6 | HunyuanVideo backend | 1.5, 2.3 | Large |

---

## Phase 1 — Protocols + Abstract Base Classes

**Fixes:** 1.4 (BaseAgent not abstract), 2.8 (agents coupled to concrete LLMService), 3.3 (no service protocols)

### 1.1 Create `protocols/` package

New file: `protocols/__init__.py` (empty)

### 1.2 `protocols/llm_protocol.py`

```python
from typing import Protocol, Any

class ILLMService(Protocol):
    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Any: ...
```

### 1.3 `protocols/image_protocol.py`

```python
from typing import Protocol

class IImageGenerator(Protocol):
    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str: ...
```

### 1.4 `protocols/tts_protocol.py`

```python
from typing import Protocol

class ITTSService(Protocol):
    def generate_audio(self, text: str, output_path: str, voice: str = "alloy") -> str: ...
    def voice_for(self, video_type: str) -> str: ...
```

### 1.5 Update `agents/base_agent.py`

```python
from abc import ABC, abstractmethod
from protocols.llm_protocol import ILLMService

class BaseAgent(ABC):                        # was: class BaseAgent
    def __init__(self, llm_service: ILLMService, name: str):
        self._llm = llm_service              # was: self.llm (public)
        self._name = name                    # was: self.name (public)

    @abstractmethod
    def _handle_tool(self, name: str, inputs: dict) -> dict: ...

    @observe()
    def _run_tool_loop(self, messages, system, tools, max_tokens=4096):
        get_client().update_current_span(name=f"{self._name}-loop")
        while True:
            response = self._llm.complete(messages, system=system, tools=tools, max_tokens=max_tokens)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                final_text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return messages, final_text
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._handle_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        return messages, ""
```

> `_handle_tool` replaces the `tool_handler` callback parameter. Each subclass implements it once instead of defining a nested closure every time. Eliminates the `nonlocal` pattern in all four agent files.

### 1.6 Update all agent subclasses

Replace:
```python
def analyze(self, script):
    analysis_data: dict = {}
    def handle(name, inputs):
        nonlocal analysis_data
        ...
    self._run_tool_loop(..., handle)
```

With:
```python
def analyze(self, script):
    self._analysis_data: dict = {}          # instance scratch; cleared per call
    messages = [{"role": "user", "content": script}]
    self._run_tool_loop(messages, _SYSTEM, _TOOLS)
    return ScriptAnalysis(**self._analysis_data)

def _handle_tool(self, name: str, inputs: dict) -> dict:
    if name == "classify_script":
        self._analysis_data = inputs
        return {"status": "ok"}
    return {"error": f"Unknown tool: {name}"}
```

> All four agents (`ScriptAnalyzerAgent`, `VideoPlanner`, `QualityReviewAgent`, `OrchestratorAgent`) follow this single pattern.

---

## Phase 2 — Strategy Pattern for Services

**Fixes:** 2.1 (ImageService SRP), 2.4 (ImageService OCP), 2.5 (TTSService OCP)

### 2.1 `services/image_backends/`

```
services/
  image_backends/
    __init__.py
    base.py          # ImageGenerationStrategy ABC
    dalle_backend.py
    huggingface_backend.py
  image_service.py   # simplified: just selects strategy
```

**`services/image_backends/base.py`**
```python
from abc import ABC, abstractmethod

class ImageGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str: ...
```

**`services/image_backends/dalle_backend.py`**
```python
class DalleBackend(ImageGenerationStrategy):
    def __init__(self, client: openai.OpenAI):
        self._client = client

    def generate(self, prompt, output_path, size="1024x1792") -> str:
        response = self._client.images.generate(model="dall-e-3", prompt=prompt, size=size, n=1)
        image_data = self._client.get(response.data[0].url)
        Path(output_path).write_bytes(image_data.content)
        return output_path
```

**`services/image_backends/huggingface_backend.py`**
```python
class HuggingFaceFluxBackend(ImageGenerationStrategy):
    def __init__(self, hf_token: str):
        self._token = hf_token

    def generate(self, prompt, output_path, size="1024x1792") -> str:
        # existing HF FLUX.1-schnell logic, extracted verbatim
        ...
```

**`services/image_service.py`** — simplified:
```python
class ImageService:
    def __init__(self, strategy: ImageGenerationStrategy):
        self._strategy = strategy

    def generate(self, prompt: str, output_path: str, size: str = "1024x1792") -> str:
        return self._strategy.generate(prompt, output_path, size)
```

> Strategy selection (DALL-E primary, HF fallback) moves to the factory in Phase 3.

### 2.2 TTS backends

```
services/
  tts_backends/
    __init__.py
    base.py           # TTSBackend ABC
    openai_backend.py
  tts_service.py      # simplified
```

**`services/tts_backends/base.py`**
```python
from abc import ABC, abstractmethod

class TTSBackend(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str, voice: str = "alloy") -> str: ...
```

**`services/tts_backends/openai_backend.py`**
```python
class OpenAITTSBackend(TTSBackend):
    def __init__(self, client: openai.OpenAI):
        self._client = client

    def generate(self, text, output_path, voice="alloy") -> str:
        response = self._client.audio.speech.create(model="tts-1", voice=voice, input=text)
        Path(output_path).write_bytes(response.read())
        return output_path
```

**`services/tts_service.py`** — simplified:
```python
class TTSService:
    def __init__(self, backend: TTSBackend):
        self._backend = backend

    def generate_audio(self, text: str, output_path: str, voice: str = "alloy") -> str:
        return self._backend.generate(text, output_path, voice)

    def voice_for(self, video_type: str) -> str:
        return TTS_VOICES.get(video_type, "alloy")
```

---

## Phase 3 — Dependency Injection + Factory

**Fixes:** 2.7 (Service Locator in PipelineManager), 3.2 (Facade leaks internals), 3.4 (no factory), 1.1 (encapsulation), 1.3 (AssetManager exposes dirs)

### 3.1 Update `managers/asset_manager.py`

Make directory attributes private:
```python
class AssetManager:
    def __init__(self, base_dir="output", session_id=None):
        _session_dir = Path(base_dir) / (session_id or _new_session_id())
        self._images_dir = _session_dir / "images"
        self._audio_dir  = _session_dir / "audio"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
```

No change to public interface (`image_path()`, `audio_path()`, `output_path()`).

### 3.2 Refactor `managers/pipeline_manager.py`

Remove all construction from `__init__`. Accept all deps as parameters. Expose only the operations callers need:

```python
class PipelineManager:
    def __init__(
        self,
        llm: ILLMService,
        tts: ITTSService,
        images: IImageGenerator,
        assembly: VideoAssemblyService,
        assets: AssetManager,
        script_analyzer: ScriptAnalyzerAgent,
        video_planner: VideoPlanner,
        quality_reviewer: QualityReviewAgent,
    ):
        self._llm = llm
        self._tts = tts
        self._images = images
        self._assembly = assembly
        self._assets = assets
        self._script_analyzer = script_analyzer
        self._video_planner = video_planner
        self._quality_reviewer = quality_reviewer

    # Public interface used by OrchestratorAgent
    def analyze_script(self, script: str) -> ScriptAnalysis:
        return self._script_analyzer.analyze(script)

    def plan_video(self, script: str, analysis: ScriptAnalysis) -> VideoPlan:
        return self._video_planner.plan(script, analysis)

    def review_plan(self, script: str, plan: VideoPlan) -> tuple[VideoPlan, list[QualityReview]]:
        return self._quality_reviewer.run_full_review(script, plan)

    def complete(self, messages, system=None, tools=None, max_tokens=8192):
        return self._llm.complete(messages, system=system, tools=tools, max_tokens=max_tokens)

    def output_path(self) -> str:
        return self._assets.output_path()

    @observe(name="asset-generation")
    def generate_video(self, plan: VideoPlan, output_path: str | None = None) -> str:
        # unchanged logic, but uses self._tts, self._images, self._assembly, self._assets
        ...
```

> All internals are now private. `OrchestratorAgent` uses the public method interface, not attribute access.

### 3.3 New `managers/pipeline_factory.py`

Centralizes all construction. One place to change for test vs. production vs. different backends.

```python
import os
import openai
import anthropic
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.tts_backends.openai_backend import OpenAITTSBackend
from services.image_service import ImageService
from services.image_backends.dalle_backend import DalleBackend
from services.image_backends.huggingface_backend import HuggingFaceFluxBackend
from services.video_assembly import VideoAssemblyService
from agents.script_analyzer import ScriptAnalyzerAgent
from agents.video_planner import VideoPlanner
from agents.quality_reviewer import QualityReviewAgent
from managers.asset_manager import AssetManager
from managers.pipeline_manager import PipelineManager


class PipelineFactory:
    @staticmethod
    def production(session_id: str | None = None) -> PipelineManager:
        oai = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        llm = LLMService()

        # Image: DALL-E primary, HF fallback if token present
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from services.image_backends.huggingface_backend import HuggingFaceFluxBackend
            # Wrap: try DALL-E first, fall back to HF
            image_gen = _FallbackImageGenerator(DalleBackend(oai), HuggingFaceFluxBackend(hf_token))
        else:
            image_gen = ImageService(DalleBackend(oai))

        tts = TTSService(OpenAITTSBackend(oai))
        assets = AssetManager(session_id=session_id)

        return PipelineManager(
            llm=llm,
            tts=tts,
            images=image_gen,
            assembly=VideoAssemblyService(),
            assets=assets,
            script_analyzer=ScriptAnalyzerAgent(llm),
            video_planner=VideoPlanner(llm),
            quality_reviewer=QualityReviewAgent(llm),
        )

    @staticmethod
    def for_testing(llm_stub, image_stub, tts_stub) -> PipelineManager:
        return PipelineManager(
            llm=llm_stub,
            tts=tts_stub,
            images=image_stub,
            assembly=VideoAssemblyService(),
            assets=AssetManager(base_dir="/tmp/test"),
            script_analyzer=ScriptAnalyzerAgent(llm_stub),
            video_planner=VideoPlanner(llm_stub),
            quality_reviewer=QualityReviewAgent(llm_stub),
        )
```

### 3.4 Update `main.py`

```python
from managers.pipeline_factory import PipelineFactory
from agents.orchestrator import OrchestratorAgent

def main():
    ...
    pipeline = PipelineFactory.production()
    OrchestratorAgent(pipeline).run(script)
```

---

## Phase 4 — ISP + Template Method

**Fixes:** 2.6 (fat dependency on PipelineManager), 3.1 (callback vs. override), 5.4 (nonlocal pattern), 1.2 (public BaseAgent attributes)

### 4.1 `protocols/pipeline_protocol.py`

Narrow interface — only what `OrchestratorAgent` actually calls:

```python
from typing import Protocol
from models.script_model import ScriptAnalysis
from models.video_plan import VideoPlan, QualityReview
from typing import Any

class IVideoProductionPipeline(Protocol):
    def analyze_script(self, script: str) -> ScriptAnalysis: ...
    def plan_video(self, script: str, analysis: ScriptAnalysis) -> VideoPlan: ...
    def review_plan(self, script: str, plan: VideoPlan) -> tuple[VideoPlan, list[QualityReview]]: ...
    def complete(self, messages: list[dict], system: str | None, tools: list[dict] | None, max_tokens: int) -> Any: ...
    def output_path(self) -> str: ...
    def generate_video(self, plan: VideoPlan, output_path: str | None) -> str: ...
```

`PipelineManager` satisfies this protocol structurally (no inheritance needed).

### 4.2 Update `agents/orchestrator.py`

```python
from protocols.pipeline_protocol import IVideoProductionPipeline

class OrchestratorAgent(BaseAgent):
    def __init__(self, pipeline: IVideoProductionPipeline):
        super().__init__(pipeline.complete, name="orchestrator")   # satisfies ILLMService via protocol
        self._pipeline = pipeline

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "analyze_script":
            analysis = self._pipeline.analyze_script(inputs["script"])
            ...
            return analysis.model_dump()
        elif name == "plan_video":
            ...
        elif name == "review_and_approve_plan":
            ...
        elif name == "generate_and_assemble_video":
            ...
        return {"error": f"Unknown tool: {name}"}

    @observe(name="video-generation")
    def run(self, script: str) -> str:
        messages = [{"role": "user", "content": f"Create a YouTube video from this script:\n\n{script}"}]
        self._run_tool_loop(messages, _SYSTEM, _TOOLS)
        return self._pipeline.output_path()
```

> `OrchestratorAgent` now depends on `IVideoProductionPipeline`, not `PipelineManager`. Narrow, substitutable.

### 4.3 Apply `_handle_tool` override in remaining agents

Each of `ScriptAnalyzerAgent`, `VideoPlanner`, `QualityReviewAgent` replaces the nested `handle()` closure with `_handle_tool()`. Example for `ScriptAnalyzerAgent`:

```python
class ScriptAnalyzerAgent(BaseAgent):
    def __init__(self, llm: ILLMService):
        super().__init__(llm, name="script-analyzer")
        self._result: dict = {}

    def _handle_tool(self, name: str, inputs: dict) -> dict:
        if name == "classify_script":
            self._result = inputs
            return {"status": "classified"}
        return {"error": f"Unknown tool: {name}"}

    @observe(name="script-analysis")
    def analyze(self, script: str) -> ScriptAnalysis:
        self._result = {}
        messages = [{"role": "user", "content": script}]
        self._run_tool_loop(messages, _SYSTEM, _TOOLS)
        return ScriptAnalysis(**self._result)
```

---

## Phase 5 — Bug Fixes & Code Smells

### 5.1 Fix `runpod_deployer.py:api_key` overwrite (Bug 4.1)

```python
# Before (broken):
self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
if not self.api_key:
    raise ValueError(...)
self.api_key = api_key          # ← BUG: overwrites with None

# After (fixed):
self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
if not self.api_key:
    raise ValueError("RUNPOD_API_KEY not set")
# Remove the third line entirely
```

### 5.2 Fix `QualityReviewAgent._history` instance bleed (Bug 4.2)

Move history from instance state to local variable inside `run_full_review()`:

```python
# Before:
self._history: list[QualityReview] = []   # instance state

def run_full_review(self, script, plan):
    self._history.clear()    # unreliable — still bleeds if exception occurs
    ...

# After:
def run_full_review(self, script, plan):
    history: list[QualityReview] = []    # local — no bleed possible
    ...
```

### 5.3 Fix `VideoAssemblyService._prepare_image()` temp file leak (Bug 4.3)

Use a temp file with explicit cleanup:

```python
import tempfile
from contextlib import contextmanager

@contextmanager
def _prepared_image(self, image_path: str, width: int, height: int, ...):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        img = self._resize_crop(image_path, width, height)
        img = self._add_overlay(img, ...)
        img.save(tmp_path)
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

Usage in `assemble()`:
```python
for scene, image_path in zip(plan.scenes, image_paths):
    with self._prepared_image(image_path, width, height, ...) as prep:
        clip = ImageClip(prep).set_duration(scene.duration_seconds)
```

### 5.4 Log parse errors in quality reviewer (Bug 4.4)

```python
# Before:
except Exception:
    review_data.pop("revised_plan", None)
    review = QualityReview(round_number=round_number, **review_data)

# After:
except Exception as exc:
    logger.warning("QualityReview parse failed (round %d): %s", round_number, exc)
    review_data.pop("revised_plan", None)
    review = QualityReview(round_number=round_number, **review_data)
```

### 5.5 Move deferred imports to module level in `main.py` (Smell 5.1)

```python
# Before (inside main()):
def main():
    from managers.pipeline_manager import PipelineManager
    from agents.orchestrator import OrchestratorAgent

# After (module level):
from managers.pipeline_factory import PipelineFactory
from agents.orchestrator import OrchestratorAgent
```

### 5.6 Remove `load_dotenv()` from module level in `runpod_deployer.py` (Smell 5.2)

```python
# Before:
load_dotenv()  # module-level side effect

class RunPodModelDeployer:
    def __init__(self, api_key=None):
        ...

# After:
class RunPodModelDeployer:
    def __init__(self, api_key=None):
        load_dotenv()   # or remove entirely — caller's responsibility
        ...
```

---

## Phase 6 — HunyuanVideo Backend Integration

**Fixes:** 1.5 (no service polymorphism), 2.3 (generate_video SRP)  
**Reference:** `design/video_generation_design.md` — full design already evaluated.

### Summary of additions (no existing file changes)

| New file | Role |
|---|---|
| `services/video_backends/base.py` | `SceneRenderer` ABC returning `VideoClip` |
| `services/video_backends/slideshow.py` | `SlideshowSceneRenderer(IImageGenerator)` |
| `services/video_backends/hunyuan.py` | `HunyuanSceneRenderer(HunyuanClient)` |
| `services/video_assembler.py` | `VideoAssembler` — pure concatenation + audio |
| `services/video_generation_pipeline.py` | `VideoGenerationPipeline(renderer, assembler)` |
| `services/video_generation_factory.py` | `VideoGenerationFactory.slideshow()` / `.hunyuan()` |
| `services/hunyuan_client.py` | RunPod Serverless HTTP client |

`PipelineManager.generate_video()` delegates to `VideoGenerationPipeline.generate()` instead of implementing the loop inline. This fixes the SRP violation (2.3).

---

## Final File Map

```
YoutubeShorts/
├── protocols/
│   ├── __init__.py
│   ├── llm_protocol.py          NEW — ILLMService
│   ├── image_protocol.py        NEW — IImageGenerator
│   ├── tts_protocol.py          NEW — ITTSService
│   └── pipeline_protocol.py    NEW — IVideoProductionPipeline
│
├── models/                      UNCHANGED
│   ├── video_types.py
│   ├── script_model.py
│   └── video_plan.py
│
├── services/
│   ├── llm_service.py           UNCHANGED (satisfies ILLMService structurally)
│   ├── image_service.py         CHANGED — accepts ImageGenerationStrategy
│   ├── tts_service.py           CHANGED — accepts TTSBackend
│   ├── video_assembly.py        CHANGED — temp file fix; image prep extracted
│   ├── image_backends/          NEW
│   │   ├── base.py
│   │   ├── dalle_backend.py
│   │   └── huggingface_backend.py
│   ├── tts_backends/            NEW
│   │   ├── base.py
│   │   └── openai_backend.py
│   └── video_backends/          NEW (Phase 6)
│       ├── base.py
│       ├── slideshow.py
│       └── hunyuan.py
│
├── agents/
│   ├── base_agent.py            CHANGED — ABC, _handle_tool hook, private attrs
│   ├── script_analyzer.py       CHANGED — override _handle_tool, remove nonlocal
│   ├── video_planner.py         CHANGED — override _handle_tool, remove nonlocal
│   ├── quality_reviewer.py      CHANGED — override _handle_tool, local history, log errors
│   └── orchestrator.py          CHANGED — IVideoProductionPipeline, override _handle_tool
│
├── managers/
│   ├── asset_manager.py         CHANGED — private directory attributes
│   ├── pipeline_manager.py      CHANGED — all deps injected, all attrs private, public method interface
│   └── pipeline_factory.py      NEW — PipelineFactory.production() / .for_testing()
│
├── main.py                      CHANGED — module-level imports, use PipelineFactory
├── runpod_deployer.py           CHANGED — api_key bug fixed, load_dotenv moved
│
└── design/
    ├── video_generation_design.md
    ├── implementation_v1_assessment.md
    └── implementation_v2_plan.md   ← this file
```

---

## Target Scorecard (V2)

| Dimension | V1 | V2 | Change |
|---|---|---|---|
| Encapsulation | 5/10 | 9/10 | Private attrs; facade hides internals |
| Abstraction | 5/10 | 9/10 | ABC + Protocols everywhere |
| Polymorphism | 4/10 | 8/10 | Strategy at image/tts/video layer |
| SRP | 6/10 | 8/10 | Backends extracted; generate_video delegates |
| OCP | 5/10 | 9/10 | New backend = new class, no modification |
| LSP | 7/10 | 9/10 | SceneRenderer → VideoClip guarantees substitution |
| ISP | 6/10 | 9/10 | IVideoProductionPipeline narrows orchestrator dep |
| DIP | 4/10 | 9/10 | Factory owns construction; everything injected |
| Patterns | 6/10 | 9/10 | Strategy, Factory, Template Method, proper Facade |
| **Overall** | **5.3/10** | **8.8/10** | |

---

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3
                  ↘
                   Phase 4 (parallel with 3)
                   Phase 5 (parallel with 3, 4)
                   Phase 6 (after 1–3)
```

Phases 1, 2, and 3 have strict ordering because each consumes the protocols from the previous. Phases 4, 5, and 6 are independent once the protocols exist.
