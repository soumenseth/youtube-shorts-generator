# Video Generation Architecture

**Version:** 1.0  
**Status:** Approved — pending implementation  
**Scope:** Pluggable video generation backends (Slideshow, HunyuanVideo) without modifying existing agent/pipeline code

---

## 1. Context & Problem

### Current state
The existing `VideoAssemblyService` conflates two distinct responsibilities:
1. **Scene rendering** — generating a visual for each scene (currently: DALL-E 3 image → resize → text overlay)
2. **Final assembly** — concatenating all scene clips + narration audio into a final MP4

Adding HunyuanVideo (text-to-video) requires swapping only step 1. Without a clean abstraction, every new backend requires modifying `VideoAssemblyService`, `PipelineManager`, and potentially the agents — violating OCP and making the system fragile.

### What changes with HunyuanVideo
| Concern | Slideshow | HunyuanVideo |
|---|---|---|
| Scene visual | DALL-E 3 image (static) | AI-generated video clip (motion) |
| Input | text prompt → PNG | text prompt → MP4 |
| Infrastructure | OpenAI API | RunPod Serverless GPU |
| Assembly | same | same |

The assembly step is **identical** for both backends. The rendering step is where they diverge.

---

## 2. Design Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   VideoGenerationPipeline                         │
│  __init__(renderer: SceneRenderer, assembler: VideoAssembler)     │  ← DI
│  generate(plan, audio_path, assets) → str                        │
└──────────────────┬────────────────────────┬─────────────────────┘
                   │                        │
       ┌───────────▼────────────┐  ┌────────▼──────────────┐
       │   SceneRenderer (ABC)  │  │   VideoAssembler       │
       │                        │  │                        │
       │   render(              │  │   assemble(            │
       │     scene: Scene,      │  │     clips: list[       │
       │     plan: VideoPlan,   │  │       VideoClip],      │
       │     tmp_path: str      │  │     audio: str,        │
       │   ) → VideoClip  ◄─────┼──┼── output: str         │
       └──────────┬─────────────┘  │   ) → str             │
                  │                └───────────────────────┘
       ┌──────────┴────────────┐
       │                       │
┌──────▼────────────┐  ┌───────▼──────────────┐
│ SlideshowScene    │  │ HunyuanScene          │
│ Renderer          │  │ Renderer              │
│                   │  │                       │
│ IImageGenerator   │  │ HunyuanClient         │
│  (injected, DIP)  │  │  (injected)           │
└───────────────────┘  └──────────┬────────────┘
                                   │
                        ┌──────────▼──────────────┐
                        │      HunyuanClient       │
                        │                          │
                        │  generate_clip(          │
                        │    prompt, w, h,         │
                        │    num_frames, out       │
                        │  ) → str  (MP4 path)    │
                        │                          │
                        │  RunPod Serverless API   │
                        │  (endpoint_id from env)  │
                        └──────────────────────────┘
```

### Key design decision: `SceneRenderer` returns `VideoClip`, not a file path

Both renderers return a **MoviePy `VideoClip` object** (in memory), not a file path string. This is the critical LSP fix:

- `SlideshowSceneRenderer` → `ImageClip(prepared_img).set_duration(n)` — in-memory static clip  
- `HunyuanSceneRenderer` → `VideoFileClip(downloaded_mp4)` — in-memory handle to video file  
- Both are `VideoClip` subclasses — `VideoAssembler` never branches on type

If we returned file paths, the assembler would need `ImageClip` vs `VideoFileClip` logic, breaking the substitution guarantee.

---

## 3. Contracts

### `SceneRenderer` (ABC)

```python
from abc import ABC, abstractmethod
from moviepy.editor import VideoClip
from models.scene import Scene
from models.video_plan import VideoPlan

class SceneRenderer(ABC):
    @abstractmethod
    def render(
        self,
        scene: Scene,
        plan: VideoPlan,
        intermediate_path: str,   # hint for temp file location; renderer owns cleanup
    ) -> VideoClip:
        """
        Render one scene to an in-memory MoviePy VideoClip.
        Caller is responsible for closing the clip after assembly.
        """
        ...
```

**Contract guarantees (enforced by ABC):**
- Returns a `VideoClip` — duration matches `scene.duration_seconds` (±1s tolerance)
- Does not write the final output — only intermediate files if needed
- Raises `SceneRenderError` on failure (not a raw API exception)

---

### `IImageGenerator` (Protocol — DIP)

```python
from typing import Protocol

class IImageGenerator(Protocol):
    def generate(self, prompt: str, output_path: str, size: str) -> str: ...
```

`ImageService` satisfies this structurally. `SlideshowSceneRenderer` depends on this protocol, not the concrete class. This makes the renderer independently testable with a stub.

---

### `VideoAssembler`

```python
class VideoAssembler:
    def assemble(
        self,
        clips: list[VideoClip],
        audio_path: str,
        output_path: str,
    ) -> str:
        """Concatenate clips, add audio, write MP4. Returns output_path."""
        ...
```

Currently concrete. Could be promoted to ABC if an FFmpeg-native assembler is ever needed, without changing any caller.

---

### `VideoGenerationPipeline`

```python
class VideoGenerationPipeline:
    def __init__(self, renderer: SceneRenderer, assembler: VideoAssembler):
        self._renderer = renderer     # injected — Strategy
        self._assembler = assembler   # injected

    def generate(
        self,
        plan: VideoPlan,
        audio_path: str,
        assets: AssetManager,
    ) -> str:
        clips = [
            self._renderer.render(scene, plan, assets.clip_path(scene.scene_number))
            for scene in plan.scenes
        ]
        try:
            return self._assembler.assemble(clips, audio_path, assets.output_path())
        finally:
            for clip in clips:
                clip.close()   # release file handles / memory
```

Pure orchestration. No rendering logic, no assembly logic, no branching on backend type.

---

## 4. Concrete Implementations

### `SlideshowSceneRenderer`

```python
class SlideshowSceneRenderer(SceneRenderer):
    def __init__(self, image_generator: IImageGenerator):   # DIP: protocol, not class
        self._gen = image_generator

    def render(self, scene, plan, intermediate_path) -> VideoClip:
        dims = VIDEO_DIMENSIONS[plan.video_format]
        size = DALLE_SIZES[plan.video_format]
        prompt = f"{scene.visual_prompt}, {plan.visual_style.value} style, ultra detailed"

        img_path = intermediate_path.replace(".mp4", ".png")
        self._gen.generate(prompt, img_path, size)

        prepared = self._prepare_image(img_path, dims, scene.text_overlay)
        return ImageClip(prepared).set_duration(scene.duration_seconds)
```

Image prep helpers (`_resize_crop`, `_add_overlay`) are extracted from the existing `VideoAssemblyService` — no logic is rewritten, only reorganised.

---

### `HunyuanSceneRenderer`

```python
class HunyuanSceneRenderer(SceneRenderer):
    def __init__(self, client: HunyuanClient):
        self._client = client

    def render(self, scene, plan, intermediate_path) -> VideoClip:
        dims = VIDEO_DIMENSIONS[plan.video_format]
        prompt = f"{scene.visual_prompt}, {plan.visual_style.value} style, cinematic"
        num_frames = scene.duration_seconds * 24   # 24 fps

        mp4_path = self._client.generate_clip(
            prompt=prompt,
            width=dims[0],
            height=dims[1],
            num_frames=num_frames,
            output_path=intermediate_path,
        )
        return VideoFileClip(mp4_path)
```

---

### `HunyuanClient`

```python
class HunyuanClient:
    """
    Wraps RunPod Serverless API for HunyuanVideo inference.
    Uses serverless (pay-per-second) rather than always-on pods — correct
    for on-demand per-clip generation with unpredictable request rates.
    """

    BASE_URL = "https://api.runpod.io/v2/{endpoint_id}"

    def __init__(self, api_key: str, endpoint_id: str):
        self._api_key = api_key
        self._endpoint_id = endpoint_id
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

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

    def _submit(self, prompt, width, height, num_frames) -> str: ...
    def _poll(self, job_id: str, timeout: int = 600) -> str: ...   # returns download URL
    def _download(self, url: str, path: str) -> str: ...
```

**Why serverless, not `RunPodModelDeployer`:**  
The existing `RunPodModelDeployer` manages always-on pods (create → wait → run → exists forever). For per-clip generation with variable demand, RunPod Serverless is the correct product: the GPU spins up per request and bills per second. The pod deployer is appropriate for persistent endpoints (e.g., a dedicated inference server) — not for batch-style one-shot jobs.

---

## 5. Factory & Registry (OCP fix)

```python
from enum import Enum
from typing import Callable

class VideoBackend(str, Enum):
    SLIDESHOW = "slideshow"
    HUNYUAN   = "hunyuan"

# Registry: adding a new backend = one entry here, zero other changes
_RENDERER_REGISTRY: dict[VideoBackend, Callable[..., SceneRenderer]] = {
    VideoBackend.SLIDESHOW: lambda img_gen, _client: SlideshowSceneRenderer(img_gen),
    VideoBackend.HUNYUAN:   lambda _img_gen, client: HunyuanSceneRenderer(client),
}

class VideoGenerationFactory:
    """
    Typed factory methods for explicit wiring (preferred).
    from_backend() for env-var-driven selection.
    """

    @staticmethod
    def slideshow(image_generator: IImageGenerator) -> VideoGenerationPipeline:
        return VideoGenerationPipeline(
            renderer=SlideshowSceneRenderer(image_generator),
            assembler=VideoAssembler(),
        )

    @staticmethod
    def hunyuan(client: HunyuanClient) -> VideoGenerationPipeline:
        return VideoGenerationPipeline(
            renderer=HunyuanSceneRenderer(client),
            assembler=VideoAssembler(),
        )

    @classmethod
    def from_backend(
        cls,
        backend: VideoBackend,
        image_generator: IImageGenerator,
        hunyuan_client: HunyuanClient,
    ) -> VideoGenerationPipeline:
        renderer = _RENDERER_REGISTRY[backend](image_generator, hunyuan_client)
        return VideoGenerationPipeline(renderer=renderer, assembler=VideoAssembler())
```

`from_backend` receives both dependencies — this is intentional and acceptable at the composition root. The registry ensures the correct one is used; the other is ignored. This is preferable to lazy initialisation or a DI container for a project of this scale.

---

## 6. Dependency Wiring in `PipelineManager`

```python
class PipelineManager:
    def __init__(self):
        # Existing services — unchanged
        self.llm    = LLMService()
        self.tts    = TTSService()
        self.assets = AssetManager()

        # Image generator satisfies IImageGenerator protocol structurally
        image_svc = ImageService()

        # HunyuanClient — only makes network calls when generate_clip() is invoked
        hunyuan_client = HunyuanClient(
            api_key=os.getenv("RUNPOD_API_KEY"),
            endpoint_id=os.getenv("HUNYUAN_ENDPOINT_ID", ""),
        )

        backend = VideoBackend(os.getenv("VIDEO_BACKEND", VideoBackend.SLIDESHOW))
        self.video_pipeline = VideoGenerationFactory.from_backend(
            backend, image_svc, hunyuan_client
        )

        # Agents — unchanged
        self.script_analyzer  = ScriptAnalyzerAgent(self.llm)
        self.video_planner    = VideoPlanner(self.llm)
        self.quality_reviewer = QualityReviewAgent(self.llm)

    def generate_video(self, plan: VideoPlan, output_path: str | None = None) -> str:
        audio = self.assets.audio_path()
        self.tts.generate_audio(
            " ".join(s.narration for s in plan.scenes),
            audio,
            voice=self.tts.voice_for(plan.video_type.value),
        )
        return self.video_pipeline.generate(plan, audio, self.assets)
```

Backend switching is a single env var. No agent code changes. No orchestrator changes.

---

## 7. New File Structure

```
services/
  scene_renderer.py              ← SceneRenderer ABC + SlideshowSceneRenderer
  hunyuan_client.py              ← HunyuanClient (RunPod Serverless)
  hunyuan_scene_renderer.py      ← HunyuanSceneRenderer
  video_assembler.py             ← VideoAssembler (extracted from VideoAssemblyService)
  video_generation_pipeline.py   ← VideoGenerationPipeline
  video_generation_factory.py    ← VideoBackend enum + registry + factory

  # Existing — untouched
  video_assembly.py              ← VideoAssemblyService (preserved, still works standalone)
  image_service.py               ← ImageService (satisfies IImageGenerator protocol)
  tts_service.py
  llm_service.py
```

`VideoAssemblyService` is NOT deleted. `SlideshowSceneRenderer` reuses its image-prep helpers (`_resize_crop`, `_add_overlay`) by importing them as module-level functions — no subclassing, no duplication.

---

## 8. SOLID Assessment (Post-Fix)

| Principle | Status | Evidence |
|---|---|---|
| **S** Single Responsibility | ✓ | Each class has one reason to change |
| **O** Open/Closed | ✓ | Registry: new backend = new file + one dict entry |
| **L** Liskov Substitution | ✓ | Both renderers return `VideoClip`; assembler works uniformly |
| **I** Interface Segregation | ✓ | `SceneRenderer` has one method; `IImageGenerator` has one method |
| **D** Dependency Inversion | ✓ | `SlideshowSceneRenderer` depends on `IImageGenerator` protocol |

---

## 9. Design Patterns Applied

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `SceneRenderer` + `VideoGenerationPipeline` | Swap rendering algorithm at runtime without changing context |
| **Factory Method** | `VideoGenerationFactory.slideshow()`, `.hunyuan()` | Typed construction — no accidental wrong-backend wiring |
| **Registry** | `_RENDERER_REGISTRY` | OCP-compliant extension point for new backends |
| **Facade** | `VideoGenerationPipeline` | Single entry point hiding render-per-scene + assembly complexity |
| **Adapter** | `SlideshowSceneRenderer` | Adapts existing `ImageService` API to `SceneRenderer` contract |
| **Protocol (Structural Typing)** | `IImageGenerator` | DIP without inheritance overhead; `ImageService` needs no modification |

---

## 10. Senior Developer Evaluation

### What is genuinely strong

**The abstraction boundary is correct.** `SceneRenderer` sits at exactly the right seam — rendering is the only thing that differs between backends. Everything upstream (agents, planner, reviewer) and downstream (assembly, TTS) is backend-agnostic. This is the hardest part of abstraction design to get right, and it is right here.

**The `VideoClip` return type is a non-obvious but correct choice.** Returning file paths looks simpler on the surface, but forces the assembler to branch on file type. Returning `VideoClip` objects keeps the assembler dumb and the LSP contract tight. The tradeoff is a coupling to MoviePy's type system — acceptable given MoviePy is the chosen assembly tool.

**DI is genuinely useful here, not ceremonial.** The injected `IImageGenerator` means `SlideshowSceneRenderer` can be unit-tested with a stub that returns a pre-made image path. No OpenAI API calls in tests. Same for `HunyuanClient`. This is the practical payoff of DIP.

**The factory split (`slideshow()` vs `hunyuan()`) prevents misconfiguration.** A single `create(backend, **all_deps)` method is a footgun — you can pass the wrong deps and get a runtime error. Named static methods make wrong usage a type error at the call site.

---

### Trade-offs accepted

**`VideoClip` return type couples the interface to MoviePy.** If MoviePy were ever replaced with an FFmpeg-native solution, `SceneRenderer`'s return type would need to change, and all implementations would be affected. The mitigation would be to define a `ClipDescriptor` dataclass (path + duration + metadata) and have the assembler interpret it. Not done here because it adds indirection with no current benefit — YAGNI applies.

**`VideoAssembler` is concrete.** It could be an ABC with a `MoviePyAssembler` implementation. Not done because there is currently one assembler and no plan to swap it. If an FFmpeg assembler is ever needed, promoting it to ABC is a one-line change.

**`from_backend()` receives unused dependencies.** When `backend=SLIDESHOW`, the `hunyuan_client` argument is constructed but never used. This is the pragmatic cost of env-var-driven backend selection at a composition root. The alternative — lazy construction — adds complexity with no observable benefit at this scale. Accepted.

---

### Risks

**`HunyuanClient._poll()` can block for minutes.** HunyuanVideo generation on RunPod takes 2–10 minutes per clip. A 5-scene video can block `generate_video()` for 50 minutes synchronously. The pipeline must either run scene generation concurrently (via `asyncio` or `ThreadPoolExecutor`) or accept this as a known limitation for now.

**RunPod cold starts.** Serverless endpoints spin up from zero; first-request latency can be 2–5 minutes on top of inference time. The `_poll()` timeout must account for this. Current design sets 600s — borderline for multi-scene generation.

**Memory: `VideoClip` objects held open across scene loop.** A 5-scene HunyuanVideo run holds 5 `VideoFileClip` handles open simultaneously. On a constrained machine this could be an issue. The `finally: clip.close()` in `generate()` ensures cleanup after assembly, but peak memory is proportional to scene count. Acceptable for typical short-form video (3–7 scenes).

---

### What is missing

**Error handling strategy is undefined.** What happens if scene 3 of 5 fails to render? Currently the exception propagates and partially generated assets are left on disk. A `SceneRenderError` with a retry policy (e.g., 2 retries, then fallback to slideshow renderer) would make the pipeline production-grade.

**No concurrent scene rendering.** Scenes are rendered sequentially. For HunyuanVideo, this is the dominant cost. A `ThreadPoolExecutor` wrapper in `VideoGenerationPipeline.generate()` would cut wall-clock time by `N_scenes` without any interface changes — the design supports this cleanly because `SceneRenderer.render()` is stateless.

**`HUNYUAN_ENDPOINT_ID` env var with no validation.** If missing, `HunyuanClient` constructs silently with an empty endpoint ID and fails only at the first `generate_clip()` call. Should validate at construction time and raise a clear `ConfigurationError`.

---

### Verdict

This is a well-proportioned design for the current problem scope. The abstractions are motivated by real variance (slideshow vs. video), not speculative extensibility. The SOLID fixes are load-bearing, not cosmetic. The two concrete risks (sequential rendering, error handling) are known and deferrable — they do not invalidate the design, they are the next iteration.

**A senior engineer would approve this design for implementation with the two risks documented as follow-up tickets.**

---

## 11. Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `VIDEO_BACKEND` | No | `slideshow` | Which renderer to use (`slideshow` or `hunyuan`) |
| `HUNYUAN_ENDPOINT_ID` | If backend=hunyuan | — | RunPod Serverless endpoint ID for HunyuanVideo |
| `RUNPOD_API_KEY` | If backend=hunyuan | — | Already in `.env` |

---

## 12. Future Extension Example

Adding **Wan2.1** as a third backend requires:

1. `services/wan_client.py` — `WanClient` wrapping its API
2. `services/wan_scene_renderer.py` — `WanSceneRenderer(SceneRenderer)`
3. One entry in `_RENDERER_REGISTRY`:
   ```python
   VideoBackend.WAN: lambda _img, _hunyuan, wan: WanSceneRenderer(wan),
   ```
4. `VideoBackend.WAN = "wan"` in the enum
5. `VideoGenerationFactory.wan(client: WanClient)` static method

Zero changes to `VideoGenerationPipeline`, `VideoAssembler`, `PipelineManager`, or any agent.
