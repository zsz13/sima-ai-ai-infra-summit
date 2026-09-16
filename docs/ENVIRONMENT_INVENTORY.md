# Environment Inventory

SiMa.ai Palette Neat SDK **2.1.3_Palette_SDK_neat_v2.1.3.0** on an Apple Silicon
MacBook Pro. Everything below was verified in this environment. Vendor claims are
labelled as such; nothing here is a performance measurement of our application.

---

## 1. Hosts and roles

| | Host | SDK container | Modalix DevKit |
|---|---|---|---|
| **What** | macOS, Apple **M5 Max** | `ghcr.io-sima-neat-sdk-v2.1.3.0` | Modalix MLSoC DevKit 3.0 |
| **Arch** | arm64 | **aarch64** Ubuntu 24.04 | aarch64 |
| **CPU/GPU** | 18 cores (6 Super + 12 Perf), 32-core GPU, Metal 4 | shares host | Cortex-A65 + **MLA** + EV74 DSP |
| **Memory** | 36 GB unified | shares host | 16 GB LPDDR5 |
| **Storage** | 1.7 TB free | — | 32 GB eMMC + 500 GB NVMe |
| **Role** | orchestration, UI, serial console | **authoring + cross-compile only** | **the only place ML executes** |
| **Status** | healthy | healthy | **unreachable** |

**The container cannot execute any model.** No Neat debs in its rootfs, no
`pyneat`, no MLA device node. It cross-compiles and serves Insight; that is all.

### Host ML tooling
Effectively bare: Python 3.13, Node 22, `uv`, `git`, `gh`, `picocom`, `mkcert`.
**No MLX, whisper.cpp, ollama, torch, ffmpeg, or cmake.** Any Mac-side ML work
starts from zero — a real cost to weigh against using the DevKit.

---

## 2. Neat runtime and toolchain

| Component | Version |
|---|---|
| Neat core / neat-runtime / neat-gst-plugins / neat-llima | 0.4.0 |
| PyNeat | 0.4.0 (**aarch64 + cp311 wheel only**) |
| neat-insight | 0.0.7 (update available — **do not update mid-hackathon**) |
| Model Compiler | 2.1.3.dev0+master.392 |

**Cross-compilation is ready now**, pre-exported in the container:
`CC/CXX = aarch64-linux-gnu-{gcc,g++}-12`, `SYSROOT=/opt/toolchain/aarch64/modalix`.
`apps-src/build.sh` auto-selects `cmake/toolchains/aarch64-modalix.cmake`.

```bash
cd /neat-resources/apps-src && ./build.sh --release
#   -> build/examples/<category>/<example>/<example>
cd /neat-resources/core-src && ./tutorials/build.sh --target tutorial_018_consume_rtsp_stream
#   -> build/tutorials-standalone/tutorial_018_consume_rtsp_stream
```

- Sysroot headers: `neat.h`, `neat/{runtime,models,nodes,node_groups,genai}.h`,
  `neat/profiler/`, `sima_lmm/`, `simaai/`, PCIe APIs.
- **GStreamer exists only in the aarch64 sysroot** (141 libs + SiMa plugins
  `libgstneat{decoder,encoder,dispatcher,cast,dequant,argmax,detess,processcvu,pciesink,pciesrc}.so`).
  Container PATH has **no** `gst-launch-1.0`; it does have `ffmpeg`/`ffprobe`.
- **Python does not avoid the DevKit.** `import pyneat` fails in the container
  (wheel is cp311, container is 3.12). Python removes cross-compilation, not hardware.

### Python interpreters
| Interpreter | Notable packages |
|---|---|
| `/opt/neat-insight/venv/bin/python3` — what `python3` resolves to | Flask 3.1.3, pillow, psutil |
| `/usr/bin/python3` | nothing (no pip) |
| `/sdk-extensions/model-compiler/bin/python` — **the ML one** | torch 2.8.0, transformers 5.5.4, onnx 1.17.0, onnxruntime 1.21.1, tensorflow 2.17.0, numpy 1.26.4, opencv 4.7/4.11, huggingface_hub 1.28.0 |

No `fastapi`/`uvicorn` anywhere. Serving = Flask, or the Neat GenAI server on-device.

---

## 3. DevKit execution path (`dk`)

`dk` and `devkit-run` are **bash functions, not binaries** — defined in
`/usr/local/bin/devkit.sh` (`devkit-run` L1021, `dk` dispatcher L1396), persisted
to `~/.devkit-sync.rc` and sourced from `~/.bashrc` **only after pairing**.
`which devkit-run` returning nothing is normal.

**Current state: unpaired.** `~/.devkit-sync.rc` absent, `~/.ssh/` empty, no
`DEVKIT_SYNC_*`, `NFS_SERVER_HOST_IP`/`DEVKIT_HOST_EXPORT_PATH` unset, host
`/etc/exports` absent.

```bash
# [HOST] the correct entry point — also creates the NFS export and injects container env
sima-cli sdk setup --devkit <IP>
# [SDK] afterwards, in a NEW interactive shell
dk status ; dk <aarch64-binary|script.py> ; dk shell ; dk sync
```

`source devkit.sh <ip>` **from inside the container will fail** (aborts at L520 on
the missing host-export env). Host-side setup must run first.

**`dk` guard rails:** refuses non-aarch64 binaries; the target **must** live under
`/workspace`; `.py` runs under the DevKit PyNeat venv; with the rsync fallback only
the target's top-level workspace folder auto-syncs.

---

## 4. Neat Insight — verified working without a DevKit

Supervised (`insight-admin status` → RUNNING). Three processes: `neat-insight`
(Flask, 9900), `vf` (WebRTC, 8081), `mediamtx` (RTSP, 8554).

| Port | Purpose |
|---|---|
| 9900/tcp | Insight UI + REST API (**HTTPS, mkcert — `curl -k`**) |
| 8081/tcp | Video Viewer (`/static/viewer.html`) — **overlays render here, not on 9900** |
| 8554/tcp | RTSP media-source server (`/src1..N`) |
| 9000–9003/udp | video RTP ingest, channel N → `9000+N` |
| 9100–9103/udp | metadata ingest, channel N → `9100+N` |
| 40000–40007/udp | WebRTC transport |
| 9999 / 10000 tcp | VS Code Web (http / https) |
| 8022/tcp | browser shell to a paired DevKit |

Application side (headers are the contract):
`nodes/groups/VideoSender.h` → `VideoSenderOptions::H264RtpUdpFromRaw(w,h,fps)`
or `::Passthrough(codec)`; `nodes/io/MetadataSender.h` → `MetadataSenderOptions`.
**Keep channel numbers aligned**: video `9000+N`, metadata `9100+N`. Metadata
types: `object-detection`, `classification`, `pose-estimation`, `segmentation`,
`tracking`.

Useful endpoints: `/api/health`, `/api/metrics` (CPU, mem, temp, **MLA**),
`/api/ingest/stats?all=1`, `/api/egress/stats`, `/api/viewer-url`,
`/api/mediasrc/{assign,start,stop}`, `/api/upload/media`.

### Verified end-to-end today (no DevKit)
`./scripts/smoke-test.sh` → **5 pass / 0 fail / 1 skip**. `video03.mp4` uploaded,
auto-optimized to H.264 baseline GOP 30, serving on `rtsp://127.0.0.1:8554/src1`;
`ffprobe` confirms **1280×720 H.264 @ 16 fps**, 30 frames decoded, exit 0.
Synthetic metadata via `neat-insight-metadata-test` → **166 messages, 0 invalid
JSON, ~28 msg/s**.

**The whole video + overlay path is proven. Only inference needs hardware.**

---

## 5. Models

### 5a. Classic CV — `.tar.gz` MPK archives, **and none are on disk**

19 stable examples, all dual-language C++/Python except the PCIe one. All use
**pre-compiled MPK archives**; no classic example compiles ONNX.

| Example | Model | Input | Output |
|---|---|---|---|
| `object-detection/single-stream-object-detector` | `yolo26m-det-bf16-mla_tess-b1` | 1 RTSP h264/h265/mjpeg or HTTP MJPEG | Insight video + detection metadata |
| `object-detection/multi-stream-object-detector` | `yolo26m-det-int8-b1` | N RTSP | Insight video + metadata |
| `object-detection/high-density-multi-stream-object-detector` | `yolo26n-det-int8-b1` | **16/24/48 RTSP** (3 profiles) | per-channel video + metadata |
| `object-detection/{yolo26,detr,ssd-mobilenet}-object-detector` | yolo26m / DETR-R50 / SSD-MobileNetV2 | image folder | annotated images |
| `segmentation/single-stream-instance-segmenter` | `yolo26m-seg-bf16-b1` | 1 RTSP | segmentation polygons (32 KB/frame budget) |
| `segmentation/yolov8-instance-segmenter` | `yolo_v8n_seg` | image folder | masks |
| `pose-estimation/multi-stream-pose-estimator` | `yolo26m-pose-int8-b1` | N RTSP | 17-pt COCO pose |
| `tracking/multi-stream-people-tracker` | `yolo26m-det-int8-b1` | N RTSP | stable per-person track IDs |
| `tracking/yolo26-tiny-drone-tracker` | `yolo26n_p2_tiny_drone_int8_qat_b1` | ≤4 RTSP | per-stream tracking |
| `face-detection/face-detector` | `retinaface_mobilenet25` | image folder | boxes + 5 landmarks |
| `face-detection/single-stream-thermal-face-detector` | `yolov5s_face_raw_split` | thermal RTSP | raw heads decoded in app code |
| `classification/image-classifier` | `resnet_50` | 1 image | top-5 |
| `depth-estimation/depth-estimator` | `depth_anything_v2_vits` | image folder | depth maps |
| `feature-extraction/superpoint-feature-extractor` | `superpoint_modalix_int8_tessellation_mla` | bundled video | Insight overlay |
| `benchmarking/model-benchmark` | any `.tar.gz` | synthetic tensors | latency, throughput, **power, energy** |
| `benchmarking/mipi-camera-capture` | none | MIPI camera, zero-copy DMA-BUF | NV12 frames + `summary.json` |

Suffixes encode the compile target: `-mla_tess` = MLA tessellation, `-ev74` = DSP,
plus `int8`/`bf16` and `-b1` batch. **No example uses a USB/V4L2 camera.** Local
display exists nowhere — streaming examples all target Insight.

Bundled datasets (~15 MB, no download needed): `assets/datasets/coco` (21 images),
`tum-rgbd/freiburg1-desk.mp4`, drone-tracker clips.

### 5b. GenAI — LLiMa model directories (**not** MPK archives)

`llima pull <name>` on the board → `/media/nvme/llima/models/<name>`.
Two families that must not be confused:

| Family | Suffix | Runnable? |
|---|---|---|
| Pre-quantized checkpoints (~38 repos) | `-Safetensors` | **No** — input to `llima-compile` |
| Precompiled runtime models | `-a16w4` / `-a16w8` | **Yes** — `llima pull` and run |

- **LLM**: Qwen2.5 (0.5–7B), Qwen3 (0.6–8B), Llama-3.1/3.2, Mistral-7B-v0.3,
  Phi-3.5/4-mini, Gemma-2/3/4, LFM2/LFM2.5 (230M–2.6B).
- **VLM**: Qwen2.5-VL 3B/7B, Qwen3-VL 2B/4B/8B, LFM2-VL, Gemma-4-E2B/E4B, llava-1.5-7b.
- **ASR**: `whisper-small-a16w8`, `whisper-medium-a16w8`.

`llima-compile` supports LLM `llama, lfm, gemma, phi, qwen, mistral`; VLM
`gemma3, gemma4, lfm2_vl, llava, paligemma, qwen2_5_vl, qwen3_vl`. Single backend:
`modalix`.

**Naming trap:** tutorials say `...-GPTQ-a16w4`, live repos are often
`...-Autoround-a16w4`. Verify before scripting.

### 5c. Whisper / ASR on Modalix — **YES, MLA-accelerated**

This was the open question; it is settled.

`/opt/toolchain/aarch64/modalix/usr/include/sima_lmm/whisper_model.hpp` declares
every stage as an MLA model: `_encoder_model_ptr`,
`_decoder_language_detect_model_ptr`, `_decoder_init_model_map`,
`_decoder_pre_model_map` — all `MLAModelWithBuffer`, with per-part ELF loaders.
Runtime lib `libdevkit_whispermodel.so`. Model completeness requires non-empty
`elf_files/*.elf` — compiled MLA machine code.

**Encoder, full decoder (init/pre/cache/post) and auto language-detect run on the
MLA.** Only ffmpeg audio decode, the log-mel spectrogram (16 kHz, n_fft 400,
hop 160, 3000 frames) and tokenization run on the A65 CPU.

`simaai/whisper-small-a16w8` = BF16 activations / INT8 weights. **48 files,
1.13 GB** (verified against the public HF API): encoder ELF 447 MB, language-detect
ELF 211 MB, token embeddings 80 MB, per-layer decoder ELFs.

```bash
llima pull whisper-small-a16w8
python3 tutorials/021_serve_genai_models/serve_genai_models.py \
        --asr /media/nvme/llima/models/whisper-small-a16w8
curl -F model=asr -F language=auto -F file=@speech.wav \
     http://<modalix-ip>:9998/v1/audio/transcriptions
```

Or in-process: `pyneat.genai.ASRModel(dir).run(request)`.

**Caveat:** Whisper is **not** `llima-compile`-able (deliberately outside the
architecture enums) and has no `-Safetensors` repo — consume-as-is. No
custom-vocabulary path in a hackathon timeframe.

---

## 6. GenAI API surface (`pyneat.genai`, device-side)

`GenAIModel` (auto-detects LLM/VLM/ASR) · `VisionLanguageModel` · `ASRModel` ·
`GenAIServer` · `graphs.speech_transcriber()` / `graphs.vision_language()`

`GenerationRequest`: `prompt`, `system_prompt`, `messages`, `images`,
`use_cached_images`, `audio`, `audio_file`, `language` (default `"auto"`),
`asr_task`, `max_new_tokens`, `enable_thinking`, `tools`, `tool_choice`.
`GenerationResult` / `TokenSample`: `text`, `reasoning`, `metrics`,
`finish_reason`, `language`, `no_speech_prob`, `avg_logprob`, `tool_calls`.

**API decision logic:** compiled `.tar.gz` → classic `Model`/`Graph`; LLiMa model
*directory* → GenAI. Caller supplies tensors → `Model`; pipeline owns the source →
`Graph`. In-process GenAI → `GenAIModel`, narrowing to `ASRModel`/`VisionLanguageModel`.
`GenAIServer` only when the boundary is genuinely HTTP.

---

## 7. Programming model (classic CV)

Namespace `simaai::neat`. Declare a dataflow `Graph`, `build()` it into a `Run`,
pull `Sample`s.

- `Model` is constructed straight from an MPK path. **`Model::Options` owns pre-
  *and* post-processing**: `NormalizePreset::COCO_YOLO`, and
  `decode_type = BoxDecodeType::YoloV26 | ::Ssd` makes Neat do anchors, scoring
  and NMS. Adding a model family is usually picking a decode type, not writing
  postprocessing.
- Node groups: `RtspDecodedInput`, `HttpMjpegDecodedInput`, `CameraInput`, `VideoSender`.
- Topology: `graphs::Branch(...)` fan-out, `graphs::Combine(..., CombinePolicy::ByFrame)`
  frame-synchronised fan-in, `graph.connect(a, b)`.
- `RunOptions`: `RunPreset::Realtime`, `queue_depth`, `OverflowPolicy::KeepLatest`,
  `OutputMemory::ZeroCopy`.
- `MetadataSender` sits **outside** the graph (plain UDP to Insight).
- **Encoder and detector must share one `Run`** so RTP and metadata timestamps
  line up for Insight overlay correlation.
- `graph.describe_backend()` prints the generated backend pipeline — the debugging hook.

Layout is identical in every example: `src/cpp/`, `src/python/`, `src/common/config.yaml`,
`tests/`. Shared helpers in `apps-src/support/`.
`scripts/create_example_scaffold.sh` generates a complete new example in house style.

---

## 8. Tutorials (`core-src/tutorials/`, 001–023, all C++ **and** Python)

001 first model · 002 async · **003 benchmark** · 004 first graph · 005 model options ·
006 preprocess · 007 detection boxes · 008 model in pipeline · 009 NumPy · 010 multi-input ·
011 interpret output · 012 profile · 013 custom data graph · 014 model in graph ·
**015 multi-stream** · 016 throughput/queues · **017 production pipeline** ·
**018 RTSP** · 019 LLM · 020 VLM · **021 serve GenAI (the ASR reference)** ·
022 GenAI in graph · **023 MIPI camera**

- **018** pairs directly with Insight's `rtsp://127.0.0.1:8554/srcN`.
- **021** serves LLM + VLM + **ASR** behind OpenAI-compatible HTTP `:9998`,
  including `/v1/audio/transcriptions`; ships 3 working Python clients and a
  sample WAV (`sima_lmm/assets/why_is_the_sky_blue.wav`).
- **022** documents the `SpeechTranscriber` fragment but its sample code only
  builds the VLM fragment — **the ASR graph path is untested by example**.
- There is **no dedicated ASR tutorial**. Use 021 + GenAI Studio.

---

## 9. GenAI example apps

| App | Lang | Models | Ready? |
|---|---|---|---|
| `genai/neat-genai-studio` | Python | `whisper-small-a16w8` (pinned) + `gte-small` (RAG) + any LLM/VLM on demand | **Closest** — `./setup.sh` then `./run.sh`; no chat model by default |
| `genai/detection-to-vlm-assistant` | Python | `yolo26m-det-bf16-mla_tess-b1` + `Qwen3-VL-4B` | **No** — 3 unfilled config placeholders, 2 model systems, needs RTSP + Insight |

GenAI Studio: two venvs (`~/pyneat` for the board server, `./.venv` for Flask UI),
OpenAI server `:9998`, control API `:9997`, HTTPS UI `:5000`. Models discovered by
scanning for `devkit/vlm_config.json` or `devkit/whisper_config.json`. TTS (piper)
is **CPU/onnxruntime, not MLA**.

---

## 10. Agent playbooks (inside the container, `/home/d/.claude/skills/`)

`neat-application-builder` (+6 references) · `sima-llima-compile-run` ·
`sima-model-quantize-compile` · `sima-model-surgery` ·
`sima-model-compiler-issue-triage` · `sima-use-neat-insight` (406 lines)

These are installed for a Claude/Codex running **inside** the container, not for
the host session. Readable from the host via `docker exec … cat`.

Full Neat Core source is on disk at `/neat-resources/core-src/` (own `CLAUDE.md`,
`include/`, `docs/`, `python/`) — **no need to guess at any API**.

---

## 11. Blocking dependencies and traps

1. **DevKit unreachable** — no USB-Ethernet adapter attached. See `HACKATHON_STATUS.md` B1.
2. **Model Zoo requires `sima-cli login`.** Every `docs.sima.ai/pkg_downloads/…`
   URL 302-redirects to Auth0, including the catalog JSON. `modelzoo list` drives
   an interactive picker and cannot be scripted. No credentials are cached.
   **No classic CV model can be obtained until someone logs in.** Only two
   loadables exist on disk, both in the sysroot: `usr/share/people.lm` (28.2 MB,
   undocumented) and `mla_init_davinci.lm` (init blob).
   *HuggingFace is unaffected — GenAI/Whisper models download anonymously.*
3. **`neat --json` reports `exposedPorts: []`** — it reads
   `$HOME/.insight-config/neat-port-map.json` which does not exist; the real file
   is under `/workspace/.ghcr.io-…/insight-config/`. Read the file directly or use
   `/api/viewer-url`. The Game Day Guide's advice to use `neat --json` does not work here.
4. **Only 4 Insight channels.** Channels ≥4 silently have no host port.
5. **One chat/VLM model resident at a time**; loading another evicts it. Whisper is
   pinned and cannot be unloaded. **Detector and VLM contend for the same MLA** —
   extra processes do not multiply throughput.
6. **Overlay correlation is ±1 ms.** Metadata must match video PTS within 90 RTP
   units. `messages_forwarded` measures DataChannel delivery, *not* correlation —
   use `matched_*` / `expired_*` / `evicted_*`, sampled twice and diffed.
   Test overlays on **8081**, not 9900. Hard-reload the viewer after any change
   (stale `drawing.js` persists).
7. **Model naming drift** `-GPTQ-` vs `-Autoround-` breaks copy-pasted scripts.
8. **`GenAIServer` has no auth, no TLS, open CORS.** Do not bind it to conference Wi-Fi.
9. **GenAI e2e tests are disabled for GA** — examples are demo-grade, not hardened.
10. **Pre-stage all model downloads.** VLMs are several GB; Whisper is 1.13 GB.

---

## 12. Assets staged so far

| Asset | Location | Note |
|---|---|---|
| `video03.mp4` | `/workspace/assets/videos/` + uploaded to Insight | 720p16 traffic clip from the official lab; playing on `src1` |
| `whisper-small-a16w8` | `/workspace/models/` | 1.13 GB, public HF, downloading — no login needed |
| `smoke-test.sh` | `/workspace/scripts/` | 6-stage check, DevKit stage degrades to SKIP |
