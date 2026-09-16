# Foreman

**Say what "good" looks like. It watches, judges, and acts — entirely on the edge.**

Foreman is a visual inspection and compliance station built on the **SiMa.ai
Modalix MLSoC DevKit 3.0**. You state the quality standard out loud, in plain
language. A detector watches the line continuously on the Modalix MLA. When an
item settles in frame, a vision-language model on the same MLA judges it against
your spoken standard and says *why*. The verdict is logged with the evidence
frame that produced it.

No training data. No retraining when the standard changes. No video leaves the room.

---

## The problem

Machine vision inspection works well for defects you can afford to train for.
Most real quality and compliance rules are not like that: they are long-tail,
they are written in words, and they change. *"The safety seal must be intact and
the batch label must face the camera."* *"No pallet may be stacked above the
yellow line."* *"Every carton needs a lid and a legible date."*

Training a detector per rule is infeasible — you need thousands of labelled
examples per defect, and the rule changes next week. So these checks stay manual,
they are inconsistent between operators, and the audit trail is a clipboard.

Meanwhile the footage that would make this automatable is exactly the footage
that legally or commercially cannot be shipped to a cloud vision API: factory
floors, pharmacy benches, retail back-rooms, customer goods.

## The solution

Put a model that already understands language *and* images on the line itself.

1. **SEE** — a YOLO26 detector runs continuously on the Modalix MLA over the live
   camera feed. Cheap, real-time, every frame.
2. **UNDERSTAND** — when the gate decides an item is present and has stopped
   moving, a vision-language model on the same MLA reads the frame against your
   spoken standard and returns a verdict *and a written reason*.
3. **ACT** — the Mac applies pass/fail policy, writes an auditable record with
   the evidence frame, updates the andon console, and re-arms.

Changing the standard is a sentence, not a sprint.

## Why it matters

- **No training data.** The standard is a prompt, not a dataset.
- **Reconfigurable by the person who owns the rule**, not by an ML team.
- **The reason is part of the output.** An operator can disagree with a sentence;
  they cannot disagree with a confidence score.
- **The video never leaves the building.** This is the reason to own the silicon
  rather than rent an API.
- **It keeps working when the network does not.** Nothing in the inspection loop
  touches the internet.

---

## Architecture

```
   microphone ─────────────┐                    ┌──── MODALIX MLSoC DevKit 3.0 ────┐
   (Mac built-in)          │  audio over HTTP   │                                   │
                           └───────────────────>│  [MODALIX] Whisper small a16w8    │
                                                │             ASR on the MLA        │
   camera ──> [MAC] ffmpeg ──> RTSP ───────────>│                                   │
   (Mac or USB)              (Insight :8554)    │  [MODALIX] YOLO26 detector        │
                                                │             MLA, continuous       │
                                                │                │         │        │
                                                │      detections│         │H.264   │
                                                │                │         │        │
                                                │  [MODALIX] Qwen3-VL 2B a16w4      │
                                                │             MLA, event-triggered  │
                                                └────┬───────────────┬──────────────┘
                                 detections + verdicts│          video│ + boxes
                                          (HTTP/SSE)  │          (UDP)│
                                                      v               v
                          ┌──────────────────────────────┐   ┌──────────────────┐
                          │ [MAC] Foreman orchestrator   │   │ [MODALIX/BROWSER]│
                          │   presence + stability gate  │   │  Neat Insight    │
                          │   session state, policy      │   │  live video with │
                          │   audit trail + evidence     │   │  box overlays    │
                          └──────────────┬───────────────┘   └──────────────────┘
                                         v
                          ┌──────────────────────────────┐
                          │ [BROWSER] Foreman console     │
                          │   verdict, reason, evidence,  │
                          │   measured latency            │
                          └──────────────────────────────┘
```

### What runs on Modalix

| Workload | Model | Why it belongs on the MLA |
|---|---|---|
| **Object detection** | `yolo_26n` (Model Zoo gen2, INT8) | Continuous CNN inference on every frame. Neat also does anchor decode and NMS in `Model::Options`. Measured 6.3 ms, ~160 fps standalone. |
| **Visual judgement** | Qwen3-VL-2B-Instruct-GPTQ-a16w4 | Transformer inference with a vision encoder. The heaviest workload in the system. Measured 1368 ms median per judgement. |
| **Speech recognition** | `simaai/whisper-small-a16w8` | SiMa ships Whisper with encoder, decoder (init/pre/cache/post) and language detection compiled as MLA ELFs. Measured RTF ~0.105. |
| **H.264 encode + RTP** | Neat `VideoSender` | Must share the detector's `Run` so RTP and metadata timestamps correlate inside Insight's ±1 ms window. |

**Every inference in this product runs on Modalix.** The Mac runs no model.

### What runs on the Mac

| Workload | Why it belongs here |
|---|---|
| **Presence + stability gate** | State logic over a metadata stream. Deciding *when* to spend a VLM inference is control, not perception. |
| **Session state and pass/fail policy** | Business logic. No accelerator benefit. |
| **Audit trail + evidence storage** | Durable storage belongs on the machine with the disk. |
| **Camera capture and RTSP publish** | The camera is attached to the Mac. |
| **Product UI** | Presentation. |

The split is deliberate, not decorative: the Mac never does inference, and the
DevKit never does bookkeeping.

### Why this split

A vision-language model takes seconds per image. It can never be the continuous
perception stage. A detector takes milliseconds and can. So the cheap model runs
always and the expensive model runs rarely, and something has to decide "rarely
— now". That decision is the gate, and it is the Mac's job.

This is also why the system stays responsive: the live view never waits for the
VLM.

---

## Data flow

1. The Mac publishes its camera to Insight's RTSP server (`rtsp://<mac>:8554/src2`).
2. The DevKit's edge agent consumes that stream and runs the detector on the MLA.
3. Every frame: boxes go to Insight over UDP 9100 for the overlay, H.264 goes to
   Insight over UDP 9000, and a compact detection event goes to the Mac over SSE.
4. The Mac's gate watches those events. When a qualifying detection holds still
   for N consecutive frames, it fires — once.
5. The Mac calls `POST /inspect` on the DevKit with the current standard.
6. The DevKit JPEG-encodes the current frame and asks the vision-language model
   on the MLA to judge it. It returns a verdict, a reason, the evidence frame,
   and its own measured inference time.
7. The Mac records the result, updates counters, writes `audit/inspections.jsonl`
   and the evidence JPEG, and pushes new state to the browser.
8. The gate re-arms only after the item leaves the frame.

Speech follows the same path: the browser records audio, the Mac forwards it to
the DevKit, Whisper transcribes it on the MLA, and the text becomes the standard.

---

## SiMa.ai technologies used

- **Palette Neat SDK 2.1.3** — cross-compilation, runtime, examples
- **Neat `Graph` / `Model` / `Run`** — the detection pipeline, with
  `BoxDecodeType::YoloV26` and `NormalizePreset::COCO_YOLO` doing preprocessing
  and NMS on-device
- **`VideoSender` + `MetadataSender`** — live video and box overlays into Insight
- **Neat GenAI (`pyneat.genai`)** — `GenAIServer` hosting the VLM and Whisper
  behind an OpenAI-compatible API on the board
- **LLiMa** — `llima pull` for the pre-compiled a16w4 / a16w8 model directories
- **Neat Insight** — RTSP media server, live viewer, box overlays, MLA metrics;
  kept as the engineering view alongside the product console
- **MLA acceleration** — every model in the system

---

## Setup

### Prerequisites

- Modalix MLSoC DevKit 3.0, powered, with **serial and Ethernet** connected
  (the Ethernet adapter must plug directly into a USB-C port, not through a hub)
- Palette Neat SDK container running (`sima-cli sdk start`)
- macOS host with `uv`, `ffmpeg`, and Docker
- A SiMa developer-portal login for the Model Zoo (`sima-cli login`)

### 1. Pair the DevKit

```bash
# [HOST]
sima-cli sdk setup --devkit <DEVKIT_IP>      # installs `dk`, exports /workspace over NFS
```

### 2. Get the models onto the board

```bash
# [HOST] Whisper is public on HuggingFace and needs no login:
docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'source /sdk-extensions/model-compiler/bin/activate && \
   python -c "from huggingface_hub import snapshot_download as d; \
              d(\"simaai/whisper-small-a16w8\", local_dir=\"/workspace/models/whisper-small-a16w8\")"'

# [HOST] The detector comes from the Model Zoo, which does need a login:
sima-cli login
docker exec ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'sima-cli modelzoo -v 2.1.3 --boardtype modalix get <detector-model>'

# [HOST] Copy/pull everything onto the DevKit:
DEVKIT_IP=<ip> ./scripts/setup-devkit.sh
```

### 3. Run it

```bash
# [HOST]
DEVKIT_IP=<ip> FOREMAN_MODEL=/workspace/models/<detector>.tar.gz ./scripts/demo.sh
```

`demo.sh` runs preflight checks, starts the camera, brings up both DevKit
processes and the console, and opens the browser. Ctrl-C stops everything.

Full operating instructions, including recovery: **[docs/DEMO.md](docs/DEMO.md)**.

---

## Build and test

```bash
uv sync
uv run pytest          # 67 tests, no DevKit required
uv run ruff check .
../scripts/smoke-test.sh   # environment check; the DevKit stage degrades to SKIP
```

The Mac-side orchestrator, the gate, and the edge agent's parsing logic are all
tested against a fake edge (`tests/fake_edge.py`) that speaks the real wire
protocol over a real socket.

> **The fake edge is a test harness and is never used in a demo.** Everything it
> returns is synthetic and it is labelled as such in its own source.
> `scripts/demo.sh` does not start it. `scripts/dev-ui.sh` does, and is for UI
> development only.

---

## Measured performance

See **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

Headline measured figures on a Modalix DevKit 3.0:

| | Measured |
|---|---|
| Detector, standalone | **6.3 ms median, ~160 fps** |
| Detector, live pipeline | 15.1 fps idle / 15.2 fps under VLM load (camera-bound) |
| Vision-language judgement | **1368 ms median** (n=31) |
| Whisper ASR | **262 ms for 2.50 s of audio, RTF ~0.105** |
| End-to-end, gate fire to verdict | **1875 ms** (of which ~12 ms is the Mac) |
| Verdict accuracy, 6 standards x 4 reps | **24/24**, 0 self-inconsistencies |
| Insight metadata-to-video correlation | **151/151, 0 expired** |
| Pipelined detector throughput | **778.9 inferences/s** at 5.21 ms |
| Board temperature, full stack running | 49-50 C (SoC), 55-57 C (board) |

**Power is not claimed.** This DevKit exposes no power rail, so the benchmark
reports 0.00 W; SiMa's "under 10 W" is their figure for the chip, not ours.

Model inference latency and end-to-end system latency are reported separately.
Nothing is recorded there that has not been measured on this hardware. The
vendor's 50 TOPS / <10 W figures are marketing claims about the chip and are not
presented as measurements of this application.

---

## Limitations

- **A vision-language model is not a metrology instrument.** Foreman judges
  things a person could judge from a photograph. It cannot measure a tolerance,
  and it will not catch a defect that is invisible at camera resolution.
- **Verdicts are not deterministic.** The same frame yields different wording
  across runs, and one observed run returned `pass` with a reason that said the
  item was *not* compliant. The model is now asked for both a boolean and a
  verdict in one call, and any disagreement is reported as `unclear` rather than
  a confident answer. Measured: 0 disagreements in 24 judgements - but the guard
  exists because the failure was real. For anything safety-critical this is a
  screening aid, not an authority.
- **One item at a time.** The gate fires on the most confident qualifying
  detection. Several items in frame at once are judged as one scene.
- **The detector and the VLM share one MLA.** Measured, the detector holds 15 fps
  with judgements running - but it is camera-bound at 15 fps and has ~10x headroom,
  so that result does not prove there is no contention at saturation.
- **Whisper cannot be fine-tuned here.** `whisper-small-a16w8` is consume-as-is —
  it is not `llima-compile`-able — so domain jargon may transcribe poorly. The
  console always shows the transcribed standard so it can be corrected by typing.
- **No re-inspection queue.** An `unclear` verdict is recorded, not retried.
- **The demo camera is the Mac's**, published over RTSP. A production
  installation would use a MIPI or USB camera on the board.

## Future work

- **Close the physical loop.** The ACT stage currently logs and displays. A GPIO
  or PLC signal off the DevKit would make it actuate a real reject gate.
- **Re-inspect on `unclear`**, from a second angle, before escalating to a human.
- **Track IDs from the tracker example** so multiple items in frame are judged
  individually rather than as one scene.
- **Cache the image embedding.** `VisionLanguageModel.encode()` allows repeated
  questions about one image without re-encoding — worth it for multi-rule standards.
- **On-device standards library** with versioning, so an audit says which revision
  of which rule produced a verdict.
- **Quantify agreement** against a human inspector on a fixed set of items, and
  publish that number rather than asserting accuracy.
