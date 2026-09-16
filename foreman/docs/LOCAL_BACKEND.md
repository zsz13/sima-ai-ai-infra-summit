# Local backend — Foreman without SiMa hardware

An **optional** way to run the whole of Foreman on this Mac: camera, detector,
Whisper and the vision-language model. It is not the demo path and it is not a
reference for Modalix behaviour.

---

## LOCAL QUICK START

**First-time setup** (once; ~2.2 GB of models, ~500 MB of wheels):

```sh
cd foreman
uv sync --group local
mkdir -p models-local && curl -L -o models-local/yolo26n.onnx \
  https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx
```

Whisper and the VLM download themselves on first run, into `~/.cache/huggingface`.

**Run:**

```sh
./scripts/dev-local.sh
```

**Open:** http://127.0.0.1:8800

**Stop:** `Ctrl-C`

After the first run, `./scripts/dev-local.sh` is the only command needed —
nothing re-downloads.

---

## The three backends

```sh
./scripts/run.sh                    # modalix - the real path, unchanged, still the default
./scripts/run.sh --backend local    # this document
./scripts/run.sh --backend fake     # synthetic harness, no models at all
```

`./scripts/dev-local.sh` and `./scripts/dev-ui.sh` are the latter two directly.
Local mode is never the default.

## What is swapped, and what is not

Only the inference. `edge/local_edge.py` serves the same Edge API
(`/health`, `/events`, `/frame.jpg`, `/inspect`, `/inspect_single`,
`/transcribe`) and **imports** its logic from `edge/foreman_edge.py`:

- the rolling evidence window and representative-frame selection
- nested-box suppression
- ROI cropping
- `build_window_prompt` — the judging prompt is byte-identical, pinned by tests
- `parse_window_judgement` — same verdict/boolean cross-check
- the EN/RU/Auto speech constraint (`LocalASR` subclasses `GenAI` and overrides
  only the single forced decode)

The host, the gate and `host/policy.py` are untouched and backend-agnostic.
`edge/foreman_edge.py` differs from the Modalix original only by advertising
`backend: "modalix"` at `/health` and by having the prompt lifted into a shared
function whose output is verified byte-for-byte identical.

## Models

| Role | Model | Runtime | Size |
|---|---|---|---|
| Detector | `onnx-community/yolo26n-ONNX` | ONNX Runtime, CPU | **9.9 MB** |
| ASR | `Systran/faster-whisper-small` (MIT) | faster-whisper / CTranslate2, int8 CPU | **486 MB** |
| VLM | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | mlx-vlm, **Metal** | **1.72 GB** |

Detector weights live in `foreman/models-local/`; Whisper and the VLM live in
`~/.cache/huggingface`. Both paths are gitignored. Budget about **2.3 GB** of
models plus **~500 MB** of Python wheels (`.venv` grows to ~736 MB).

YOLO26n is deliberate: it is the same model family the board runs as
`yolo_26n_mpk`, so the COCO class semantics line up. The weights are the upstream
floats, not the quantised MLA build.

`mlx-whisper` was rejected for ASR — it resolves `torch`, `scipy` and `numba`,
while CTranslate2 needs none of them and reuses the onnxruntime the detector
already brings.

## Local results are never Modalix results

The board runs quantised models compiled to MLA `.elf` artifacts, which run
nowhere else and are not translated here. Expect different numbers and sometimes
different answers. So the backend is named everywhere:

| | Modalix | Local | Fake |
|---|---|---|---|
| Console header | `Modalix connected` (green) | **`Local inference` (amber)** | `Test harness` (amber) |
| `/health` | `backend: "modalix"` | `backend: "local"` + notice | `backend: "fake"` |
| Every audit record | `backend: "modalix"` | `backend: "local"` | `backend: "fake"` |
| CSV / JSON / ZIP export | `backend` column | `backend` column | `backend` column |

Local runs write to `audit-local/`, never to the Modalix `audit/`. The launcher
refuses to continue if the host does not report `backend=local`.

## Measured on this Mac — NOT Modalix numbers

Apple M5 Max, 36 GB. **These belong in no comparison with `docs/BENCHMARKS.md`,**
which is a different chip running different weights.

| | Local (this Mac) | Modalix, for contrast |
|---|---|---|
| Detector, one frame | ~11 ms standalone, **14–15 ms in the live loop** | 6.23 ms |
| Detector, live | **15.0 fps** (camera-bound), unchanged with the VLM resident | 15 fps (camera-bound) |
| ASR, forced language, 2 s clip | **~830 ms** | ~245 ms |
| ASR, Auto EN/RU, 2 s clip | **~1670 ms** (two decodes) | ~490 ms |
| VLM, 3 images | **~915–935 ms**, ~224 tok/s | ~2.1–2.5 s |
| VLM first load | **~0.5 s** warm, ~24 s cold (includes download) | n/a |
| VLM peak memory | **3.11 GB** | n/a |
| End-to-end verdict | **~916–935 ms** | ~2.5 s |

The VLM being faster than the board is not a win for this Mac in any meaningful
sense: it is a different chip, different quantisation and a different runtime.

### Execution provider: CPU for the detector, deliberately

ONNX Runtime offers `CoreMLExecutionProvider` and it is **not** used by default:

| Provider | Top score, same frame | Latency |
|---|---|---|
| CPU | **0.9530** | 10.0 ms |
| CoreML | 0.0001 | 8.8 ms |

CoreML splits the graph into 18 partitions and returns effectively empty output —
a detector that silently sees nothing — for ~12% speed. Opt in with
`FOREMAN_LOCAL_ORT_COREML=1`, and re-verify against CPU before trusting it.
The VLM does use Metal, via MLX.

## Using it

**Speech language** — the header selector offers `English`, `Русский` and
`Auto EN/RU`. It applies to the next recording, is stored in `localStorage`, and
changing it loads nothing and restarts nothing. Auto decodes the clip both ways
and keeps the more likely reading, so a third language cannot come back.

**Checking which backend you are on** — the header reads `Local inference` with
an amber dot, never `Modalix connected` with a green one. `/health` and every
exported row carry `backend`.

**macOS camera permission** — the first run prompts for camera access for the
terminal running `ffmpeg`. If frames never arrive, check
System Settings → Privacy & Security → Camera.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `port 8100/8800 is already in use` | Another Foreman is running. The launcher prints the offending process; stop it, or set `FOREMAN_EDGE_PORT` / `FOREMAN_PORT`. |
| `detector weights missing` | Run the `curl` in the quick start. |
| `local dependencies still missing` | `uv sync --group local`. |
| Camera Check shows a frame but no boxes | Normal if nothing in view is one of the 80 COCO classes. Camera Check shows exactly what the detector reports. |
| No frames at all | macOS camera permission, or another process holds the camera. |
| Verdict is `UNCLEAR` mentioning the local VLM | `mlx-vlm` is not installed. Detector-grounded rules still work; nothing is fabricated. |
| Everything is slow on the first inspection | The VLM loads on start; if it did not, the first call pays ~0.5 s. |

## Differences from Modalix

- Upstream float/4-bit weights, not the quantised MLA build. Predictions differ.
- The detector runs on CPU here (see above); the board uses the MLA.
- `/inspect_single` (the pre-temporal fallback) is not implemented locally.
- Neat Insight, the RTSP path and the DevKit's own telemetry do not exist here.
