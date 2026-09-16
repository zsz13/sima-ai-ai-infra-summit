# Benchmarks

**Every number here was measured on this hardware.** Nothing is estimated,
scaled, or copied from a datasheet.

SiMa.ai's published figures for Modalix (50 TOPS, under 10 W) are claims about
the chip from vendor material. They are not measurements of this application and
are not reported as such anywhere in this project.

> **Two pipeline generations are reported below.** Sections 1-7 are the original
> **single-frame** pipeline, kept verbatim as historical measurements. Section 8
> is the **temporal grounding** pipeline that replaced it. They are not
> interchangeable: the temporal pipeline sends three frames per judgement instead
> of one, so its latency and its effect on detector throughput are different by
> design. Do not compare a number from section 3 with one from section 8 without
> saying which pipeline it came from.

## Setup under test

| | |
|---|---|
| DevKit | Modalix MLSoC DevKit 3.0, `modalix`, aarch64, 16 cores, 5.9 GB RAM, 1.83 GB CMA |
| Host | MacBook Pro, Apple M5 Max, 36 GB, macOS |
| Link | USB-Ethernet, 1000baseT, Mac `192.168.2.1` <-> DevKit `192.168.2.2` |
| Workspace sync | rsync-over-SSH to `/workspace-rsync` (host NFS export could not be configured) |
| Detector | `yolo_26n`, SiMa Model Zoo gen2, INT8, 640x640, COCO-80 |
| VLM | `Qwen3-VL-2B-Instruct-GPTQ-a16w4` (pre-installed on the board) |
| ASR | `simaai/whisper-small-a16w8` |
| Camera | MacBook camera -> H.264 -> RTSP -> DevKit, 1280x720 @ 15 fps |

All three models were resident on the MLA simultaneously for the end-to-end runs.

---

## 1. Detector, standalone

`pyneat.Model.run()` on a single decoded image, batch 1. **What is timed:**
on-device preprocessing, MLA inference, and YOLO26 box decode with NMS - not an
MLA-kernel-only figure. JPEG decode and resize happen once, outside the loop.
5 warm-up iterations excluded.

| Run | Image | Iterations | Median | Min | Max | p95 | Implied single-stream |
|---|---|---|---|---|---|---|---|
| 1 | `000000081061.jpg` | 50 | **6.23 ms** | 6.17 | 6.47 | 6.28 | 160.6 fps |
| 2 | `000000116439.jpg` | 5 | 6.35 ms | 6.33 | 6.45 | 6.36 | 157.6 fps |
| 3 | `000000129492.jpg` | 5 | 6.38 ms | 6.33 | 6.40 | 6.38 | 156.8 fps |

**Median ~6.3 ms, ~157-161 fps single-stream.**

### Pipelined throughput

The shipped `benchmarking/model-benchmark` example, 2000 synthetic frames, batch 1:

| | Measured |
|---|---|
| Latency | **5.21 ms** |
| Throughput | **778.9 inferences/s** |

Throughput far exceeds `1 / latency` because the benchmark keeps several
inferences in flight; the 160 fps figure above is the strictly synchronous
one-at-a-time number from `Model.run()`. Both are real; they measure different
things, and Foreman's pipeline uses the synchronous path.

Model load: 457 ms warm; 4,943 ms on first load (includes unpacking the MPK to
`/media/nvme/simaai/coprocessing/models/`). Graph build: 191 ms.

Detections were checked as semantically correct on three different scenes -
indoor furniture, boats/surfboard/umbrella, street with cars - not merely non-empty.

## 2. Detector, in the live pipeline

Sustained throughput from the live camera, sampled once per second from
`/api/state`.

| Condition | Median | Min | Max | Samples |
|---|---|---|---|---|
| Idle (no judgements) | **15.1 fps** | 14.7 | 15.4 | 30 |
| Under continuous VLM load | **15.2 fps** | 14.6 | 15.5 | 35 |

**Throughput drop under VLM load: -0.7% (within noise).**

**Read this honestly.** The detector is **camera-bound at 15 fps, not
compute-bound** - standalone it sustains ~160 fps, so it runs at roughly 10% of
capacity. The result shows the pipeline comfortably keeps up while the VLM runs
concurrently. It is **not** evidence that detector and VLM do not contend at
saturation; that case was not tested, because no 160 fps source was available.

## 3. Vision-language judgement

31 judgements issued back-to-back while the detector ran on live video.

| | Median | Min | Max | n |
|---|---|---|---|---|
| VLM judgement latency | **1368 ms** | 1129 | 1771 | 31 |

Text-only generation on the same model: TTFT 0.09 s, **125 tokens/s**.
Image generation: TTFT 0.26-0.55 s, **~44 tokens/s**.

### Verdict accuracy and stability

6 standards with known ground truth against a live camera frame, each repeated 4x.

| | Result |
|---|---|
| Correct verdicts | **24/24 (100%)** |
| Self-inconsistent replies (guard fired) | **0/24** |
| Median latency | 1423 ms (min 891, max 1627) |

Standards used: person wearing glasses (pass), wearing a lanyard (pass), wearing
a hard hat (fail), a dog in the picture (fail), wearing a bright yellow safety
vest (fail), a person is visible (pass).

**24 judgements on one scene is a smoke test, not an accuracy claim.** It shows
the model discriminates and is stable on this scene; it does not establish a
defect-detection rate on any real inspection task.

## 4. Speech recognition

`simaai/whisper-small-a16w8` on the MLA, 2.50 s of 48 kHz mono audio.

| Run | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Total (ms) | 262 | 262 | 262 | 262 | 284 | 261 |

**Median 262 ms for 2.50 s of audio -> real-time factor ~0.105, about 9.5x faster
than real time.** Transcript: `" Why is the sky blue?"`, `language=en`,
`avg_logprob -0.045`, `no_speech_prob 0.092`.

Through the full path (browser -> Mac host -> DevKit): **256 ms** model inference,
289 ms total round trip.

### 4.1 Language constraint (English / Russian only)

Whisper-small is multilingual and its compiled `language_detect` stage picks from
~99 languages. On real microphone audio it chose **Bulgarian** for the English
sentence "This person must be holding a phone", producing
`"Това пързина не ме ме ме обгърваме."` The detection stage is the failure, so
Foreman never uses it: every decode pins an explicit ISO code.

The server honours `language=en` / `language=ru` and skips detection entirely.
Measured with `say`-generated speech (Samantha / Daniel / Alex for English,
Milena for Russian), 2 s clips, `avg_logprob` of the forced decode:

| Clip | spoken | forced `en` | forced `ru` | argmax |
|---|---|---|---|---|
| e1 | English | **-0.061** | -1.230 | en |
| e2 | English | **-0.075** | -1.299 | en |
| e3 | English | **-0.106** | -0.238 | en |
| r1 | Russian | -0.512 | **-0.056** | ru |
| r2 | Russian | -0.570 | **-0.050** | ru |
| r3 | Russian | -0.572 | **-0.049** | ru |

**Auto EN/RU decodes the clip both ways and keeps the more likely reading**, so a
third language cannot be returned whatever the audio sounds like. 6/6 correct
here, 12/12 through the edge API. Cost is one extra decode: **~245 ms forced,
~490 ms auto**, both well inside the gate.

A forced decode does not always honour the language - asked for Russian on e3,
Whisper returned the English sentence verbatim at a healthy -0.238. A decode
whose alphabet contradicts the language it was forced into is therefore discarded
before likelihood is compared.

### 4.2 Rejecting speech that is not usable

`no_speech_prob` is the discriminator, not `avg_logprob`:

| Input | `no_speech_prob` | `avg_logprob` | text returned |
|---|---|---|---|
| real speech (6 clips) | 0.002 - 0.005 | -0.05 to -0.11 | correct |
| 2 s digital silence | 0.944 | -0.281 | `Редактор субтитров Н.Закомолдина...` |
| 2 s pink noise | 0.905 | -0.529 | `СПОКОЙНАЯ МУЗЫКА` |
| speech buried in noise | 0.832 | -0.489 | `СПОКОЙНАЯ МУЗЫКА` |

Whisper hallucinates *confidently* on silence - those are well-known artefacts and
they score a perfectly healthy likelihood, so a likelihood threshold alone would
accept them. The gap in `no_speech_prob` is three orders of magnitude, which is why
the threshold sits at 0.60 and is not finely tuned. A rejected transcript leaves
the standard in force untouched and the console says "Speech unclear - please
repeat."

## 5. End-to-end

Item settles in frame -> verdict rendered in the console, measured on the Mac.

| Stage | Measured |
|---|---|
| Detector, per frame in pipeline | 15 fps sustained |
| Gate | `stable_frames` / 15 fps ~ 0.4 s at the default of 6 frames |
| VLM judgement | 1368 ms median |
| **End-to-end, gate fire -> verdict on screen** | **1875 ms** (single logged run) |

From `audit/inspections.jsonl`: `inference_ms 1863.3`, `end_to_end_ms 1875.1` -
i.e. **the Mac-side orchestration, HTTP, evidence storage and UI update together
cost ~12 ms**; essentially all of the latency is the model, on the board.

## 6. Infrastructure

| Quantity | Measured |
|---|---|
| Live camera -> RTSP -> SDK container | 1280x720 H.264 @ 15 fps, decoded, exit 0 |
| Insight video ingest from DevKit | H.264, SPS/PPS seen, 923.9 kbps, 15.0 fps |
| Insight metadata ingest from DevKit | 15 msgs/sec, 0 invalid JSON |
| **Insight metadata-to-video correlation** | **151/151 matched, 0 expired over 10 s (100%)** |
| `dk hello.py` round trip | `[DEVKIT][STDOUT] Hello from your DevKit!` |
| Host test suite | 67 tests, 3.7 s |

The 100% correlation rate validates keeping the H.264 encoder and the detector in
a single Neat `Run`: Insight matches metadata to video within +/-1 ms, and split
timelines drift apart permanently.

---

## Findings worth carrying forward

### INT8 detector scores land on a coarse grid

Confidence values from `yolo_26n` are quantized to a discrete set - observed:
0.500, 0.503, 0.510, 0.512, 0.519, 0.529, 0.567, 0.593, 0.606, 0.684, 0.692 -
and **top out around 0.69**. A threshold above ~0.7 rejects everything; 0.50-0.52
is the noise band. The gate therefore defaults to `FOREMAN_MIN_CONF=0.55`, chosen
from this measurement rather than guessed.

### The 4B VLM on this board is broken; the 2B is not

`Qwen3-VL-4B-Instruct-GPTQ-a16w4` returns degenerate output (a run of `!`
characters) for **every** image input - over HTTP and through the in-process
`pyneat.genai` API, at 448x448 and at native size, with and without a temperature
parameter. Text-only generation on the same model is fine (125 tokens/s), so the
language path works and the vision path does not.

`Qwen3-VL-2B-Instruct-GPTQ-a16w4` is correct **and 4x faster** (1086 ms vs
4010 ms in-process on the same image). Foreman uses the 2B.

### CMA must be reclaimed before loading a large model

The MLA bulk loader allocates from CMA, and CMA pages held by the page cache are
not reclaimed on demand. Loading the VLM failed with
`MLA_LOAD_FAILED ... completed=186 failed_index=143` while `CmaFree` was 407 MB
of 1830 MB, with no other process running. `sync; echo 3 > /proc/sys/vm/drop_caches`
raised `CmaFree` to **1630 MB** and the load then succeeded. `scripts/run-genai.sh`
now does this before loading.

### The decoder pads the luma plane height

A 1280x720 NV12 frame arrives as **1,474,560 bytes, not 1,382,400**: Y occupies
768 rows (720 aligned up to 64) and chroma 384. Reshaping naively to
`(height * 3 // 2, width)` reads chroma from the wrong offset, producing a green
band and a ghosted second copy - which the VLM noticed and reported as "the image
is blurry and has color distortion". `nv12_to_bgr()` derives the aligned height
from the payload size.

---

## How to reproduce

```bash
# [SDK] detector, standalone
dk foreman/edge/first_inference.py \
   --model /media/nvme/foreman/models/yolo_26n_mpk.tar.gz \
   --labels /media/nvme/foreman/coco.txt \
   --image /workspace-rsync/assets/images/000000081061.jpg --iterations 50

# [HOST] detector throughput idle vs under VLM load
python3 scripts/measure-load.py

# [DEVKIT] verdict accuracy and stability (copy scripts/consistency-eval.py over first)
python3 /tmp/consistency-eval.py 4

# [HOST] summarise the audit trail
python3 - <<'PY'
import json, statistics as st
rows = [json.loads(l) for l in open("audit/inspections.jsonl")]
for key in ("inference_ms", "end_to_end_ms"):
    vals = [r["metrics"][key] for r in rows if key in r["metrics"]]
    if vals:
        print(f"{key:15} n={len(vals):3} median={st.median(vals):8.1f} "
              f"min={min(vals):8.1f} max={max(vals):8.1f}")
PY
```

## 7. Power: not measurable on this DevKit

`benchmarking/model-benchmark` reports power and energy, and it was run:

```
Power avg:    0.00 W
Energy:       0.00 J
Measurement detail: e2e+throughput-only
```

Both are **zero because this board exposes no power rail to the API**, not
because the load is free. `/sys/class/hwmon` offers only
`simaai_modalix_thermal_sensor` and `lm96163` - temperature sensors, no power monitor.

**No power figure is claimed anywhere in this project.** SiMa's "under 10 W" is
their published number for the chip and is quoted as theirs, never as ours.

What *is* measurable is temperature, with all three models resident and the
detector running on live video:

| Sensor | Reading |
|---|---|
| `simaai_modalix_thermal_sensor` (14 zones) | 49-50 C |
| `lm96163` temp1 / temp2 | 55 C / 57 C |

Load average 3.49 across 16 cores. Passive, with the DevKit's stock cooling.

## 8. Temporal grounding (current pipeline)

Measured after the change from single-frame to a 3-second rolling evidence
window. Same DevKit, same detector, same VLM; the camera is 15 fps nominal.

### Cost

| | Single-frame (historical) | **Temporal (current)** |
|---|---|---|
| Frames judged per inspection | 1 | **3**, spread over a 3 s window |
| VLM calls per inspection | 1 | **1** (multi-image, verified supported) |
| VLM latency, median | 1368 ms (n=31) | **2133 ms** (n=10, min 1818, max 2404) |
| Detector fps, idle | 15.1 | **16.7** (n=25) |
| Detector fps, under load | 15.2 (-0.7%) | **15.2 (-9.0%)** (n=25) |
| Frame selection | n/a | **0.3 ms** (n=10) |
| Host overhead per inspection | ~12 ms | **19.5 ms** median |
| Evidence buffer | none | **46 frames, 16 images, 0.97 MB** |

**1.56x the VLM latency for 3x the evidence**, because one multi-image request is
used rather than three separate calls.

The **-9.0% detector throughput under load is a real cost**, and larger than the
-0.7% the single-frame pipeline showed - the multi-image request does more MLA
work. The detector is still camera-bound at 15 fps with roughly 10x headroom, so
the live view is unaffected; this figure would matter at saturation.

### Multi-image support

Verified on this hardware before designing around it: three images in one
OpenAI-compatible request are received and described **individually and in order**
(`{"count": 3}`, correct word and colour per image) in **3059 ms**. A contact
sheet and N separate calls were therefore not needed.

### Temporal grounding thresholds

`host/policy.py` defaults, chosen from measurements on this board rather than guessed:

| Threshold | Default | Why |
|---|---|---|
| `min_confidence` | 0.55 | INT8 detector scores land on a coarse grid topping out at ~0.69; 0.50-0.52 is the noise band (section "Findings"). |
| `absent_ratio_max` | 0.10 | An object genuinely held in frame for 3 s is detected far more often than 10% of frames. Measured: a present person scores 100% (49/49); a phone that is truly absent scores 0%. |
| `present_ratio_min` | 0.60 | Leaves room for occlusion and motion blur. Measured intermediate case: a phone at the edge of visibility scored 11% (5/45), which lands in the UNCLEAR band, not PASS. |
| `min_window_frames` | 10 | Below this the window cannot support a judgement; the orchestrator re-arms the gate rather than judging. |

### Observed detector behaviour used to set them

| Scene | Class | Presence |
|---|---|---|
| Person at a desk, no phone | `person` | **49/49 (100%)**, median confidence 0.69 |
| Person at a desk, no phone | `cell phone` | **0/49 (0%)** |
| Person at a desk, no bottle | `bottle` | **0/49 (0%)** |
| Person, phone at the edge of visibility | `cell phone` | **5/45 (11%)** -> UNCLEAR band |

## Not yet measured

- **Board power and energy per inference** - not exposed by this hardware
  (see section 7).
- **Detector throughput at saturation** (a source faster than 15 fps), which is
  the case where MLA contention would actually show.
- **Accuracy on a real inspection task** with a held-out item set and a human
  reference.
