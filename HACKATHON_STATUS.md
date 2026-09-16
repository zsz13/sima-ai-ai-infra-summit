# Hackathon Status

_Last updated: full pipeline working end to end on real Modalix hardware._

# Objective

Build a working end-to-end ML application in which the **SiMa.ai Modalix MLSoC
DevKit 3.0 performs meaningful, measured ML workloads via the MLA**, built with
the Palette Neat toolchain, with an Apple M5 Max MacBook Pro as a hybrid
orchestration/UI node where that is the better engineering choice.

**Status: achieved.** Three models run on the MLA (detector, vision-language
model, Whisper). The Mac runs no model.

# Challenge Requirements

No formal judging rubric exists (see `CHALLENGE.md`). The track statement is
**"SiMa - Building Physical AI That sees, understands, and acts"**, used as
product direction:

| Stage | Foreman |
|---|---|
| **SEE** | `yolo_26n` detector on the MLA, every frame of a live camera |
| **UNDERSTAND** | `Qwen3-VL-2B` on the MLA judges the frame against a spoken standard and states a reason |
| **ACT** | Verdict, evidence frame, audit record, andon console |

# Architecture

**"Foreman" - spoken-spec zero-shot inspection.** Rationale and the two rejected
candidates in `foreman/docs/ARCHITECTURE.md`. Product lives in `foreman/`.

Camera -> DevKit -> detector on MLA (continuous, cheap) -> Mac gate decides when an
item has settled -> DevKit VLM on MLA judges it (expensive, rare) -> Mac records
verdict + evidence and updates the console. Video and detection boxes also stream
to Neat Insight for the engineering view.

Every inference is on Modalix. The Mac does gating, policy, audit and UI.

# Working

## Verified on real hardware

| Milestone | Evidence |
|---|---|
| DevKit reachable | `192.168.2.2`, 1.3 ms RTT, 1000baseT via Internet Sharing |
| Official smoke test | `dk hello.py` -> `[DEVKIT][STDOUT] Hello from your DevKit!` |
| Model Zoo | 190 gen2 models resolved; `yolo_26n` downloaded and staged |
| **First MLA inference** | 6.23 ms median over 50 runs, correct COCO detections |
| Live camera path | Mac camera -> RTSP -> DevKit, 1280x720 H.264 @ 15 fps |
| Detector on live video | 15 fps sustained, real detections streamed back to the Mac |
| **Neat Insight overlays** | live DevKit video + boxes, **151/151 metadata matched, 0 expired** |
| **Whisper on the MLA** | `" Why is the sky blue?"`, 262 ms for 2.50 s audio (RTF ~0.105) |
| **VLM on the MLA** | correct, discriminating verdicts with grounded reasons, 1368 ms median |
| **Full Foreman loop** | gate fires -> VLM judges -> verdict + evidence + audit, 1875 ms end to end |
| Speech -> standard | browser audio -> Mac -> DevKit Whisper -> standard set, 289 ms round trip |
| Host test suite | 67 tests pass, `ruff check` clean |

Screenshots: `foreman/docs/ui-live.png` (product console, live verdict),
`foreman/docs/insight-overlay.png` (Insight with DevKit boxes).

## Stack currently running

| Where | Process |
|---|---|
| Mac | camera -> RTSP (`scripts/stream-camera.sh 0 src2`) |
| Mac | Foreman console on `http://127.0.0.1:8800` |
| SDK container | Neat Insight on `https://127.0.0.1:9900`, viewer on `:8081` |
| DevKit | GenAI server (`vlm` + `asr`) on `:9998` |
| DevKit | Foreman edge agent (detector + API) on `:8100` |

# In Progress

Nothing blocking. Remaining optional work is listed under *Next 3 Actions*.

# Blockers

**B1 (no Ethernet) and B4 (Model Zoo login) are CLEARED.** Remaining issues are
worked around, not blocking:

### B5 - The 4B VLM on this board is broken (WORKED AROUND)

`Qwen3-VL-4B-Instruct-GPTQ-a16w4` returns degenerate output (a run of `!`) for
every image input - over HTTP and via the in-process `pyneat.genai` API, at
448x448 and native size, with and without a temperature parameter. Text-only
generation on the same model is fine (125 tokens/s), so the language path works
and the vision path does not.
**`Qwen3-VL-2B-Instruct-GPTQ-a16w4` is correct and 4x faster.** Foreman uses the 2B.

### B6 - Host NFS export unavailable (WORKED AROUND)

`sima-cli sdk setup --devkit` could not configure the macOS NFS export (it needs a
sudo password it could not obtain), so workspace sync uses the **rsync-over-SSH
fallback**. `dk` works normally; board paths are `/workspace-rsync/...`.

### B7 - `sdk setup` recreated the SDK container (NOTED)

`sima-cli sdk setup --devkit ... --noninteractive --yes` **removed and recreated**
`ghcr.io-sima-neat-sdk-v2.1.3.0` rather than reusing it. No data lost - `/workspace`
is a host bind mount - but container-local state (SSH keys) was reset and the
Model Compiler extension re-downloaded (~9 GB). Worth knowing before re-running it.

### B8 - CMA must be reclaimed before loading a large model

MLA bulk load fails with `MLA_LOAD_FAILED` when `CmaFree` is low, even with
nothing else running, because CMA pages held by the page cache are not reclaimed
on demand. `scripts/run-genai.sh` now runs `sync; echo 3 > /proc/sys/vm/drop_caches`
first (measured: 407 MB -> 1630 MB free).

# Known-Good Commands

```bash
# [HOST] the whole demo
cd ~/workspace/foreman
DEVKIT_IP=192.168.2.2 FOREMAN_MODEL=/media/nvme/foreman/models/yolo_26n_mpk.tar.gz \
  ./scripts/demo.sh
./scripts/stop.sh

# [HOST] SDK container / shell
sima-cli sdk neat
docker exec -u d ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc '<cmd>'

# [SDK] run on the DevKit (dk is a shell function from ~/.devkit-sync.rc)
source ~/.devkit-sync.rc
dk status ; dk sync foreman ; dk hello_kit/hello.py

# [HOST] DevKit shell + serial
ssh sima@192.168.2.2          # key auth configured
sima-cli serial               # login sima / edgeai

# [HOST] health checks
curl -s http://192.168.2.2:8100/health          # edge agent
ssh sima@192.168.2.2 'curl -s localhost:9998/v1/models'   # GenAI server
curl -s http://127.0.0.1:8800/api/state         # Foreman console
```

## Board paths

| What | Where |
|---|---|
| Detector MPK | `/media/nvme/foreman/models/yolo_26n_mpk.tar.gz` |
| COCO labels | `/media/nvme/foreman/coco.txt` |
| LLiMa model catalog | `/media/nvme/llima/models/` (`llima list`) |
| Synced workspace | `/workspace-rsync/foreman/` |
| Logs | `/tmp/foreman-edge.log`, `/tmp/foreman-genai.log` |

## Service port map

| Service | Port | Proto |
|---|---|---|
| Foreman console (Mac) | 8800 | tcp |
| Foreman edge API (DevKit) | 8100 | tcp |
| GenAI server (DevKit, localhost) | 9998 | tcp |
| Neat Insight UI | 9900 | tcp (https) |
| Insight video viewer | 8081 | tcp (https) |
| RTSP | 8554 | tcp |
| Insight video / metadata ingest | 9000-9003 / 9100-9103 | udp |
| VS Code Web | 9999 / 10000 | tcp |

# Benchmarks

Full detail in `foreman/docs/BENCHMARKS.md`. Headlines, all measured here:

| | Measured |
|---|---|
| Detector standalone | **6.3 ms median, ~160 fps** |
| Detector in pipeline | 15.1 fps idle / 15.2 fps under VLM load (camera-bound) |
| VLM judgement | **1368 ms median** (n=31) |
| Whisper ASR | **262 ms for 2.50 s audio, RTF ~0.105** |
| End-to-end verdict | **1875 ms**, of which ~12 ms is the Mac |
| Verdict accuracy | **24/24** over 6 standards x 4 reps, 0 self-inconsistencies |

Not yet measured: board power/energy, detector throughput at saturation,
accuracy on a real inspection task.

# Demo Status

**Working end to end on real hardware.** `foreman/scripts/demo.sh` brings up the
whole stack with preflight checks; `stop.sh` tears it down.
`foreman/docs/DEMO.md` has labelled `[HOST]`/`[SDK]`/`[DEVKIT]` commands, expected
output, failure recovery, and a recorded-clip fallback.
`foreman/docs/PITCH.md` has the pitch and a 2-minute demo script.

**No fake output anywhere in the demo path.** The test harness is labelled in its
own source, `demo.sh` never starts it, and the console reports "Modalix
unreachable" rather than a stale verdict when the edge is down.

# Next 3 Actions

1. **Measure board power** with `benchmarking/model-benchmark` so the "edge AI at
   low power" claim is ours, not the vendor's.
2. **Update `demo.sh` for the rsync topology** and rehearse it cold from
   `stop.sh` to a live verdict, so the one-command path is proven, not assumed.
3. **Rehearse the pitch** against `docs/PITCH.md`, including the failure drills.
