# Architecture

Designed against the track statement **"Physical AI that sees, understands, and
acts"** and the environment established in `ENVIRONMENT_INVENTORY.md`.

Every workload below is tagged `[MODALIX]`, `[MAC]`, `[BROWSER]`, or
`[OPTIONAL CLOUD]`.

---

## Hard constraints the design must respect

| Constraint | Source | Design consequence |
|---|---|---|
| Container cannot execute models | verified: no pyneat, no MLA node | All inference is on the DevKit. The Mac cannot substitute. |
| **One MLA, shared** | tutorial 021 README | Detector and VLM contend. Extra processes do **not** multiply throughput. Budget one continuous model + one event-triggered model. |
| Whisper is **pinned** in memory | GenAI Studio | ASR always costs MLA memory; only one chat/VLM model is resident beside it. |
| 16 GB LPDDR5 on board | DevKit 3.0 spec | Whisper (1.13 GB) + one 4B VLM is comfortable; two large models is not. |
| Insight = **4 channels** | `neat-port-map.json` | Max 4 video + 4 metadata channels. Channels ≥4 silently have no host port. |
| Overlay correlation is **±1 ms** | Insight skill | Encoder and detector must live in **one `Run`** on one GStreamer timeline. |
| Model Zoo needs login (**B4**) | verified Auth0 redirect | Any classic-CV model is gated. HuggingFace/GenAI is not. |
| VLM latency is seconds, not ms | model class | A VLM can never be the continuous perception stage. It must be event-triggered. |

That last row is the single most important architectural fact: **the SEE stage
must be cheap and continuous; the UNDERSTAND stage must be expensive and rare.**

---

## Candidate 1 — "Say-What-You-Watch": voice-programmable sentinel

Speak a rule in plain English; the system compiles it into a machine-checkable
watch condition and enforces it continuously against a live detection/track stream.

```
Microphone ──> [MODALIX] Whisper ASR (MLA)
                    │ transcript
                    v
              [MODALIX] small LLM (MLA) ──> structured rule JSON
                    │
Camera/RTSP ──> [MODALIX] YOLO26 detect + track (MLA, continuous)
                    │ detections + track IDs
                    v
              [MAC] rule evaluator / state machine / dwell timers
                    │
                    +──> [BROWSER] live overlay + rule status
                    +──> ACT: alert, webhook, spoken confirmation
```

| Dimension | Assessment |
|---|---|
| Problem usefulness | High — reconfiguring a vision system today means a developer, not a sentence. |
| UX | Excellent. "Tell it what to watch for" is instantly graspable. |
| Implementation time | **Highest.** The NL→rule compiler is the whole risk and it is bespoke. |
| Modalix utilization | Very high — 3 model families on the MLA. |
| Existing SiMa assets | `single-stream-object-detector`, tracking example, tutorial 021 (ASR), tutorial 019 (LLM). |
| Mac utilization | Rule evaluation, timers, UI. |
| Demo quality | High — speaking a new rule mid-demo is a genuine "wow". |
| Reliability | **Medium-low.** LLM rule compilation is the failure point, and it fails *visibly*, on stage. |
| Differentiation | High. |
| Risk | **High** — 3 MLA models contending; bespoke compiler; depends on B4. |

## Candidate 2 — "Foreman": spoken-spec zero-shot inspection & compliance cell

You state the standard once, in speech. A cheap detector watches continuously and
triggers a VLM only when an item is present and settled. The VLM judges the item
against the spoken standard and the system acts on the verdict.

```
Microphone ──> [MODALIX] Whisper ASR (MLA) ──> spoken standard, once
                                                      │
Camera/RTSP ──> [MODALIX] YOLO26 detect (MLA, continuous, cheap) ── SEE
                    │           │
                    │           └─ presence + stability gate  [MAC]
                    │                        │ (fires once per item)
                    │                        v
                    │              [MODALIX] VLM Qwen3-VL (MLA)  ── UNDERSTAND
                    │                        │ verdict + reason
                    v                        v
              [MAC] session state, audit log, PASS/FAIL policy  ── ACT
                    │
                    +──> [BROWSER] product UI: verdict, reason, evidence frame
                    +──> [MODALIX] Insight: live video + detection overlay
                    +──> ACT: accept/reject signal, ticket, spoken result
```

| Dimension | Assessment |
|---|---|
| Problem usefulness | **Very high.** Per-defect detector training is infeasible for long-tail QA/compliance. Zero-shot against a stated standard is the actual unmet need. |
| UX | Excellent and tactile — hold an item up, get a verdict with a reason. |
| Implementation time | Medium. Event-gating is simple; no bespoke compiler. |
| Modalix utilization | **Very high and well-matched** — VLM is unambiguously accelerator-class work; ASR also on MLA. |
| Existing SiMa assets | **`detection-to-vlm-assistant` is already this shape** + tutorial 021 for ASR. |
| Mac utilization | Gating, state, audit trail, UI. Honest division of labour. |
| Demo quality | **Highest.** Three stages visible in one gesture; the VLM's stated *reason* is the "understand" proof. |
| Reliability | **High.** No hard real-time requirement on the expensive model; the cheap model carries the live view. |
| Differentiation | High — zero-shot, no training data, spec changes by voice. |
| Risk | **Medium** — VLM latency must be hidden by the gate; depends on B4 for the detector (degrades to motion-gate). |

## Candidate 3 — "Watchfloor": multi-stream situational-awareness console

Four concurrent camera streams, continuous detection + tracking, cross-stream
state reasoning, and a prioritized action queue with a voice query interface.

```
4x RTSP ──> [MODALIX] YOLO26n detect + track, 4 channels (MLA)  ── SEE
                    │ per-channel detections + track IDs
                    v
              [MAC] zone logic, dwell, crowding, anomaly scoring  ── UNDERSTAND
                    │
Microphone ──> [MODALIX] Whisper ASR (MLA) ──> "what's happening at camera 3?"
                    │                                  │
                    v                                  v
              [MAC] live-state query answering ────────┘
                    │
                    +──> [BROWSER] 4-up wall + ranked dispatch queue  ── ACT
```

| Dimension | Assessment |
|---|---|
| Problem usefulness | High — operator overload in multi-camera monitoring is real. |
| UX | Good; a 4-up wall reads as "serious system". |
| Implementation time | **Lowest.** `high-density-multi-stream-object-detector` is near-ready; most work is config + Mac-side logic. |
| Modalix utilization | **Highest raw throughput** — and yields a defensible measured number ("N streams at M FPS, measured"). |
| Existing SiMa assets | `high-density-multi-stream-object-detector`, `multi-stream-people-tracker`, tutorial 015. |
| Mac utilization | All event reasoning and UI. |
| Demo quality | Medium-high. Impressive, but **ACT is the weakest** — a queue is a list, not an action. |
| Reliability | **Highest** — fewest novel parts. |
| Differentiation | **Lowest.** This is the archetypal edge-AI demo; several teams will build it. |
| Risk | **Low-medium** — depends on B4; needs 4 sources (Insight can supply them). |

---

## Decision

**Primary: Candidate 2 — "Foreman".**
**Fallback: Candidate 3 — "Watchfloor".**

### Why Foreman

1. **It makes the three-stage loop literal.** SEE is a continuous detector,
   UNDERSTAND is a VLM that states its reasoning in words, ACT is a verdict that
   changes something. A judge sees all three in one gesture, in well under 20 seconds.
2. **The workload split is honest, not decorative.** Candidate 1 puts three models
   on one contended MLA; Candidate 3 puts a trivial workload on the Mac. Foreman
   gives Modalix exactly the work an accelerator is for — continuous CNN inference
   plus transformer inference — and gives the Mac exactly what a laptop is for:
   state, policy, audit, UI.
3. **VLM latency stops being a liability.** The expensive model is event-triggered,
   so seconds-per-inference is a feature (deliberate, per-item judgement), not a
   dropped-frames problem. Candidate 1 has no such shelter.
4. **It reuses a SiMa example of the same shape.** `detection-to-vlm-assistant` is
   detector→VLM on Modalix already, which is the user's stated preference order:
   adapt an existing example over building from scratch.
5. **The edge argument is genuine, not recited.** Factory and retail camera footage
   is exactly the data that legally and commercially cannot go to a cloud. Zero-shot
   inspection *without* shipping video off-site is a real reason to own this silicon.
6. **Originality.** Object detection on an edge box is table stakes. *Restating the
   quality standard by voice and having the system re-inspect against it, with no
   retraining,* is not.

### Why not the others

- **Candidate 1** is the most exciting idea and the most likely to fail on stage.
  The NL→rule compiler is bespoke, unvalidated, and fails visibly. Its best ideas
  (voice input, natural-language configuration) are absorbed into Foreman at a
  fraction of the risk: Foreman takes a spoken *standard*, which the VLM consumes
  directly as a prompt — no compiler in the middle.
- **Candidate 3** is kept as the fallback precisely because it is the safest and
  fastest. If the DevKit lands late or the MLA contention proves worse than
  documented, we ship Watchfloor. Its weakness is that it is the demo everyone
  else builds, and its ACT stage is thin.

### What would change this decision

- **B4 unresolved (no Model Zoo login)** → the detector is unavailable. Foreman
  degrades to a Mac-side motion/stability gate driving the same MLA VLM; SEE
  becomes weaker but the product survives. Candidate 3 would *not* survive.
- **VLM measured slower than ~4 s/item** → raise the gate threshold and inspect
  on explicit trigger rather than automatic presence.

---

## Chosen architecture in detail

```
                         ┌───────────────────────────────────┐
   microphone            │   MODALIX MLSoC DevKit 3.0        │
   (Mac built-in)        │   16GB LPDDR5 · MLA · A65         │
        │                │                                   │
        │  WAV/PCM       │  ┌─────────────────────────────┐  │
        └───── HTTP ────>│  │ [MODALIX] Whisper small     │  │
                         │  │  a16w8  ASR  — on MLA       │  │  SEE (audio)
                         │  └──────────────┬──────────────┘  │
                         │                 │ transcript       │
                         │                 v                  │
   camera / RTSP         │  ┌─────────────────────────────┐  │
        │                │  │  spoken standard (text)     │  │
        └───── RTSP ────>│  └──────────────┬──────────────┘  │
                         │                 │                  │
                         │  ┌──────────────┴──────────────┐  │
                         │  │ [MODALIX] YOLO26 detect     │  │  SEE (vision)
                         │  │  MLA · continuous · cheap   │  │
                         │  └───┬─────────────────────┬───┘  │
                         │      │ detections          │ H264 │
                         │      │ + boxes             │      │
                         │      │        ┌────────────┴────┐ │
                         │      │        │ VideoSender     │ │
                         │      │        │ (same Run!)     │ │
                         │      │        └────────┬────────┘ │
                         │      │                 │          │
                         │  ┌───┴─────────────────┴───────┐  │
                         │  │ [MODALIX] Qwen3-VL 4B a16w4 │  │  UNDERSTAND
                         │  │  MLA · event-triggered      │  │
                         │  │  prompt = spoken standard   │  │
                         │  └──────────────┬──────────────┘  │
                         └─────────────────┼─────────────────┘
                    metadata UDP 9100 ─────┼───── video UDP 9000
                    verdict JSON / HTTP ───┤
                                           v
                         ┌─────────────────────────────────────┐
                         │  MacBook Pro M5 Max (36 GB)         │
                         │                                     │
                         │  [MAC] presence + stability gate    │   decides WHEN
                         │  [MAC] session state machine        │   to spend the VLM
                         │  [MAC] PASS/FAIL policy             │   ACT
                         │  [MAC] audit log + evidence frames  │
                         │  [MAC] local HTTP API               │
                         └──────────────┬──────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    v                   v                   v
        ┌───────────────────┐  ┌────────────────┐  ┌──────────────────┐
        │ [BROWSER]         │  │ [BROWSER]      │  │  ACT             │
        │ Foreman product   │  │ Neat Insight   │  │  accept/reject   │
        │ UI — verdict,     │  │ :8081 live     │  │  ticket          │
        │ reason, evidence, │  │ video + box    │  │  spoken result   │
        │ live metrics      │  │ overlay        │  │                  │
        └───────────────────┘  └────────────────┘  └──────────────────┘
```

### Workload assignment and justification

| Workload | Where | Why there |
|---|---|---|
| Whisper ASR | **`[MODALIX]`** | SiMa ships an MLA-compiled Whisper (encoder + full decoder + language-detect as MLA ELFs). Running it on the Mac would mean installing an ML stack from scratch **and** wasting the accelerator that already has it. |
| Object detection | **`[MODALIX]`** | Continuous CNN inference is precisely MLA work. Neat also does anchors/NMS in `Model::Options`. |
| VLM judgement | **`[MODALIX]`** | Transformer inference on 4B params; the single heaviest workload and the clearest reason the silicon exists. |
| H.264 encode + RTP | **`[MODALIX]`** | Must share the detector's `Run` for ±1 ms overlay correlation. Encoding on the Mac would break it. |
| Presence/stability gate | **`[MAC]`** | Pure state logic over a metadata stream. Putting it on the DevKit would burn MLA-adjacent CPU for nothing. |
| Session state, policy, audit | **`[MAC]`** | Business logic and durable storage. No accelerator benefit. |
| Product UI | **`[BROWSER]` / `[MAC]`** | Presentation. Insight stays as the engineering/profiling view. |
| Cloud | **none** | Deliberately absent. The privacy argument is the product; a cloud dependency would undercut it. |

### Communication

Simplest reliable mechanisms already present — no new infrastructure:

| Link | Mechanism |
|---|---|
| Mac → DevKit (audio, VLM prompts) | HTTP, OpenAI-compatible `:9998` (tutorial 021) |
| DevKit → Insight (video) | UDP RTP `9000+N` (`VideoSender`) |
| DevKit → Insight (detections) | UDP JSON `9100+N` (`MetadataSender`) |
| DevKit → Mac (detections, verdicts) | HTTP/JSON to the Mac orchestrator |
| Mac ⇄ DevKit (files, models) | the existing `/workspace` NFS mount |
| Browser → Mac | HTTP + SSE |

No Kafka, Redis, or queue broker. None is warranted.

---

## Measurement plan

Recorded in `docs/BENCHMARKS.md`, measured, never estimated. **Model inference
latency and end-to-end system latency are reported separately.**

| Metric | Definition |
|---|---|
| Detector inference latency | MLA time per frame, from Neat's profiler |
| Detector throughput | sustained FPS on the live stream |
| ASR inference latency | MLA time per utterance + audio duration (report RTF) |
| VLM inference latency | MLA time per judgement; TTFT and tokens/s from `GenerationMetrics` |
| **End-to-end verdict latency** | item settles in frame → verdict on screen |
| Power / energy | via `benchmarking/model-benchmark`, which reports both |
| Mac CPU / memory | orchestrator cost, to prove the split is honest |

No number from the vendor flyer (50 TOPS, <10 W) will be presented as ours.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| DevKit never comes up (B1) | Medium | Blocked on a USB-Ethernet adapter. Nothing ships without it. |
| Model Zoo login (B4) | Medium | Degrade SEE to a Mac-side motion/stability gate; VLM path is HuggingFace-only and unaffected. |
| MLA contention detector↔VLM | Medium | Gate ensures the VLM runs rarely; measure detector FPS with and without VLM load and publish both. |
| VLM too slow | Medium | Switch to explicit trigger; or drop to LFM2-VL-1.6B. |
| Model name drift `-GPTQ-` vs `-Autoround-` | High | Resolve names against the live HF API before scripting. |
| Insight overlay desync | Medium | Keep encoder + detector in one `Run`; verify with `matched_*` counters. |
| Model download time at venue | High | **Already mitigated** — Whisper pre-staged to `/workspace/models/`. |
| Demo needs a physical prop | Certain | Prepare props + a recorded fallback clip served through Insight. |
