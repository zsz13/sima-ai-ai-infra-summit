# Pitch and demo script

> Every number below is measured on our own DevKit and matches `BENCHMARKS.md`.
> Do not quote a number that is not in that file. If a judge asks something we
> have not measured — **power, above all** — the answer is "we haven't measured
> that", not the vendor's figure.

---

## 1. One sentence

**Foreman lets you state a quality standard out loud, and a Modalix edge box then
watches the line and judges every item against it — with no training data and no
video ever leaving the room.**

## 2. Thirty seconds

Machine vision handles the defects you can afford to train for. Most real quality
rules aren't like that — they're written in words, they're long-tail, and they
change. So they stay manual, inconsistent, and recorded on a clipboard.

Foreman puts a model that understands both language and images on the line
itself. You say *"every carton needs a lid and a label facing up."* A detector
runs continuously on the Modalix MLA. When an item settles, a vision-language
model on the same chip judges it against your sentence and tells you **why** it
passed or failed. Changing the standard is a sentence, not a sprint.

All of it — speech recognition, detection, and visual reasoning — runs on the
Modalix DevKit. Nothing goes to a cloud. That's the point: factory and retail
footage is exactly the video that can't leave the building.

## 3. Two-minute demo script

**[0:00 — the console, before touching anything]**
> "This is a line inspection station. Everything you're about to see runs on this
> board" — *point at the DevKit* — "not in a cloud, and not on the laptop. The
> laptop is a screen and a microphone."

**[0:15 — state the standard]**
Press **State the standard**, and say:
> *"Every box must have a label facing up and the lid closed."*

Point at the top bar as the sentence appears.
> "That was Whisper. Running on the Modalix MLA — SiMa ship it compiled to the
> accelerator, encoder and decoder both."

**[0:35 — present a good item]**
Place a closed, labelled box under the camera. Point at the thin amber bar.
> "The detector is running on every frame. It's waiting for the item to stop
> moving — because the next step is expensive and we only want to pay for it once."

Wait for **PASS**.
> "And there's the reason, in words. Not a confidence score — a sentence an
> operator can agree or disagree with."

**[1:00 — present a bad item]**
Open the lid, or turn the label away. Present it.
> "Same standard. Different item."

**FAIL**, with the reason naming the actual problem.

**[1:20 — the part that matters]**
Press **State the standard** again:
> *"I don't care about the label. I only care that the lid is closed."*

Present the *same* item that just failed. It passes.
> "No retraining. No new dataset. No redeploy. The rule changed because someone
> said a different sentence. That's the thing you can't do with a trained detector."

**[1:40 — the engineering view]**

Switch to the Neat Insight tab.
> "Same stream, with the detector's boxes overlaid — that's Insight, SiMa's own
> tooling, reading video and metadata straight off the board. And every verdict
> you just saw is on disk with the exact frame that produced it, which is the
> audit trail this job actually needs."

**[2:00 — stop]**

### Demo failure drills

- **Camera dies** → `SKIP_CAMERA=1 SLOT=src1`, serve the recorded clip from Insight.
- **Verdict slow** → say so: "that's a 4-billion-parameter model on a 10-watt part,
  and it's sharing the accelerator with the detector."
- **Edge drops** → the console says *Modalix unreachable*. Point at it: "it tells
  you the truth instead of showing a stale verdict." Then `./scripts/stop.sh` and
  re-run `demo.sh`.

## 4. Architecture in one breath

> Camera into the DevKit over RTSP. A YOLO26 detector runs on the MLA every
> frame, and pushes video and boxes to Neat Insight. The Mac watches the
> detection stream and decides *when* an item has settled — that's the only
> interesting thing the Mac does. It then asks the board to judge the current
> frame, and a vision-language model on the same MLA returns a verdict and a
> reason. The Mac records it with the evidence frame.
>
> Every inference is on Modalix. The Mac runs no model.

The split is the design: a VLM takes seconds, so it can never be the continuous
perception stage; a detector takes milliseconds, so it can. Something has to
decide "expensive model — now", and that's control logic, not perception.

## 5. Three strongest technical points

1. **Three model families resident on one accelerator, correctly scheduled.**
   Whisper ASR, a CNN detector and a vision-language model all on the Modalix MLA
   simultaneously, with an explicit gate so the expensive one runs once per item
   rather than per frame. We publish detector throughput **under VLM load**, not
   just idle — and we say plainly that the detector is camera-bound at 15 fps, so
   that result does not prove there is no contention at saturation.
2. **Correct use of the Neat graph model.** The H.264 encoder and the detector
   share a single `Run`, because Insight correlates RTP and metadata timestamps
   within ±1 ms — split them across two Runs and the overlays drift apart
   permanently. Preprocessing, anchor decode and NMS are pushed into
   `Model::Options` and run on-device.
3. **The edge agent is standard-library-only** — nothing to install in the board's
   PyNeat environment. 67 tests run the real wire protocol over a real socket with
   no hardware. And we found and fixed three real hardware-level defects: the
   decoder pads the luma plane to 768 rows (a naive reshape gives a green band and
   a ghosted image), CMA must be force-reclaimed before a large MLA load, and the
   4B VLM on this board returns degenerate output for every image while the 2B is
   correct and 4× faster.

## 6. Three strongest product points

1. **Zero training data.** The standard is a prompt. The long tail of quality
   rules becomes addressable for the first time.
2. **The reason is the product.** A sentence can be overruled by a human;
   a confidence score can only be argued with. That is what makes it deployable
   next to a person.
3. **The footage never leaves.** This is the commercial unlock, not a feature
   bullet. The sites that most need automated inspection are the ones that cannot
   send video to an API.

## 7. Measured performance

All on a Modalix MLSoC DevKit 3.0. Full method in `BENCHMARKS.md`.

| | Measured |
|---|---|
| Detector, standalone | **6.3 ms median → ~160 fps** (50 runs) |
| Detector, pipelined | **778.9 inferences/s** at 5.21 ms (2000 frames) |
| Detector in the live pipeline | 15.1 fps idle / 15.2 fps under VLM load |
| VLM judgement | **1368 ms median** (n=31, min 1129, max 1771) |
| Whisper ASR | **262 ms for 2.50 s audio → RTF ≈ 0.105, ~9.5× real time** |
| **End-to-end: gate fires → verdict on screen** | **1875 ms**, of which **~12 ms is the Mac** |
| Verdict accuracy | **24/24** over 6 standards × 4 repetitions, 0 self-inconsistencies |
| Insight metadata↔video correlation | **151/151 matched, 0 expired** |
| Board temperature, full stack | 49–50 °C SoC, 55–57 °C board |
| Board power | **not measurable — this DevKit exposes no power rail** |

### The two numbers worth saying out loud

> "The detector is **6 milliseconds**. The judgement is **1.4 seconds**. That
> ratio *is* the architecture — we run the cheap one on every frame and the
> expensive one once per item."

> "End to end it's **1.9 seconds**, and **12 milliseconds of that is the laptop**.
> Essentially all the latency is the model, on the board."

### If asked about power

"We couldn't measure it. This DevKit doesn't expose a power rail — the benchmark
returns 0.00 W, which means unavailable, not free. SiMa publish under 10 W for
the chip; that's their number, not ours. What we did measure is 49–50 °C on the
SoC with three models resident and passive cooling." 

## 8. Why edge AI, and why Modalix

The honest argument is not latency. It is that **this video cannot be sent
anywhere.** Factory floors, pharmacy benches, retail back-rooms — the footage is
commercially sensitive, often personally identifying, and frequently contractually
prohibited from leaving the site. That rules out every cloud vision API,
regardless of how good it is.

Which leaves: run a language-capable vision model on-site. That needs an
accelerator that handles both CNN and transformer inference in a thermal envelope
you can put on a bench, and a toolchain where an ASR model, a detector, and a VLM
are all already compiled for it. Modalix and Palette Neat is that combination —
SiMa ship a pre-compiled Whisper, pre-compiled VLMs, and a detector zoo, so the
hard part became application design rather than model porting.

Secondary, real, but not the headline: it keeps working with the network down,
and it sends kilobytes of verdicts instead of megabits of video.

## 9. Limitations — say these before a judge finds them

- A VLM is **not a metrology instrument**. It judges what a person could judge
  from a photograph. It cannot measure a tolerance.
- **Verdicts are not deterministic.** One observed run returned `pass` with a
  reason that said the item was *not* compliant. The model is now asked for a
  boolean and a verdict in the same call and any disagreement is reported as
  `unclear`. Measured 0 disagreements in 24 judgements — but the guard exists
  because the failure was real. Screening aid, not an authority.
- **One item at a time.** Several items in frame are judged as one scene.
- **Detector and VLM contend for one MLA.** That is why the loaded frame rate is
  published alongside the idle one.
- **Whisper can't be fine-tuned here** — it is consume-as-is, so domain jargon may
  mis-transcribe. The console always shows the transcript so it can be corrected.
- **The camera is the Mac's**, published over RTSP. Production would use a MIPI or
  USB camera on the board.

## 10. Roadmap

**Next:** close the physical loop — a GPIO or PLC signal off the DevKit driving a
real reject gate, so ACT actuates rather than logs.

**Then:** re-inspect `unclear` verdicts from a second angle before escalating;
per-item track IDs so multiple items are judged individually; cached image
embeddings (`VisionLanguageModel.encode()`) so a multi-rule standard costs one
encode instead of N.

**To productise:** a versioned on-device standards library, so an audit record
says which revision of which rule produced a verdict — and a measured
agreement rate against a human inspector on a fixed item set, published as a
number rather than asserted.
