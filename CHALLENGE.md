# Challenge Specification — Status: INCOMPLETE

## Verdict

**A formal judging specification is missing.** A track statement exists (below)
and is being used as product direction; a rubric does not.

The organizer material available on this machine contains setup instructions and a
practice lab, but **no problem statement, no judging criteria, no rubric, no
submission requirements, and no evaluation deadline.**

## Sources inspected

| Document | Location | Pages | Date |
|---|---|---|---|
| SiMa.ai Hackathon Game Day Guide v1.1 | `~/Downloads/SiMa.ai Hackathon GameDayGuide - v1.1.pdf` | 36 | Sep 10, 2026 |
| SiMa.ai Hackathon Prerequisites | `~/Downloads/SiMa.ai Hackathon PreReqs.pdf` | 9 | Sep 10, 2026 |
| SiMa.ai DevKit Flyer — AI Infra & Hackathon | `~/Downloads/SiMaAI DevKit Flyer - AI Infra & Hackathon.pdf` | 1 | — |

`/workspace` itself contained no organizer assets (only `test-sync.txt` and SDK
runtime config directories).

## What the organizer material actually states

### The brief, verbatim in substance

Game Day Guide, p.24, section *"Let the Games Begin"*:

> "Now that you have successfully built and run an application - experiment with the
> tools and devkit and build the best design you can in the allotted time for this
> hackathon. Good luck!"

That is the entire competitive brief. There is no further qualification.

### Stated context (Game Day Guide, p.3)

- Participants use **SiMa.ai Palette Neat Software Suite** with the **Palette Neat
  agentic development environment** to build **ML applications for the SiMa.ai
  Modalix MLSoC**.
- Organizer's capability claim for Modalix: *"high performance vision language
  models, large multimodal models, and various computer vision applications at
  under 10 W, with 50 TOPS performance."*
  **Treat as vendor marketing, not as a measurement of our application.**
- Each team is granted up to **$200 in OpenAI Codex credits**; a temporary OpenAI
  API key is issued per team. Beyond $200, teams use personal accounts.

### Target hardware (flyer + guide)

**Modalix MLSoC DevKit 3.0** (blue kit, removable lid):
- Modalix SoM + carrier board, power adapter, USB cable
- HDMI out
- **16 GB LPDDR5**
- **32 GB eMMC + 500 GB NVMe**
- RJ45 Ethernet (`end0`), UART over USB serial

### Reference workflow the organizers demonstrate (Practice Lab, pp.20–23)

The known-good baseline, driven through the agentic environment:

- **C++** object-detection project at `/workspace/labs/agent-neat`
- **YOLOv8n**, existing model if available, otherwise **SiMa.ai Model Zoo**
- Inference **on the attached Modalix DevKit using the MLA**
- Detection threshold **0.52**
- Input: an MP4 file served to the app as an **RTSP stream by Neat Insight**
  (`https://artifacts.sima-neat.com/assets/videos/720p16/video03.mp4`)
- Output: labeled bounding boxes + detection metadata rendered in **Neat Insight**
- Insight workflow: *Media Sources* tab → upload media; *Streaming* tab → assign
  RTSP stream; *Video Viewer* tab → observe; launch app from SDK terminal; Ctrl-C to stop
- Application config may need IP/ports obtained from `neat --json`

Stated motivating domains for this class of app: *drones, AMRs (autonomous mobile
robots), retail and security systems, especially with multiple live camera feeds.*

## Requirements that are explicitly absent

None of the following appear anywhere in the supplied material:

- [ ] Problem statement or theme
- [ ] Judging criteria / scoring rubric / weightings
- [ ] Submission format, artifacts, or deadline
- [ ] Demo length or presentation format
- [ ] Team size or eligibility rules
- [ ] Mandatory vs. optional feature list
- [ ] Required datasets or test inputs
- [ ] Any constraint beyond "build the best design you can"

**No requirements have been invented to fill these gaps.**

## Track statement (supplied by the user from the live Lablab dashboard)

> **"SiMa - Building Physical AI That sees, understands, and acts"**

This is treated as **important product direction**, and the solution is designed
explicitly around the three-stage loop:

| Stage | Meaning for this build |
|---|---|
| **SEE** | Real-world sensing/perception, with meaningful accelerated inference on Modalix |
| **UNDERSTAND** | Context, state, anomaly/event interpretation, multimodal fusion, semantic understanding |
| **ACT** | A useful automated response — alert, recommendation, control decision, workflow action, or simulated physical action |

All three stages must be obvious to a judge within seconds of seeing the demo.

### Scoring — explicitly NOT official

The public Lablab AI Infra Summit page does **not** expose a judging rubric or
scoring weights for this event. Lablab's *general* hackathon guidance discusses
four common dimensions:

- Presentation
- Business Value
- Application of Technology
- Originality

**These are not confirmed as this event's official criteria and must not be
represented as such.** They are used only as a sanity check on the design.

## Working interpretation (pending organizer confirmation)

Absent a rubric, the only defensible reading is that the work is judged on the
merits of an application that:

1. runs a meaningful ML workload **on the Modalix MLSoC using the MLA**, and
2. is built with the **Palette Neat** toolchain, and
3. actually works end to end and demonstrates well.

This is the assumption the architecture is being designed against. It will be
revised the moment an actual rubric is supplied.

## Open questions for the organizers

1. Is there a theme or problem domain, or is the field genuinely open?
2. What are the judging criteria and their weights?
3. What must be submitted, in what form, and by when?
4. How long is the demo/pitch slot?
5. Is internet access permitted during judging (relevant to any cloud fallback)?
