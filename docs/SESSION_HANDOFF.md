# Session handoff

Everything needed to resume Foreman after a context reset. Factual, current, no secrets.

## Architecture

Spoken-spec zero-shot inspection. Camera -> Modalix detector (continuous) ->
3 s rolling evidence window -> representative frames + ROI close-ups ->
one multi-image VLM judgement -> **detector-grounded** PASS / FAIL / UNCLEAR.

```
Mac camera --ffmpeg--> RTSP :8554/src2 --> DevKit
  [MODALIX] yolo_26n detector on the MLA, every frame
  [MODALIX] 3 s rolling buffer: detections x~45, JPEGs x~15
  [MODALIX] pick 3 representative frames (objects + sharpness + >=0.6 s apart)
  [MODALIX] + ROI close-ups of the smallest required object
  [MODALIX] ONE multi-image VLM call on the MLA
  [MAC]     temporal aggregation + grounding policy -> verdict
  [MAC]     audit trail, andon console
  [MODALIX] video + boxes also stream to Neat Insight
```

**The edge returns evidence, never a final verdict.** The rule that can override
the model lives in `foreman/host/policy.py` as pure functions, so all seven
regression cases are ordinary unit tests with no hardware.

Grounding policy, top down: window too short -> UNCLEAR; required object
presence <= 10% -> **FAIL** (model cannot override); prohibited >= 60% -> FAIL;
required below 60% -> UNCLEAR; otherwise the model judges the relationship.

### Speech input

Spoken standards are accepted in **English and Russian only**. The console's
header selector sends `language=en|ru|auto` with each recording;
`auto` decodes the clip both ways on the MLA and keeps the more likely reading,
so a third language can never come back. Nothing is loaded or restarted when the
selection changes - it is a per-request decoding parameter, and it applies to the
next recording. See docs/BENCHMARKS.md 4.1 and 4.2.

Both languages resolve to the same internal representation
(`host/standard_parser.py`), so a Russian rule is detector-grounded exactly like
its English twin. This is a safety property, not a convenience: a standard that
parses to no objects gives `policy.py` nothing to weigh and hands the verdict to
the model alone.

### Console

`foreman/frontend/` is three static files - `index.html`, `styles.css`, `app.js` -
served straight off disk by `host/app.py` with `no-store`. **No build step and no
framework**, deliberately: there is nothing to compile or install before a demo,
and one less thing that can fail in front of a judge. Two hash routes,
`#/inspection` and `#/camera`, so header nav, browser Back and Escape all behave.

## Hardware and addresses

| | |
|---|---|
| DevKit | **192.168.2.2**, `modalix`, aarch64, 16 cores, 5.9 GB RAM, 1.83 GB CMA |
| Mac on the DevKit subnet | **192.168.2.1** (bridge100, macOS Internet Sharing) |
| Link | USB-Ethernet, 1000baseT |
| Workspace sync | **rsync-over-SSH** to `/workspace-rsync` (host NFS export unavailable) |
| SSH | key auth from host, SDK container root, and container user `d` |

## Active models and why

| Role | Model | Why this one |
|---|---|---|
| Detector | `yolo_26n` (Model Zoo gen2, INT8, 640x640, COCO-80) | 6.3 ms, ~160 fps standalone; leaves MLA headroom for the VLM. Presence detection does not need a larger model. |
| VLM | **`Qwen3-VL-2B-Instruct-GPTQ-a16w4`** | The 4B on this board returns degenerate output (a run of `!`) for **every** image, via HTTP and in-process, at every size. The 2B is correct **and 4x faster**. Do not switch back. |
| ASR | `simaai/whisper-small-a16w8` | SiMa ships it compiled to MLA ELFs; RTF ~0.105. |

Board paths: models `/media/nvme/llima/models/`, detector
`/media/nvme/foreman/models/yolo_26n_mpk.tar.gz`, labels
`/media/nvme/foreman/coco.txt`, synced source `/workspace-rsync/foreman/`.

## Known-good startup / shutdown

```bash
# [HOST] everything, with preflight checks
cd ~/workspace/foreman
DEVKIT_IP=192.168.2.2 ./scripts/demo.sh        # camera + genai + edge + console
./scripts/stop.sh                              # full teardown

# [HOST] pieces, if bringing up by hand - ORDER MATTERS
./scripts/stream-camera.sh 0 src2              # camera FIRST; the edge needs the
                                               # RTSP source to exist at startup
ssh sima@192.168.2.2 'bash /workspace-rsync/foreman/scripts/run-genai.sh'
ssh sima@192.168.2.2 'bash /workspace-rsync/foreman/scripts/run-edge.sh'
./scripts/run-host.sh

# [HOST] health
curl -s http://192.168.2.2:8100/health          # edge + evidence buffer
curl -s http://127.0.0.1:8800/api/state         # console
curl -s http://127.0.0.1:8800/api/live          # raw detector output
ssh sima@192.168.2.2 'curl -s localhost:9998/v1/models'

# [SDK] push source to the board
docker exec -u d ghcr.io-sima-neat-sdk-v2.1.3.0 bash -lc \
  'source ~/.devkit-sync.rc; cd /workspace; dk sync foreman'
```

URLs: console `http://127.0.0.1:8800`, Insight viewer
`https://127.0.0.1:8081/static/viewer.html?src=0`, Insight console
`https://127.0.0.1:9900`.

## Git

- Repo root `/Users/d/workspace`; remote `origin` =
  `git@github.com:zsz13/sima-ai-ai-infra-summit-hackathon.git`
- Branch **`feat/foreman-temporal-grounding`**, pushed. `main` is local only and
  shares **no ancestor** with `origin/main` (a LICENSE-only initial commit).
- Key commits: `9943a5c` single-frame baseline, `b7c977d` temporal grounding,
  `dcca177` PID-file fix, `c9e5398` gitignore hardening.
- **Identity:** commits must use the configured `zsz13` identity. The five
  commits above were mistakenly authored as `Foreman <noreply@anthropic.com>`
  and one carries a Claude co-author trailer; left alone because they are pushed.
  Never add Claude/Anthropic attribution.

## Runtime processes

| Where | Process | Notes |
|---|---|---|
| Mac | `ffmpeg` avfoundation -> RTSP `src2` | needs `-framerate 30 -pixel_format uyvy422` |
| Mac | `host.app` console on :8800 | |
| SDK container | Neat Insight :9900, viewer :8081, RTSP :8554 | supervised |
| DevKit | GenAI server :9998 (`vlm` + `asr`) | PID `/tmp/foreman-genai.pid` |
| DevKit | Foreman edge :8100 | PID `/tmp/foreman-edge.pid` |

Exactly **one** of each DevKit process. Both launchers refuse to start a second.

## Verified benchmarks

| | Measured |
|---|---|
| Detector standalone | 6.23 ms median, ~160 fps (50 runs) |
| Detector pipelined | 778.9 inferences/s at 5.21 ms |
| Detector live | 15.1 fps idle / 15.2 under load (camera-bound) |
| VLM, 3 frames | 2133 ms median (n=31) |
| VLM, 3 frames + 2 ROI crops | ~2.5 s |
| Whisper ASR | 262 ms for 2.50 s audio, RTF ~0.105 |
| End-to-end verdict | 1875 ms, ~12 ms of it on the Mac |
| Insight correlation | 151/151 matched, 0 expired |
| Evidence buffer | 46 frames, 16 images, ~1.0 MB |
| Tests | 148 passing, ruff clean |

**Power is not measurable on this board** - no power rail is exposed; the
benchmark returns 0.00 W. Never quote SiMa's "under 10 W" as ours.

## Bugs already found and fixed - do not re-debug

1. **VLM hallucination.** Single-frame judgement let the model assert a phone
   that was never detected. Fixed by temporal evidence + detector grounding.
2. **4B VLM broken on this board** - degenerate `!` output for all images. Use the 2B.
3. **CMA exhaustion on model load.** MLA bulk load fails when `CmaFree` is low
   because page-cache CMA is not reclaimed. `run-genai.sh` drops caches first
   (407 MB -> 1630 MB).
4. **NV12 luma plane is height-padded** - 1280x720 arrives as 1,474,560 bytes
   (Y = 768 rows). A naive reshape gave a green band and a ghosted image.
5. **Duplicate DevKit processes.** The GenAI server runs as `python3 -`, so no
   `pkill` pattern on the script name matched; three copies took CmaFree to
   15 MB. Fixed with PID files.
6. **Evidence frames looked cropped.** Source JPEGs were always full 1280x720;
   the cells were 3.21:1 with `place-items:center` + `height:100%` +
   `overflow:hidden`, so the `<img>` overflowed and was clipped. Cells are now
   true 16:9.
7. **Overlay CSS was scoped to `.shot`**, so boxes never drew on the live view.
8. **Browser cached `index.html`**, hiding a CSS fix. The console is now served
   `no-store`.
9. **Model download silently incomplete** - `snapshot_download` dropped one 56 MB
   Whisper ELF. Verify against the HF manifest after downloading.
10. **`timeout` does not exist on macOS**; use `curl --max-time`.
11. **`pkill -f foreman_edge.py` over SSH kills its own session** - the pattern
    matches the remote command line. Use `[f]oreman_edge.py`.
12. **Whisper chose Bulgarian for English speech.** whisper-small's compiled
    `language_detect` stage picks from ~99 languages and got it badly wrong on
    real microphone audio. Foreman now never uses that stage: every decode pins
    an explicit ISO code, and Auto EN/RU decodes both ways and keeps the more
    likely reading. See docs/BENCHMARKS.md 4.1.
13. **Camera Check consumed the item.** The gate advanced while paused, latching
    whatever was in view as already inspected, so returning to the inspection
    view sat idle until that item left frame and came back - the console looked
    dead. The gate is now held armed while paused
    (`host/orchestrator.py`, regression test
    `test_camera_check_does_not_consume_the_item`).
14. **Camera Check could trap you.** It was a hidden panel, not a route, and the
    page was pinned to `height:100%` with no scroll, so on a short viewport the
    controls were clipped and unreachable. It is now a hash route (`#/camera`)
    with header nav, browser Back and Escape, and the page scrolls.
15. **DevKit processes vanish "for no reason".** Two causes, both real:
    (a) `scripts/demo.sh` installs `trap cleanup EXIT INT TERM`, and that cleanup
    kills the edge and the GenAI server on the board - so anything that ends a
    `demo.sh` run tears down the whole stack, which looks like a crash;
    (b) both programs trap SIGHUP and shut down, so `nohup ... &` inside an `ssh`
    heredoc is not enough on its own. When starting them by hand outside
    `demo.sh`, use `setsid ... </dev/null` so they lead their own session, and
    check with `ps -o sid=` that SID == PID.
16. **The edge does not retry RTSP.** If the camera stream is absent when the
    edge starts, gstreamer reports `resource_not_found` and the agent serves
    healthy-but-frameless forever. Start the camera first, then the edge.

## Must NOT be repeated

- **Do not run `sima-cli sdk setup`.** It *removed and recreated* the SDK
  container, reset container SSH keys and re-downloaded ~9 GB. Pairing is done.
- Do not reinstall the SDK, reset networking, or change the DevKit IP.
- Do not create a second SDK container.
- Do not force-push or rewrite pushed history.
- Do not switch to the 4B VLM.
- Do not raise the global detector threshold to hide duplicate boxes or to
  tidy up Camera Check; it shows what the detector actually reports.
- Do not let Whisper auto-detect freely - it returns Bulgarian. Pin the code.

## Fallback modes

| Fallback | How |
|---|---|
| Single-frame inspection (pre-temporal) | `FOREMAN_TEMPORAL=0` on the host; edge serves `/inspect_single` |
| No camera | `SKIP_CAMERA=1 SLOT=src1` and serve a clip from Insight |
| Nested-box suppression off | `--no-suppress-nested` on the edge |
| ROI close-ups off | `--no-roi` on the edge |
| Detector-only debugging | **Camera check** button in the console |

## Remaining tasks

1. Physical validation needing props: person holding a phone (PASS), phone on the
   desk not held (FAIL), bottle in view (PASS), bottle with/without a cap, and a
   short occlusion. Only no-prop cases are verified so far.
2. ~~The standard parser is English-only.~~ **Done.** `host/standard_parser.py`
   now carries an English and a Russian lexicon behind one engine; both produce
   the same `required`/`prohibited` sets. Verified on hardware: the Russian
   standard "Человек должен держать телефон." returned FAIL via `detector-absent`
   with person 46/47 and cell phone 0/47 - identical grounding to the English
   phrasing. Other languages still parse to nothing and say so in the console.
3. Detector throughput at saturation (a source faster than 15 fps) is unmeasured.
4. Accuracy on a real inspection task with a held-out item set.

## Exact next step

Ask the operator to hold a phone in view, then set the standard
"the person must be holding a phone" and confirm PASS; then put the phone on the
desk in view and confirm FAIL. Those two cases exercise detector grounding and
the VLM relationship judgement respectively, and are the last unverified part of
the temporal pipeline. Worth doing in both languages now that Russian grounds:
"Человек должен держать телефон."

Speech itself is verified with generated audio (`say -v Samantha` / `-v Milena`,
12/12 correct through the edge API) but **not yet with a live microphone**, which
is the one part of the speech path a person has to exercise.
