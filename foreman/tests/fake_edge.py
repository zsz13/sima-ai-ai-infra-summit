"""TEST HARNESS ONLY — a stand-in for the Modalix edge agent.

=====================================================================
THIS IS NOT PART OF THE DEMO AND MUST NEVER BE RUN DURING ONE.
It produces synthetic detections and canned verdicts so the Mac-side
orchestrator and UI can be developed and tested without a DevKit.
Every number it returns is fabricated. `scripts/demo.sh` does not and
must not start it; the real edge agent refuses to report fake data.
=====================================================================

Implements exactly the wire contract documented in host/edge_client.py.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse


# A visible placeholder JPEG, generated once at import, so evidence handling
# can be exercised and dev screenshots are legible. Synthetic, like everything here.
def _placeholder_jpeg() -> str:
    """A 480x270 grey frame with a lighter rectangle, as a minimal baseline JPEG."""
    import io
    w, h = 480, 270
    # Minimal greyscale JPEG: standard tables + a flat DC-only scan is fiddly to
    # hand-roll, so fall back to a PPM-like solid via a tiny pure-python encoder
    # is overkill. Use a base64 of a pre-made 8x8 grey JPEG scaled by the browser
    # only if Pillow is unavailable.
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return TINY_FALLBACK_B64
    img = Image.new("RGB", (w, h), (26, 32, 37))
    d = ImageDraw.Draw(img)
    d.rectangle([140, 70, 340, 200], fill=(58, 70, 80), outline=(120, 140, 150), width=2)
    d.rectangle([170, 100, 240, 125], fill=(150, 160, 168))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


TINY_FALLBACK_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
    "AAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)

TINY_JPEG_B64 = _placeholder_jpeg()

app = FastAPI(title="fake-edge (TEST HARNESS)")

STATE = {
    "present": False,      # is an item in frame?
    "label": "box",        # what the detector reports seeing
    #: Several classes in the same frame, for multi-rule tests. When None the
    #: harness emits the single STATE["label"].
    "labels": None,
    "bbox": [0.30, 0.30, 0.60, 0.60],
    "verdict": "pass",
    "reason": "Synthetic verdict from the test harness. Not a real inference.",
    "inspect_delay": 0.05,
    "inspect_calls": 0,
    #: One entry per rule when a request carries `rules`. Each is
    #: (verdict, reason, observed_relationship). Falls back to the single-rule
    #: STATE["verdict"]/["reason"] when unset.
    "rule_results": None,
    #: How long the harness pretends a manual capture takes. Short enough to keep
    #: the suite fast, long enough that the orchestrator's ring really does hold
    #: frames for the reported window.
    "capture_sim_s": 0.8,
    "fps": 30.0,
    "window_frames": 45,
    "per_frame": [True, True, True],
    "vlm_evidence": ["a box is visible"],
    "vlm_missing": [],
    "detector_summary": "- box: detected in 45/45 frames (100%).",
}


def _detections() -> list[dict]:
    """Detections for one frame: every configured class, or the single label."""
    if not STATE["present"]:
        return []
    labels = STATE["labels"] or [STATE["label"]]
    return [{"label": lb, "confidence": 0.93, "bbox": list(STATE["bbox"]),
             "track_id": i + 1} for i, lb in enumerate(labels)]


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "harness": True, "backend": "fake",
            "models": {"detector": "fake", "vlm": "fake", "asr": "fake"}}


@app.get("/events")
async def events() -> StreamingResponse:
    async def gen():
        frame_id = 0
        while True:
            frame_id += 1
            dets = _detections()
            payload = {"frame_id": frame_id, "ts": time.time(), "detections": dets}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0 / STATE["fps"])

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/inspect")
async def inspect(body: dict) -> dict:
    """Temporal window contract. Mirrors the real edge's response shape."""
    STATE["inspect_calls"] += 1
    await asyncio.sleep(STATE["inspect_delay"])
    now = time.time()
    window_s = float(body.get("window_s") or 3.0)
    want = int(body.get("num_frames") or 3)
    total = int(STATE["window_frames"])
    # A manual capture judges frames collected AFTER the request arrives, so a
    # real edge blocks for the whole capture and its window is the interval that
    # just elapsed. The harness blocks too - briefly - because a window that has
    # not happened yet would contain no frames and would hide exactly the bug the
    # preparation delay exists to prevent. Only the auto path looks backwards.
    manual = float(body.get("capture_s") or 0.0) > 0
    if manual:
        arrival = now
        await asyncio.sleep(STATE["capture_sim_s"])
        now = time.time()
        start = arrival
        window_s = now - arrival
    else:
        start = now - window_s

    selected = [{
        "frame_id": 1000 + i,
        "ts": start + (i + 0.5) * window_s / max(want, 1),
        "rel_ts": round((i + 0.5) * window_s / max(want, 1), 3),
        "sharpness": 120.0 - i,
        "detections": _detections(),
        "jpeg_b64": TINY_JPEG_B64,
    } for i in range(want)]

    # A manual request arrives only after the host has finished preparing, so
    # the capture it reports starts now - never at the button press. Mirrors what
    # both real edges return, which is what lets a test measure the delay.
    capture_s = float(body.get("capture_s") or 0.0)
    capture = ({"mode": "manual", "requested_s": capture_s,
                "start_ts": start, "end_ts": now,
                "detector_frames": total}
               if capture_s > 0 else {"mode": "auto"})

    def judgement(verdict, reason, observed=None):
        return {
            "verdict": verdict,
            "reason": reason,
            "evidence": list(STATE["vlm_evidence"]),
            "missing_evidence": list(STATE["vlm_missing"]),
            "observed_relationship": observed,
            "per_frame": [list(STATE["per_frame"])[i % len(STATE["per_frame"])]
                          for i in range(want)] if STATE["per_frame"] else [None] * want,
        }

    # One judgement per rule, from the SAME selected frames. The harness never
    # re-captures: `inspect_calls` counts requests, so a test can prove that two
    # rules cost one capture rather than two.
    rules = body.get("rules") or []
    if rules:
        configured = STATE["rule_results"] or []
        vlms = []
        for i, _rule in enumerate(rules):
            if i < len(configured):
                v, r, *rest = configured[i]
                vlms.append(judgement(v, r, rest[0] if rest else None))
            else:
                vlms.append(judgement(STATE["verdict"], STATE["reason"]))
    else:
        vlms = [judgement(STATE["verdict"], STATE["reason"])]

    return {
        "capture": capture,
        "window": {"start_ts": start, "end_ts": start + window_s, "duration_s": window_s,
                   "total_frames": total, "image_frames": max(want, total // 3)},
        "detector_summary": STATE["detector_summary"],
        "selected": selected,
        "vlms": vlms,
        "vlm": {
            "verdict": STATE["verdict"],
            "reason": STATE["reason"],
            "evidence": list(STATE["vlm_evidence"]),
            "missing_evidence": list(STATE["vlm_missing"]),
            # One entry per image sent, like a real model: pad by repeating the
            # configured pattern rather than returning a short list, which would
            # understate how many frames agreed.
            "per_frame": [list(STATE["per_frame"])[i % len(STATE["per_frame"])]
                          for i in range(want)] if STATE["per_frame"] else [None] * want,
        },
        "metrics": {"inference_ms": 3050.0, "ttft_ms": 260.0,
                    "vlm_calls": len(vlms), "frames_sent": want, "selection_ms": 1.2,
                    "temporal_frames": want},
    }


@app.post("/inspect_single")
async def inspect_single(body: dict) -> dict:
    """Pre-temporal single-frame contract, kept for the fallback path."""
    STATE["inspect_calls"] += 1
    await asyncio.sleep(STATE["inspect_delay"])
    return {
        "verdict": STATE["verdict"],
        "reason": STATE["reason"],
        "evidence_jpeg_b64": TINY_JPEG_B64,
        "metrics": {"inference_ms": 1234.5, "ttft_ms": 210.0, "tokens_per_s": 18.4},
    }


#: What the harness "hears" for each speech language, so the bilingual path can
#: be exercised without a microphone or a DevKit.
FAKE_SPEECH = {
    "en": "every box must have a label facing up and the lid closed",
    "ru": "человек должен держать телефон",
}


@app.post("/transcribe")
async def transcribe(file: UploadFile, language: str = "auto") -> dict:
    await file.read()
    if STATE.get("speech_unclear"):
        return {
            "text": "", "language": "en", "mode": language, "accepted": False,
            "reject_reason": "the recording does not appear to contain speech",
            "avg_logprob": -0.53, "no_speech_prob": 0.91,
            "metrics": {"inference_ms": 240.0, "asr_calls": 1},
        }
    decoded = "ru" if language == "ru" else "en"
    calls = 2 if language == "auto" else 1
    return {
        "text": FAKE_SPEECH[decoded],
        "language": decoded,
        "mode": language,
        "accepted": True,
        "reject_reason": "",
        "avg_logprob": -0.06,
        "no_speech_prob": 0.01,
        "metrics": {"inference_ms": 380.0 * calls, "asr_calls": float(calls)},
    }


# --- harness control, used by tests -----------------------------------

@app.post("/_harness/state")
async def set_state(body: dict) -> dict:
    STATE.update(body)
    return dict(STATE)


@app.get("/_harness/state")
async def get_state() -> dict:
    return dict(STATE)
