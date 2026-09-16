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
    "bbox": [0.30, 0.30, 0.60, 0.60],
    "verdict": "pass",
    "reason": "Synthetic verdict from the test harness. Not a real inference.",
    "inspect_delay": 0.05,
    "inspect_calls": 0,
    "fps": 30.0,
}


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "harness": True, "models": {"detector": "fake", "vlm": "fake", "asr": "fake"}}


@app.get("/events")
async def events() -> StreamingResponse:
    async def gen():
        frame_id = 0
        while True:
            frame_id += 1
            dets = []
            if STATE["present"]:
                dets = [{
                    "label": "box",
                    "confidence": 0.93,
                    "bbox": list(STATE["bbox"]),
                    "track_id": 1,
                }]
            payload = {"frame_id": frame_id, "ts": time.time(), "detections": dets}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0 / STATE["fps"])

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/inspect")
async def inspect(body: dict) -> dict:
    STATE["inspect_calls"] += 1
    await asyncio.sleep(STATE["inspect_delay"])
    return {
        "verdict": STATE["verdict"],
        "reason": STATE["reason"],
        "evidence_jpeg_b64": TINY_JPEG_B64,
        "metrics": {"inference_ms": 1234.5, "ttft_ms": 210.0, "tokens_per_s": 18.4},
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile) -> dict:
    await file.read()
    return {
        "text": "every box must have a label facing up and the lid closed",
        "language": "en",
        "no_speech_prob": 0.01,
        "metrics": {"inference_ms": 380.0},
    }


# --- harness control, used by tests -----------------------------------

@app.post("/_harness/state")
async def set_state(body: dict) -> dict:
    STATE.update(body)
    return dict(STATE)


@app.get("/_harness/state")
async def get_state() -> dict:
    return dict(STATE)
