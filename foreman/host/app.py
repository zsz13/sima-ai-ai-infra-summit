"""Foreman host API + UI server. Runs on the Mac.

    uv run --with fastapi --with uvicorn --with httpx --with python-multipart \
        python -m host.app

Environment:
    FOREMAN_EDGE_URL   base URL of the Modalix edge agent (default http://192.168.2.2:8100)
    FOREMAN_PORT       port for this server (default 8800)
    FOREMAN_AUDIT_DIR  audit trail location (default ./audit)
    FOREMAN_INSIGHT_URL  Neat Insight viewer URL surfaced in the UI
    FOREMAN_TEMPORAL     1 (default) for temporal grounding, 0 for the single-frame fallback
    FOREMAN_WINDOW_S     rolling evidence window, seconds (default 3.0)
    FOREMAN_EVIDENCE_FRAMES  representative frames per inspection (default 3)
    FOREMAN_ABSENT_RATIO_MAX / FOREMAN_PRESENT_RATIO_MIN / FOREMAN_MIN_WINDOW_FRAMES
                         grounding thresholds; see host/policy.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .edge_client import EdgeClient, EdgeError
from .gate import GateConfig
from .orchestrator import Orchestrator
from .policy import GroundingConfig

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

EDGE_URL = os.environ.get("FOREMAN_EDGE_URL", "http://192.168.2.2:8100")
PORT = int(os.environ.get("FOREMAN_PORT", "8800"))
AUDIT_DIR = Path(os.environ.get("FOREMAN_AUDIT_DIR", ROOT / "audit"))
INSIGHT_URL = os.environ.get(
    "FOREMAN_INSIGHT_URL", "https://127.0.0.1:8081/static/viewer.html?src=0"
)

#: Temporal grounding is the default. FOREMAN_TEMPORAL=0 restores the previous
#: single-frame behaviour, kept as a fallback while the temporal path is proven.
TEMPORAL = os.environ.get("FOREMAN_TEMPORAL", "1") != "0"

orchestrator = Orchestrator(
    edge=EdgeClient(EDGE_URL),
    audit_dir=AUDIT_DIR,
    gate_config=GateConfig(
        min_confidence=float(os.environ.get("FOREMAN_MIN_CONF", "0.55")),
        stable_frames=int(os.environ.get("FOREMAN_STABLE_FRAMES", "6")),
        absent_frames=int(os.environ.get("FOREMAN_ABSENT_FRAMES", "10")),
    ),
    temporal=TEMPORAL,
    window_s=float(os.environ.get("FOREMAN_WINDOW_S", "3.0")),
    evidence_frames=int(os.environ.get("FOREMAN_EVIDENCE_FRAMES", "3")),
    grounding=GroundingConfig(
        absent_ratio_max=float(os.environ.get("FOREMAN_ABSENT_RATIO_MAX", "0.10")),
        present_ratio_min=float(os.environ.get("FOREMAN_PRESENT_RATIO_MIN", "0.60")),
        min_confidence=float(os.environ.get("FOREMAN_MIN_CONF", "0.55")),
        min_window_frames=int(os.environ.get("FOREMAN_MIN_WINDOW_FRAMES", "10")),
    ),
)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    await orchestrator.start()
    yield
    await orchestrator.stop()
    await orchestrator.edge.aclose()


app = FastAPI(title="Foreman", lifespan=lifespan)


@app.get("/api/config")
async def config() -> dict:
    return {
        "edge_url": EDGE_URL,
        "insight_url": INSIGHT_URL,
        "temporal": TEMPORAL,
        "window_s": orchestrator.window_s,
        "evidence_frames": orchestrator.evidence_frames,
    }


@app.get("/api/state")
async def state() -> dict:
    return orchestrator.snapshot()


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """Server-sent events carrying the full UI state on every change."""
    async def gen():
        q = orchestrator.subscribe()
        try:
            yield f"data: {json.dumps(orchestrator.snapshot())}\n\n"
            while True:
                try:
                    snap = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(snap)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            orchestrator.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/standard")
async def set_standard(body: dict) -> dict:
    text = (body or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "text must be a non-empty string")
    orchestrator.set_standard(text)
    return {"standard": orchestrator.standard}


@app.post("/api/standard/speak")
async def set_standard_from_speech(file: UploadFile) -> dict:
    """Upload spoken audio -> Whisper on the Modalix MLA -> new standard."""
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio upload")
    try:
        transcript = await orchestrator.edge.transcribe(audio, file.filename or "speech.wav")
    except EdgeError as exc:
        raise HTTPException(503, f"ASR unavailable: {exc}") from exc
    if not transcript.text:
        raise HTTPException(422, "no speech detected")
    orchestrator.set_standard(transcript.text)
    return {
        "standard": orchestrator.standard,
        "language": transcript.language,
        "no_speech_prob": transcript.no_speech_prob,
        "metrics": transcript.metrics,
    }


@app.get("/api/live")
async def live() -> dict:
    """Raw detector output for Camera Check. Read-only; triggers no inference."""
    return orchestrator.live()


@app.get("/api/live/frame.jpg")
async def live_frame() -> Response:
    """Proxy the DevKit's latest decoded frame, so the browser needs no route to it."""
    try:
        r = await orchestrator.edge._client.get("/frame.jpg", timeout=5.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"no live frame: {exc}") from exc
    return Response(content=r.content, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/mode")
async def set_mode(body: dict) -> dict:
    """Enter or leave Camera Check.

    Camera Check pauses inspections only. The detector keeps running on the
    DevKit, the live view keeps updating, and the current standard is left
    untouched - it is a debugging view, not a state change.
    """
    camera = bool((body or {}).get("camera"))
    orchestrator.paused = camera
    orchestrator._publish()
    return {"camera": camera, "paused": orchestrator.paused,
            "standard": orchestrator.standard}


@app.post("/api/session/clear")
async def clear_session() -> dict:
    orchestrator.clear_session()
    return {"ok": True}


@app.get("/api/evidence/{name}")
async def evidence(name: str) -> FileResponse:
    # Resolve and confine to the evidence directory: a crafted name must not
    # be able to read arbitrary files off the Mac.
    base = (AUDIT_DIR / "evidence").resolve()
    path = (base / name).resolve()
    if not path.is_file() or base not in path.parents:
        raise HTTPException(404, "no such evidence frame")
    return FileResponse(path, media_type="image/jpeg")


class NoCacheStatic(StaticFiles):
    """Serve the console with no-store.

    A cached index.html silently hid a CSS fix during development; the same class
    of problem bites Neat Insight's viewer. The console is a handful of KB, so
    revalidating every load costs nothing and removes a whole category of
    "why is my change not showing" during a demo.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


if FRONTEND.is_dir():
    app.mount("/", NoCacheStatic(directory=FRONTEND, html=True), name="frontend")


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
