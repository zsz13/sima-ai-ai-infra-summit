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
    FOREMAN_EVIDENCE_FRAMES  representative frames per inspection (default 6)
    FOREMAN_CAPTURE_S    manual "Inspect now" capture length, seconds (default 3.0)
    FOREMAN_ABSENT_RATIO_MAX / FOREMAN_PRESENT_RATIO_MIN / FOREMAN_MIN_WINDOW_FRAMES
                         grounding thresholds; see host/policy.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .audit import (
    DEFAULT_LIMIT,
    build_package,
    parse_ids,
    read_records,
    select_by_ids,
    summarise,
    to_csv,
    to_json,
)
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
    evidence_frames=int(os.environ.get("FOREMAN_EVIDENCE_FRAMES", "6")),
    capture_s=float(os.environ.get("FOREMAN_CAPTURE_S", "3.0")),
    grounding=GroundingConfig(
        absent_ratio_max=float(os.environ.get("FOREMAN_ABSENT_RATIO_MAX", "0.10")),
        present_ratio_min=float(os.environ.get("FOREMAN_PRESENT_RATIO_MIN", "0.60")),
        min_confidence=float(os.environ.get("FOREMAN_MIN_CONF", "0.55")),
        min_window_frames=int(os.environ.get("FOREMAN_MIN_WINDOW_FRAMES", "10")),
        # The policy reasons about the same number of time segments as the
        # operator sees evidence images from, so "6 of 6 segments" in the reason
        # lines up with the six frames on screen.
        coverage_buckets=int(os.environ.get("FOREMAN_EVIDENCE_FRAMES", "6")),
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
    """Apply one or two rules.

    `{"text": "..."}` is the original single-rule form and still behaves exactly
    as it did. `{"rules": [...]}` applies up to two, each kept verbatim - they
    are never joined into one sentence, because two rules concatenated would let
    one rule's negation scope leak into the other.
    """
    body = body or {}
    raw = body.get("rules")
    if raw is None:
        text = body.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(400, "text must be a non-empty string")
        raw = [text]
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise HTTPException(400, "rules must be a list of strings")
    cleaned = [t for t in (t.strip() for t in raw) if t]
    if not cleaned:
        raise HTTPException(400, "at least one non-empty rule is required")
    try:
        orchestrator.set_rules(cleaned)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"standard": orchestrator.standard,
            "parsed": orchestrator.parsed_standard(),
            "rules": list(orchestrator.rules),
            "parsed_rules": orchestrator.parsed_rules()}


@app.post("/api/standard/speak")
async def set_standard_from_speech(file: UploadFile, language: str = "auto") -> dict:
    """Upload spoken audio -> Whisper on the Modalix MLA -> new standard.

    `language` is "en", "ru" or "auto" (English or Russian only). A transcript
    the edge rejected as unclear is returned with accepted=false and the standard
    in force is left untouched, so a garbled rule can never silently take over an
    inspection.
    """
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio upload")
    if language not in ("en", "ru", "auto"):
        raise HTTPException(400, "language must be en, ru or auto")
    # Recognition takes about a second. If the operator clears the session while
    # it runs, the transcript must not be applied on arrival.
    epoch = orchestrator.session_epoch
    try:
        transcript = await orchestrator.edge.transcribe(
            audio, file.filename or "speech.wav", language)
    except EdgeError as exc:
        raise HTTPException(503, f"ASR unavailable: {exc}") from exc

    result = {
        "accepted": transcript.accepted,
        "transcript": transcript.text,
        "language": transcript.language,
        "mode": transcript.mode,
        "avg_logprob": transcript.avg_logprob,
        "no_speech_prob": transcript.no_speech_prob,
        "metrics": transcript.metrics,
        "standard": orchestrator.standard,
    }
    if not transcript.accepted:
        result["reason"] = transcript.reject_reason or "the speech was not clear enough"
        return result
    if not orchestrator.set_standard(transcript.text, language=transcript.language,
                                     epoch=epoch):
        result["accepted"] = False
        result["reason"] = ("the session was cleared while this was being "
                            "transcribed, so it was not applied")
        result["standard"] = orchestrator.standard
        return result
    result["standard"] = orchestrator.standard
    result["parsed"] = orchestrator.parsed_standard()
    return result


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


@app.get("/api/live/debug")
async def live_debug(threshold: float = 0.20) -> dict:
    """Low-confidence detector output for Camera Check's debug view.

    Read-only and inert: the edge re-runs the detector on the current frame and
    returns the result directly. None of it enters the evidence window, the gate
    or host/policy.py, so lowering the debug threshold cannot change a verdict.
    Only backends that implement /detect_debug answer; the others say so.
    """
    try:
        r = await orchestrator.edge._client.get(
            "/detect_debug", params={"threshold": threshold}, timeout=10.0)
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"debug detection unavailable: {exc}") from exc
    if r.status_code == 404:
        raise HTTPException(501, "this backend does not provide debug detection")
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()


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


@app.post("/api/inspection_mode")
async def set_inspection_mode(body: dict) -> dict:
    """Choose whether inspections start by hand or automatically.

    Switching does not touch the standard: how an inspection starts and what it
    checks are separate decisions.
    """
    mode = str((body or {}).get("mode", "")).lower()
    try:
        orchestrator.set_inspection_mode(mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"inspection_mode": orchestrator.inspection_mode,
            "standard": orchestrator.standard}


@app.post("/api/inspect_now")
async def inspect_now(body: dict | None = None) -> dict:
    """Manual inspection: capture from this moment, then judge.

    Unlike the automatic path, the evidence window starts when this is called, so
    it judges what the operator just did rather than whatever was already in the
    rolling buffer. Everything after capture is the same code: same frame
    selection, same detector grounding, same policy.
    """
    capture_s = float((body or {}).get("capture_s") or 0.0) or None
    if capture_s is not None and not 0.5 <= capture_s <= 15.0:
        raise HTTPException(400, "capture_s must be between 0.5 and 15 seconds")
    result = await orchestrator.inspect_now(capture_s)
    if not result.get("started"):
        raise HTTPException(409, result.get("reason", "could not start an inspection"))
    return result


@app.post("/api/session/clear")
async def clear_session() -> dict:
    orchestrator.clear_session()
    return {"ok": True}


#: The audit trail is the only source of truth for history and export. Both
#: endpoints are read-only and run the file read in a worker thread, so a long
#: trail can never stall the event loop that is driving inspections.
AUDIT_FILE = AUDIT_DIR / "inspections.jsonl"
EVIDENCE_DIR = AUDIT_DIR / "evidence"


async def _records_for_export(limit: int, ids: str | None) -> list[dict]:
    """Records for an export, optionally narrowed to specific inspection ids.

    `ids` is matched against what the audit trail already contains, so an id the
    trail does not hold selects nothing. A caller cannot name an evidence file
    directly - only an inspection - which is what keeps these endpoints from
    turning into an arbitrary file reader.
    """
    records = await asyncio.to_thread(read_records, AUDIT_FILE, limit)
    return select_by_ids(records, parse_ids(ids))


@app.get("/api/history")
async def history(limit: int = DEFAULT_LIMIT) -> dict:
    """Recent inspections, newest first, straight off the audit trail."""
    records = await asyncio.to_thread(read_records, AUDIT_FILE, limit)
    return {"records": [summarise(r) for r in records], "count": len(records)}


@app.get("/api/export.csv")
async def export_csv(limit: int = 2000, ids: str | None = None) -> Response:
    """Download the recorded inspections as CSV. Nothing is modified or cleared."""
    records = await _records_for_export(limit, ids)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "-selected" if ids else ""
    return Response(
        content=await asyncio.to_thread(to_csv, records, EVIDENCE_DIR),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="foreman-inspections{tag}-{stamp}.csv"',
                 "Cache-Control": "no-store"},
    )


@app.get("/api/export.json")
async def export_json(limit: int = 2000, ids: str | None = None) -> Response:
    """The same records as CSV, with evidence as structured references.

    Evidence is never embedded as base64: it would inflate the payload by a
    third and make the file unreadable for no gain. Use the .zip export to get
    the images themselves.
    """
    records = await _records_for_export(limit, ids)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "-selected" if ids else ""
    return Response(
        content=await asyncio.to_thread(to_json, records, EVIDENCE_DIR),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="foreman-inspections{tag}-{stamp}.json"',
                 "Cache-Control": "no-store"},
    )


@app.get("/api/export.zip")
async def export_zip(limit: int = 2000, ids: str | None = None) -> Response:
    """A self-contained package: report.csv, report.json and the evidence JPEGs.

    The images are copied byte for byte from the audit trail's own files. The
    trail is opened read-only and is never modified by an export.
    """
    records = await _records_for_export(limit, ids)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "-selected" if ids else ""
    blob = await asyncio.to_thread(build_package, records, EVIDENCE_DIR)
    return Response(
        content=blob, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="foreman-report{tag}-{stamp}.zip"',
                 "Cache-Control": "no-store"},
    )


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
