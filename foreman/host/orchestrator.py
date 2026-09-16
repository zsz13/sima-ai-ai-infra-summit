"""Foreman orchestrator: the [MAC] half of the system.

Owns everything that is *not* inference:
  - the current inspection standard (stated by voice, transcribed on the DevKit)
  - the presence/stability gate that decides when to spend a VLM inference
  - session state, pass/fail policy, counters
  - a durable audit trail with evidence frames

It never fabricates a verdict. If the edge is unreachable the UI says so.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .edge_client import EdgeClient, EdgeError, Frame, Verdict
from .gate import Gate, GateConfig
from .policy import GroundingConfig, VlmJudgement, build_evidence, decide
from .standard_parser import parse_standard

MAX_RECENT = 40
RECONNECT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)


@dataclass
class Inspection:
    """One inspection result.

    The first nine fields are unchanged from the single-frame version so existing
    audit files and readers stay valid; everything temporal is additive.
    """

    id: str
    ts: float
    verdict: str
    reason: str
    standard: str
    trigger_label: str | None
    trigger_confidence: float | None
    metrics: dict[str, float]
    evidence_path: str | None = None          # first frame, for backward compatibility

    # --- temporal grounding ---
    decided_by: str = ""
    evidence_paths: list[str] = field(default_factory=list)
    window: dict = field(default_factory=dict)
    required_objects: list[dict] = field(default_factory=list)
    prohibited_objects: list[dict] = field(default_factory=list)
    vlm: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)

    def public(self) -> dict:
        return asdict(self)


@dataclass
class Counters:
    passed: int = 0
    failed: int = 0
    unclear: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.unclear


@dataclass
class Orchestrator:
    edge: EdgeClient
    audit_dir: Path
    gate_config: GateConfig = field(default_factory=GateConfig)

    standard: str = ""
    connected: bool = False
    last_error: str | None = None
    frames_seen: int = 0
    warming_up: bool = False
    #: Camera Check mode. The detector keeps running and the live view keeps
    #: updating, but no inspection is triggered: no VLM call, no PASS/FAIL record.
    paused: bool = False
    detections_last_frame: int = 0
    fps: float = 0.0
    inspecting: bool = False

    counters: Counters = field(default_factory=Counters)
    recent: deque[Inspection] = field(default_factory=lambda: deque(maxlen=MAX_RECENT))

    temporal: bool = True
    window_s: float = 3.0
    evidence_frames: int = 3
    grounding: GroundingConfig = field(default_factory=GroundingConfig)

    _gate: Gate = field(init=False)
    #: (ts, detections) for every frame received, trimmed to the window. This is
    #: the Mac's own copy of the temporal evidence, so aggregation and policy are
    #: testable without hardware.
    _ring: deque = field(default_factory=lambda: deque(maxlen=600), init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, init=False)
    _fps_window: deque[float] = field(default_factory=lambda: deque(maxlen=30), init=False)

    def __post_init__(self) -> None:
        self._gate = Gate(self.gate_config)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        (self.audit_dir / "evidence").mkdir(exist_ok=True)

    # --- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # --- UI subscription ------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _publish(self) -> None:
        snapshot = self.snapshot()
        for q in list(self._subscribers):
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                # A slow browser must not stall the pipeline. Drop the oldest
                # update and keep the newest, which is the one that matters.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(snapshot)

    def snapshot(self) -> dict:
        return {
            "connected": self.connected,
            "last_error": self.last_error,
            "standard": self.standard,
            "gate_state": self._gate.state.value,
            "gate_progress": round(self._gate.progress, 3),
            "inspecting": self.inspecting,
            "warming_up": self.warming_up,
            "paused": self.paused,
            "window_frames": sum(1 for ts, _ in self._ring
                                 if self._ring and ts >= self._ring[-1][0] - self.window_s),
            "frames_seen": self.frames_seen,
            "detections_last_frame": self.detections_last_frame,
            "fps": round(self.fps, 1),
            "temporal": self.temporal,
            "window_s": self.window_s,
            "counters": asdict(self.counters) | {"total": self.counters.total},
            "recent": [i.public() for i in reversed(self.recent)],
        }

    # --- control --------------------------------------------------------

    def set_standard(self, text: str) -> None:
        """Adopt a new inspection standard and re-arm the gate."""
        self.standard = text.strip()
        self._gate.reset()
        self._publish()

    def clear_session(self) -> None:
        self.counters = Counters()
        self.recent.clear()
        self._gate.reset()
        self._publish()

    def live(self) -> dict:
        """Raw detector output for Camera Check. Never triggers inference."""
        ts, dets = (self._ring[-1] if self._ring else (0.0, []))
        counts: dict[str, int] = {}
        best: dict[str, float] = {}
        for d in dets:
            label = d["label"]
            counts[label] = counts.get(label, 0) + 1
            best[label] = max(best.get(label, 0.0), float(d.get("confidence", 0.0)))
        return {
            "ts": ts,
            "age_s": round(time.time() - ts, 2) if ts else None,
            "fps": round(self.fps, 1),
            "connected": self.connected,
            "paused": self.paused,
            "min_confidence": self.grounding.min_confidence,
            "gate_min_confidence": self.gate_config.min_confidence,
            "detections": dets,
            "classes": sorted(
                ({"label": k, "count": counts[k], "confidence": round(best[k], 3)}
                 for k in counts),
                key=lambda c: (-c["confidence"], c["label"])),
            "window_frames": sum(1 for t, _ in self._ring
                                 if self._ring and t >= self._ring[-1][0] - self.window_s),
        }

    def reset(self) -> None:
        """Drop buffered evidence too, e.g. when the camera source changes."""
        self._ring.clear()
        self.clear_session()

    # --- main loop ------------------------------------------------------

    async def _run(self) -> None:
        attempt = 0
        while True:
            try:
                await self.edge.health()
                self.connected = True
                self.last_error = None
                attempt = 0
                self._publish()
                async for frame in self.edge.events():
                    await self._on_frame(frame)
            except asyncio.CancelledError:
                raise
            except EdgeError as exc:
                self.connected = False
                self.last_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - surface, never crash the loop
                self.connected = False
                self.last_error = f"{type(exc).__name__}: {exc}"
            else:
                self.connected = False
                self.last_error = "edge event stream ended"

            self._publish()
            delay = RECONNECT_BACKOFF[min(attempt, len(RECONNECT_BACKOFF) - 1)]
            attempt += 1
            await asyncio.sleep(delay)

    def _window_ready(self) -> bool:
        """Enough buffered frames in the last `window_s` to judge anything."""
        if not self._ring:
            return False
        newest = self._ring[-1][0]
        recent = sum(1 for ts, _ in self._ring if ts >= newest - self.window_s)
        return recent >= self.grounding.min_window_frames

    def _detections_between(self, start_ts: float, end_ts: float) -> list[list[dict]]:
        """This Mac's detections for exactly the window the edge reported, so both
        sides aggregate the same frames."""
        pad = 0.05  # tolerate clock skew between the two hosts
        return [dets for ts, dets in list(self._ring)
                if start_ts - pad <= ts <= end_ts + pad]

    async def _on_frame(self, frame: Frame) -> None:
        self.frames_seen += 1
        self._ring.append((frame.ts, [
            {"label": d.label, "confidence": d.confidence, "bbox": list(d.bbox)}
            for d in frame.detections]))
        self.detections_last_frame = len(frame.detections)
        self._fps_window.append(time.monotonic())
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            self.fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0

        should_inspect = self._gate.update(frame.detections)

        if should_inspect and self.standard and not self.inspecting and not self.paused:
            if self.temporal and not self._window_ready():
                # The gate settled before the evidence window had filled - which
                # happens on the first item after connecting or after a new
                # standard. Re-arm instead of consuming the item, so it is judged
                # on a full window a moment later rather than refused for lack of
                # evidence.
                self._gate.reset()
                self.warming_up = True
                self._publish()
                return
            self.warming_up = False
            asyncio.create_task(self._inspect())
        else:
            self.warming_up = self.temporal and not self._window_ready()
            self._publish()

    async def _inspect(self) -> None:
        """Judge one item. Temporal by default; single-frame only as a fallback."""
        self.inspecting = True
        self._publish()
        trigger = self._gate.last_trigger
        started = time.monotonic()
        try:
            if self.temporal:
                record = await self._inspect_temporal(trigger, started)
            else:
                record = await self._inspect_single_frame(trigger, started)
        except EdgeError as exc:
            self.counters.errors += 1
            self.last_error = str(exc)
            self.inspecting = False
            self._publish()
            return

        if record.verdict == "pass":
            self.counters.passed += 1
        elif record.verdict == "fail":
            self.counters.failed += 1
        else:
            self.counters.unclear += 1

        self.recent.append(record)
        self._append_audit(record)
        self.inspecting = False
        self._publish()

    async def _inspect_temporal(self, trigger, started: float) -> Inspection:
        """Detector evidence across the window decides what the VLM is allowed to say."""
        parsed = parse_standard(self.standard)
        evidence = await self.edge.inspect_window(
            self.standard, list(parsed.required), list(parsed.prohibited),
            self.window_s, self.evidence_frames)

        frames = self._detections_between(evidence.start_ts, evidence.end_ts)
        if not frames:
            # The edge counted frames we never received; fall back to its count so
            # the window-length check still means something.
            frames = [[]] * evidence.total_frames

        detector = build_evidence(
            frames, [*parsed.required, *parsed.prohibited], self.grounding)

        vlm_raw = evidence.vlm or {}
        judgement = VlmJudgement(
            verdict=str(vlm_raw.get("verdict", "unclear")).lower(),
            reason=str(vlm_raw.get("reason", "")),
            evidence=tuple(vlm_raw.get("evidence") or ()),
            missing_evidence=tuple(vlm_raw.get("missing_evidence") or ()),
            per_frame=tuple(vlm_raw.get("per_frame") or ()),
        )
        decision = decide(parsed, detector, judgement, self.grounding)

        paths = [p for p in (self._save_evidence_bytes(f.jpeg) for f in evidence.selected) if p]
        elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics = dict(evidence.metrics)
        metrics["end_to_end_ms"] = round(elapsed_ms, 1)
        metrics["host_ms"] = round(elapsed_ms - metrics.get("inference_ms", 0.0), 1)

        return Inspection(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            verdict=decision.verdict,
            reason=decision.reason,
            standard=self.standard,
            trigger_label=trigger.label if trigger else None,
            trigger_confidence=trigger.confidence if trigger else None,
            metrics=metrics,
            evidence_path=paths[0] if paths else None,
            decided_by=decision.decided_by,
            evidence_paths=paths,
            window={
                "start_ts": evidence.start_ts,
                "end_ts": evidence.end_ts,
                "duration_s": evidence.duration_s,
                "total_frames": evidence.total_frames,
                "image_frames": evidence.image_frames,
                "frames_aggregated": len(frames),
                "detector_summary": evidence.detector_summary,
            },
            required_objects=decision.required,
            prohibited_objects=decision.prohibited,
            vlm=decision.vlm,
            notes=decision.notes,
            frames=[{
                "path": path,
                "frame_id": f.frame_id,
                "rel_ts": f.rel_ts,
                "sharpness": f.sharpness,
                "detections": [{"label": d.label, "confidence": round(d.confidence, 3),
                                "bbox": [round(v, 4) for v in d.bbox]}
                               for d in f.detections],
            } for f, path in zip(evidence.selected, paths, strict=False)],
        )

    async def _inspect_single_frame(self, trigger, started: float) -> Inspection:
        """Pre-temporal path: one frame, no detector grounding. FOREMAN_TEMPORAL=0."""
        verdict: Verdict = await self.edge.inspect(self.standard)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics = dict(verdict.metrics)
        metrics["end_to_end_ms"] = round(elapsed_ms, 1)
        path = self._save_evidence(verdict.evidence_jpeg_b64)
        return Inspection(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            verdict=verdict.verdict,
            reason=verdict.reason,
            standard=self.standard,
            trigger_label=trigger.label if trigger else None,
            trigger_confidence=trigger.confidence if trigger else None,
            metrics=metrics,
            evidence_path=path,
            decided_by="single-frame-fallback",
            evidence_paths=[path] if path else [],
            notes=["Single-frame fallback: no temporal evidence, no detector grounding."],
        )

    # --- persistence ----------------------------------------------------

    def _save_evidence(self, b64: str | None) -> str | None:
        if not b64:
            return None
        try:
            return self._save_evidence_bytes(base64.b64decode(b64, validate=True))
        except (ValueError, TypeError):
            return None

    _evidence_seq: int = 0

    def _save_evidence_bytes(self, raw: bytes | None) -> str | None:
        if not raw:
            return None
        self._evidence_seq += 1
        name = f"{int(time.time() * 1000)}-{self._evidence_seq}.jpg"
        (self.audit_dir / "evidence" / name).write_bytes(raw)
        return f"evidence/{name}"

    def _append_audit(self, record: Inspection) -> None:
        line = json.dumps(record.public(), separators=(",", ":"))
        with (self.audit_dir / "inspections.jsonl").open("a") as fh:
            fh.write(line + "\n")
