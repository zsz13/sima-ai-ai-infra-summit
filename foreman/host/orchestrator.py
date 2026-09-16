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

MAX_RECENT = 40
RECONNECT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0)


@dataclass
class Inspection:
    id: str
    ts: float
    verdict: str
    reason: str
    standard: str
    trigger_label: str | None
    trigger_confidence: float | None
    metrics: dict[str, float]
    evidence_path: str | None = None

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
    detections_last_frame: int = 0
    fps: float = 0.0
    inspecting: bool = False

    counters: Counters = field(default_factory=Counters)
    recent: deque[Inspection] = field(default_factory=lambda: deque(maxlen=MAX_RECENT))

    _gate: Gate = field(init=False)
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
            "frames_seen": self.frames_seen,
            "detections_last_frame": self.detections_last_frame,
            "fps": round(self.fps, 1),
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

    async def _on_frame(self, frame: Frame) -> None:
        self.frames_seen += 1
        self.detections_last_frame = len(frame.detections)
        self._fps_window.append(time.monotonic())
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            self.fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0

        should_inspect = self._gate.update(frame.detections)

        if should_inspect and self.standard and not self.inspecting:
            asyncio.create_task(self._inspect())
        else:
            self._publish()

    async def _inspect(self) -> None:
        """Run one VLM judgement on the DevKit and record the result."""
        self.inspecting = True
        self._publish()
        trigger = self._gate.last_trigger
        started = time.monotonic()
        try:
            verdict: Verdict = await self.edge.inspect(self.standard)
        except EdgeError as exc:
            self.counters.errors += 1
            self.last_error = str(exc)
            self.inspecting = False
            self._publish()
            return

        elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics = dict(verdict.metrics)
        # Reported separately and honestly: the model's own inference time comes
        # from the edge; this is the full round trip measured on the Mac.
        metrics["end_to_end_ms"] = round(elapsed_ms, 1)

        record = Inspection(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            verdict=verdict.verdict,
            reason=verdict.reason,
            standard=self.standard,
            trigger_label=trigger.label if trigger else None,
            trigger_confidence=trigger.confidence if trigger else None,
            metrics=metrics,
            evidence_path=self._save_evidence(verdict.evidence_jpeg_b64),
        )

        if verdict.verdict == "pass":
            self.counters.passed += 1
        elif verdict.verdict == "fail":
            self.counters.failed += 1
        else:
            self.counters.unclear += 1

        self.recent.append(record)
        self._append_audit(record)
        self.inspecting = False
        self._publish()

    # --- persistence ----------------------------------------------------

    def _save_evidence(self, b64: str | None) -> str | None:
        if not b64:
            return None
        try:
            raw = base64.b64decode(b64, validate=True)
        except (ValueError, TypeError):
            return None
        name = f"{int(time.time() * 1000)}.jpg"
        (self.audit_dir / "evidence" / name).write_bytes(raw)
        return f"evidence/{name}"

    def _append_audit(self, record: Inspection) -> None:
        line = json.dumps(record.public(), separators=(",", ":"))
        with (self.audit_dir / "inspections.jsonl").open("a") as fh:
            fh.write(line + "\n")
