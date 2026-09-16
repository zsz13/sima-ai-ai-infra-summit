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
from .policy import (
    GroundingConfig,
    VlmJudgement,
    aggregate_verdicts,
    build_evidence,
    decide,
)
from .standard_parser import parse_standard, resolve_language

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
    #: which inference backend produced this verdict. Written into every audit
    #: record so a development result can never be read back as hardware output.
    backend: str = "unknown"
    #: how the evidence was gathered: "manual" (the operator pressed Inspect now
    #: and the window starts from that moment) or "auto" (the rolling window the
    #: gate fired on). Stored so a reviewer can tell which a record came from.
    capture: dict = field(default_factory=dict)
    #: one entry per rule: its verbatim text, its own verdict, reason,
    #: attribution, normalised semantics and the objects it was judged against.
    #: A single-rule inspection has exactly one entry, and the top-level
    #: verdict/reason/standard mirror it, so older readers keep working.
    #: `verdict` above is the aggregate across these.
    rules: list[dict] = field(default_factory=list)

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
    #: "en" or "ru" - which lexicon parses the standard above
    standard_language: str = "en"
    #: The rules in force, in order, verbatim as the operator entered them.
    #: At most two. `standard` and `standard_language` mirror rules[0] so every
    #: existing single-rule caller, audit record and export keeps working; the
    #: list is the source of truth.
    rules: list[str] = field(default_factory=list)
    rule_languages: list[str] = field(default_factory=list)
    #: which inference backend is answering: "modalix", "local" or "fake".
    #: Reported by the edge itself at /health and carried into every audit
    #: record, so a development result cannot be read as hardware output.
    backend: str = "unknown"
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
    #: "manual" while an operator-triggered inspection is in flight, "idle"
    #: otherwise. The host cannot see the boundary between capture, selection and
    #: the model call - the edge does all three in one blocking request - so it
    #: publishes the start time and duration and lets the console derive the
    #: phase, rather than asserting a phase it does not observe.
    capture_phase: str = "idle"
    capture_started_at: float = 0.0
    capture_s: float = 3.0
    #: Manual only. A moment between the button press and the first recorded
    #: frame, so the operator can get into position without the act of reaching
    #: for the button becoming the evidence. Deliberately not part of the window:
    #: the detector keeps running for the live view, but nothing captured during
    #: preparation reaches the inspection.
    prepare_s: float = 0.8
    #: When the current manual preparation began, 0.0 when not preparing.
    prepare_started_at: float = 0.0
    #: "manual" (default) or "auto". Manual means setting a standard arms the
    #: system and nothing else; only "Inspect now" starts an inspection. Auto
    #: restores the rolling gate. The two never run at once - an operator who
    #: pressed a button should not also be racing an automatic trigger.
    inspection_mode: str = "manual"
    #: Bumped by clear_session(). Any async work that was already in flight
    #: captures this at the start and refuses to apply its result if it has
    #: changed, so a slow transcription or a running inspection cannot reinstate
    #: a standard - or record a verdict - after the operator cleared the session.
    session_epoch: int = 0

    counters: Counters = field(default_factory=Counters)
    recent: deque[Inspection] = field(default_factory=lambda: deque(maxlen=MAX_RECENT))

    temporal: bool = True
    window_s: float = 3.0
    evidence_frames: int = 6
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
            "standard_language": self.standard_language,
            "rules": list(self.rules),
            "parsed_rules": self.parsed_rules(),
            "backend": self.backend,
            "inspection_mode": self.inspection_mode,
            "capture_phase": self.capture_phase,
            "capture_started_at": self.capture_started_at,
            "capture_s": self.capture_s,
            "prepare_s": self.prepare_s,
            "prepare_started_at": self.prepare_started_at,
            "parsed_standard": self.parsed_standard(),
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

    def set_standard(self, text: str, language: str | None = None,
                     epoch: int | None = None) -> bool:
        """Adopt a new inspection standard and re-arm the gate.

        `language` is what the speech recogniser decoded ("en" or "ru"). It is a
        hint, not an instruction: the lexicon is chosen from the script the text
        is actually written in, so a mislabelled transcript cannot be parsed with
        a lexicon that matches none of its words - which would ground nothing and
        quietly hand the verdict to the vision-language model alone.
        """
        if epoch is not None and epoch != self.session_epoch:
            # The session was cleared while this was being produced. Applying it
            # now would silently resurrect a standard the operator discarded.
            return False
        self._apply_rules([text], language)
        self._gate.reset()
        self._publish()
        return True

    #: Two is the supported maximum. It is enforced rather than assumed, so a
    #: caller that tries for three gets an error instead of a silently dropped
    #: rule the operator still believes is in force.
    MAX_RULES = 2

    def set_rules(self, texts: list[str], language: str | None = None,
                  epoch: int | None = None) -> bool:
        """Replace the rules in force. One or two, each kept verbatim."""
        if epoch is not None and epoch != self.session_epoch:
            return False
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if len(cleaned) > self.MAX_RULES:
            raise ValueError(f"at most two rules are supported, got {len(cleaned)}")
        self._apply_rules(cleaned, language)
        self._gate.reset()
        self._publish()
        return True

    def _apply_rules(self, texts: list[str], language: str | None) -> None:
        self.rules = [t.strip() for t in texts if t and t.strip()]
        # Each rule resolves its own language, so an English rule and a Russian
        # one can be in force at the same time.
        self.rule_languages = [resolve_language(t, language) for t in self.rules]
        self.standard = self.rules[0] if self.rules else ""
        self.standard_language = self.rule_languages[0] if self.rule_languages else "en"

    def set_inspection_mode(self, mode: str) -> str:
        """Switch between manual and auto without disturbing the standard.

        Auto -> manual stops automatic triggering immediately; the standard in
        force is deliberately kept, because changing how inspections start is not
        the same as changing what is being inspected.
        """
        if mode not in ("manual", "auto"):
            raise ValueError("mode must be 'manual' or 'auto'")
        self.inspection_mode = mode
        self._gate.reset()
        self._publish()
        return self.inspection_mode

    async def inspect_now(self, capture_s: float | None = None) -> dict:
        """Manual inspection: capture from this moment, then judge.

        The window deliberately starts when the operator presses the button, so
        what is judged is what they just did - not whatever happened to be in the
        rolling buffer beforehand. Everything after capture is the same code path
        as an automatic inspection: same selection, same grounding, same policy.
        """
        if not self.standard:
            return {"started": False, "reason": "no standard is set"}
        if self.inspecting:
            return {"started": False, "reason": "an inspection is already running"}
        if not self.connected:
            return {"started": False, "reason": "the edge is not connected"}
        self.capture_s = float(capture_s or self.capture_s)
        asyncio.create_task(self._inspect(manual=True))
        return {"started": True, "capture_s": self.capture_s}

    def parsed_standard(self) -> dict:
        """The detector-groundable reading of the current standard, for the UI.

        Showing this is how an operator can tell that a rule was understood -
        and, just as importantly, when it was not and the verdict will therefore
        rest on the vision-language model alone.
        """
        return parse_standard(self.standard, self.standard_language).public()

    def parsed_rules(self) -> list[dict]:
        """Each rule parsed on its own.

        Deliberately not one parse of the two texts joined together: the rules
        are independent, and concatenating them would let one rule's negation
        scope leak into the other.
        """
        return [parse_standard(t, lang).public()
                for t, lang in zip(self.rules, self.rule_languages, strict=False)]

    def tracked_objects(self) -> list[str]:
        """The union of every class either rule needs measured.

        One detector pass serves both rules, so the window has to track
        everything either of them mentions - but each rule is still judged
        against only its own objects.
        """
        out: list[str] = []
        for text, lang in zip(self.rules, self.rule_languages, strict=False):
            for label in parse_standard(text, lang).tracked_objects:
                if label not in out:
                    out.append(label)
        return out

    def clear_session(self) -> None:
        """Clear the session completely, including the standard in force.

        The standard used to survive this, which meant "Clear session" re-armed
        the gate and the next object in view was immediately inspected against
        the rule the operator had just tried to get rid of. Clearing means
        clearing: after this, nothing is inspected until a new standard is set.

        The epoch bump is what makes it stick. Work already in flight - a
        transcription, an inspection - checks the epoch before applying its
        result, so a reply that arrives after the clear is discarded instead of
        reinstating the old state.
        """
        self.session_epoch += 1
        self.rules = []
        self.rule_languages = []
        self.standard = ""
        self.standard_language = "en"
        self.counters = Counters()
        self.recent.clear()
        self._gate.reset()
        self.warming_up = False
        self.capture_phase = "idle"
        self.capture_started_at = 0.0
        self.last_error = None
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
                health = await self.edge.health()
                self.backend = str(health.get("backend") or "unknown")
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

        if self.paused:
            # Camera Check must not consume items. Advancing the gate while
            # paused latches the item in view as already inspected, so returning
            # to the inspection view would sit idle until that item left the
            # frame and came back - the console looks dead for no visible reason.
            # Keep the gate armed instead; the evidence ring above still fills.
            self._gate.reset()
            self._publish()
            return

        should_inspect = self._gate.update(frame.detections)

        # Manual mode keeps the gate running - its state drives the "hold still"
        # hint and Camera Check - but never lets it start an inspection.
        if (should_inspect and self.standard and not self.inspecting
                and self.inspection_mode == "auto"):
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

    async def _inspect(self, manual: bool = False) -> None:
        """Judge one item. Temporal by default; single-frame only as a fallback."""
        epoch = self.session_epoch
        self.inspecting = True
        prepared_at = 0.0
        if manual:
            # Preparation first, and it is announced as its own phase so the
            # console never implies that evidence is being recorded yet.
            self.capture_phase = "preparing"
            self.prepare_started_at = time.time()
            self.capture_started_at = 0.0
            self._publish()
            if self.prepare_s > 0:
                await asyncio.sleep(self.prepare_s)
            # Anything that happened during preparation can cancel it. Clearing
            # the session must not leave a capture to fire afterwards, and an
            # edge that went away must not be handed a request seconds later.
            if epoch != self.session_epoch or not self.connected:
                self.inspecting = False
                self.capture_phase = "idle"
                self.prepare_started_at = 0.0
                self.capture_started_at = 0.0
                self._publish()
                return
            prepared_at = time.time()
            self.capture_phase = "manual"
            self.capture_started_at = prepared_at
        self._publish()
        trigger = self._gate.last_trigger
        started = time.monotonic()
        try:
            if self.temporal:
                record = await self._inspect_temporal(trigger, started, manual=manual,
                                                      prepared_at=prepared_at)
            else:
                record = await self._inspect_single_frame(trigger, started)
        except EdgeError as exc:
            self.counters.errors += 1
            self.last_error = str(exc)
            self.inspecting = False
            self.capture_phase = "idle"
            self._publish()
            return
        finally:
            self.capture_phase = "idle"
            self.prepare_started_at = 0.0

        if epoch != self.session_epoch:
            # Cleared while this was running. Discard it rather than counting a
            # verdict against a standard that is no longer in force.
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

    async def _inspect_temporal(self, trigger, started: float,
                                manual: bool = False,
                                prepared_at: float = 0.0) -> Inspection:
        """Detector evidence across the window decides what the VLM is allowed to say."""
        texts = self.rules or ([self.standard] if self.standard else [])
        langs = self.rule_languages or [self.standard_language]
        parsed_rules = [parse_standard(t, lang)
                        for t, lang in zip(texts, langs, strict=False)]
        parsed = parsed_rules[0]

        # The union of what either rule needs, so one detector pass and one set
        # of evidence frames serve both. Each rule is still judged against only
        # its own objects further down.
        union_required: list[str] = []
        union_prohibited: list[str] = []
        for pr in parsed_rules:
            for c in pr.required:
                if c not in union_required:
                    union_required.append(c)
            for c in pr.prohibited:
                if c not in union_prohibited:
                    union_prohibited.append(c)

        # One request: one capture, one detector pass, one frame selection, then
        # one judgement per rule over those same frames. Two requests would mean
        # two windows, and the rules would no longer be talking about the same
        # three seconds.
        rule_specs = [{
            "standard": pr.raw,
            "required_objects": list(pr.required),
            "prohibited_objects": list(pr.prohibited),
            "relation": ({"name": pr.relation, "subject": pr.relation_subject,
                          "object": pr.relation_object,
                          "expected": bool(pr.relation_expected)}
                         if pr.relation and pr.relation_object else None),
        } for pr in parsed_rules]

        # The edge blocks for the whole capture in manual mode, so the phase only
        # advances once it returns with the frames already selected.
        evidence = await self.edge.inspect_window(
            self.standard, union_required, union_prohibited,
            self.window_s, self.evidence_frames,
            capture_s=self.capture_s if manual else 0.0,
            relation=parsed.relation, relation_subject=parsed.relation_subject,
            relation_object=parsed.relation_object,
            relation_expected=parsed.relation_expected,
            rules=rule_specs if len(rule_specs) > 1 else None)
        frames = self._detections_between(evidence.start_ts, evidence.end_ts)
        if not frames:
            # The edge counted frames we never received; fall back to its count so
            # the window-length check still means something.
            frames = [[]] * evidence.total_frames

        # ONE aggregation over the whole window, shared by every rule. The
        # detector never runs twice: tracked_objects is the union, and each rule
        # then reads only the classes it names.
        all_tracked: list[str] = []
        for pr in parsed_rules:
            for c in pr.tracked_objects:
                if c not in all_tracked:
                    all_tracked.append(c)
        detector = build_evidence(frames, all_tracked, self.grounding)

        temporal_n = int(evidence.metrics.get("temporal_frames")
                         or len(((evidence.vlm or {}).get("per_frame")) or ()))

        def judgement_from(raw: dict) -> VlmJudgement:
            # Count agreement over distinct temporal frames only. ROI crops are
            # close-ups of a moment already represented, so including them would
            # overstate how many independent views agreed.
            return VlmJudgement(
                verdict=str(raw.get("verdict", "unclear")).lower(),
                reason=str(raw.get("reason", "")),
                evidence=tuple(raw.get("evidence") or ()),
                missing_evidence=tuple(raw.get("missing_evidence") or ()),
                per_frame=tuple((raw.get("per_frame") or ())[:temporal_n]),
                observed_relationship=raw.get("observed_relationship"),
            )

        judgements = evidence.vlms or [evidence.vlm or {}]
        rule_records: list[dict] = []
        decisions = []
        for i, pr in enumerate(parsed_rules):
            raw = judgements[i] if i < len(judgements) else {}
            # Each rule sees only the evidence for the classes it names, which is
            # what keeps rule 1's reason from discussing rule 2's objects.
            scoped = {k: v for k, v in detector.items() if k in set(pr.tracked_objects)}
            d = decide(pr, scoped, judgement_from(raw), self.grounding)
            decisions.append(d)
            rule_records.append({
                "index": i + 1,
                "text": pr.raw,
                "language": pr.language,
                "verdict": d.verdict,
                "reason": d.reason,
                "decided_by": d.decided_by,
                "parsed": pr.public(),
                "required_objects": d.required,
                "prohibited_objects": d.prohibited,
                "vlm": d.vlm,
                "notes": d.notes,
            })

        decision = decisions[0]
        overall = aggregate_verdicts([d.verdict for d in decisions])
        # With one rule the record is exactly what it always was. With two, the
        # headline is the aggregate and each rule keeps its own sentence.
        if len(decisions) > 1:
            overall_reason = " ".join(
                f"Rule {r['index']}: {r['verdict'].upper()}. {r['reason']}"
                for r in rule_records)
        else:
            overall_reason = decision.reason

        paths = [p for p in (self._save_evidence_bytes(f.jpeg) for f in evidence.selected) if p]
        elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics = dict(evidence.metrics)
        metrics["end_to_end_ms"] = round(elapsed_ms, 1)
        metrics["host_ms"] = round(elapsed_ms - metrics.get("inference_ms", 0.0), 1)

        return Inspection(
            id=uuid.uuid4().hex[:12],
            ts=time.time(),
            backend=self.backend,
            capture=(evidence.capture or {}) | {
                "selected_frames": len(paths),
                "selected_rel_ts": [round(f.rel_ts, 3) for f in evidence.selected],
            } | ({"prepare_s": self.prepare_s, "prepared_at": prepared_at}
                 if manual and prepared_at else {}),
            verdict=overall,
            reason=overall_reason,
            standard=self.standard,
            rules=rule_records,
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
            backend=self.backend,
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
