"""HTTP client for the Foreman edge agent running on the Modalix DevKit.

The wire contract between Mac and DevKit, in one place:

  GET  /health                 -> {"ok": true, "models": {...}}
  GET  /events                 -> text/event-stream of detection frames
                                  data: {"frame_id":int,"ts":float,
                                         "detections":[{"label","confidence",
                                                        "bbox":[x1,y1,x2,y2],
                                                        "track_id"}]}
  POST /inspect                 {"standard", "required_objects", "prohibited_objects",
                                 "window_s", "num_frames"}
                               -> {"window": {...}, "detector_summary": str,
                                   "selected": [{frame_id, ts, rel_ts, sharpness,
                                                 detections, jpeg_b64}],
                                   "vlm": {verdict, per_frame, evidence,
                                           missing_evidence, reason},
                                   "metrics": {...}}
                                  Returns evidence, NOT a final verdict: the
                                  grounding policy runs on the Mac.
  POST /inspect_single          {"standard"}  -> single-frame fallback, see above
  POST /transcribe              multipart audio
                               -> {"text": str, "language": str,
                                   "no_speech_prob": float,
                                   "metrics": {"inference_ms"}}

Boxes are normalised to [0,1]. All inference named here happens on the
Modalix MLA; this module only moves bytes.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .gate import Detection


class EdgeError(RuntimeError):
    """The edge agent was unreachable or returned something unusable."""


@dataclass(frozen=True)
class Frame:
    frame_id: int
    ts: float
    detections: list[Detection]


@dataclass(frozen=True)
class SelectedFrame:
    """One representative frame the edge chose from the evidence window."""
    frame_id: int
    ts: float
    rel_ts: float
    sharpness: float
    detections: list[Detection]
    jpeg: bytes


@dataclass(frozen=True)
class WindowEvidence:
    """Raw evidence for one inspection. Deliberately carries NO final verdict -
    the grounding policy that can override the model runs on the Mac."""
    start_ts: float
    end_ts: float
    duration_s: float
    total_frames: int
    image_frames: int
    detector_summary: str
    selected: list[SelectedFrame]
    vlm: dict
    #: one judgement per rule, in rule order, all from the SAME selected frames.
    #: A single-rule inspection has exactly one entry and it equals `vlm`.
    vlms: list[dict]
    metrics: dict
    #: how this window was gathered: mode ("manual"/"auto"), the requested
    #: duration, the real start/end and how many detector frames it covered
    capture: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    verdict: str          # "pass" | "fail" | "unclear"
    reason: str
    evidence_jpeg_b64: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class Transcript:
    text: str
    #: the language actually decoded, always "en" or "ru"
    language: str
    #: what was requested: "en", "ru" or "auto"
    mode: str
    #: False when the audio was too poor to build a standard from; the caller
    #: must then leave the standard in force untouched
    accepted: bool
    reject_reason: str
    avg_logprob: float
    no_speech_prob: float
    metrics: dict[str, float]


def _parse_detection(d: dict) -> Detection:
    bbox = d["bbox"]
    if len(bbox) != 4:
        raise EdgeError(f"bbox must have 4 values, got {bbox!r}")
    return Detection(
        label=str(d["label"]),
        confidence=float(d["confidence"]),
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        track_id=d.get("track_id"),
    )


class EdgeClient:
    """Async client. One instance per process; call `aclose()` on shutdown."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        # Inspection is a VLM call and takes seconds; the event stream never
        # times out on read. Connect timeout stays short so a dead board is
        # reported quickly instead of hanging the UI.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0, read=timeout),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        try:
            r = await self._client.get("/health", timeout=5.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise EdgeError(f"edge health check failed: {exc}") from exc

    async def events(self) -> AsyncIterator[Frame]:
        """Yield detection frames from the edge's SSE stream.

        Raises EdgeError if the stream cannot be opened. Malformed individual
        events are skipped rather than killing the stream, because one bad
        frame must not take the demo down.
        """
        try:
            async with self._client.stream("GET", "/events", timeout=None) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        obj = json.loads(payload)
                        yield Frame(
                            frame_id=int(obj["frame_id"]),
                            ts=float(obj["ts"]),
                            detections=[_parse_detection(d) for d in obj.get("detections", [])],
                        )
                    except (ValueError, KeyError, TypeError, EdgeError):
                        continue
        except httpx.HTTPError as exc:
            raise EdgeError(f"edge event stream failed: {exc}") from exc

    async def inspect(self, standard: str) -> Verdict:
        """Single-frame judgement - the pre-temporal fallback path.

        Used only when the orchestrator runs with FOREMAN_TEMPORAL=0. It has no
        temporal evidence and no detector grounding.
        """
        try:
            r = await self._client.post("/inspect_single", json={"standard": standard})
            r.raise_for_status()
            obj = r.json()
        except httpx.HTTPError as exc:
            raise EdgeError(f"inspect failed: {exc}") from exc
        except ValueError as exc:
            raise EdgeError(f"inspect returned invalid JSON: {exc}") from exc

        verdict = str(obj.get("verdict", "unclear")).lower()
        if verdict not in {"pass", "fail", "unclear"}:
            verdict = "unclear"
        return Verdict(
            verdict=verdict,
            reason=str(obj.get("reason", "")),
            evidence_jpeg_b64=obj.get("evidence_jpeg_b64"),
            metrics={k: float(v) for k, v in (obj.get("metrics") or {}).items()},
        )

    async def inspect_window(
        self,
        standard: str,
        required: list[str],
        prohibited: list[str],
        window_s: float,
        num_frames: int,
        capture_s: float = 0.0,
        relation: str | None = None,
        relation_subject: str | None = None,
        relation_object: str | None = None,
        relation_expected: bool | None = None,
        rules: list[dict] | None = None,
    ) -> WindowEvidence:
        """Ask the edge to judge an evidence window.

        One multi-image VLM call; takes a few seconds. `capture_s` > 0 switches
        to manual capture: the edge collects that many seconds of *new* frames
        from the moment of the call, instead of using the rolling buffer.
        """
        try:
            body = {
                "standard": standard,
                "required_objects": required,
                "prohibited_objects": prohibited,
                "window_s": window_s,
                "num_frames": num_frames,
                "capture_s": capture_s,
            }
            # Only sent for a relationship standard, so a presence rule produces
            # exactly the request - and exactly the prompt - it always did.
            if relation and relation_subject and relation_object:
                body["relation"] = {
                    "name": relation,
                    "subject": relation_subject,
                    "object": relation_object,
                    "expected": bool(relation_expected),
                }
            # One capture, one detector pass, one frame selection - then one
            # judgement per rule over those same frames. Sent as a list so the
            # edge never has to re-open the camera for the second rule.
            if rules:
                body["rules"] = rules
            r = await self._client.post("/inspect", json=body)
            r.raise_for_status()
            obj = r.json()
        except httpx.HTTPError as exc:
            raise EdgeError(f"inspect failed: {exc}") from exc
        except ValueError as exc:
            raise EdgeError(f"inspect returned invalid JSON: {exc}") from exc

        window = obj.get("window") or {}
        selected: list[SelectedFrame] = []
        for item in obj.get("selected") or []:
            try:
                jpeg = base64.b64decode(item.get("jpeg_b64") or "", validate=True)
            except (ValueError, TypeError):
                continue
            selected.append(SelectedFrame(
                frame_id=int(item.get("frame_id", 0)),
                ts=float(item.get("ts", 0.0)),
                rel_ts=float(item.get("rel_ts", 0.0)),
                sharpness=float(item.get("sharpness", 0.0)),
                detections=[_parse_detection(d) for d in item.get("detections", [])],
                jpeg=jpeg,
            ))
        if not selected:
            raise EdgeError("edge returned no usable evidence frames")

        return WindowEvidence(
            start_ts=float(window.get("start_ts", 0.0)),
            end_ts=float(window.get("end_ts", 0.0)),
            duration_s=float(window.get("duration_s", 0.0)),
            total_frames=int(window.get("total_frames", 0)),
            image_frames=int(window.get("image_frames", 0)),
            detector_summary=str(obj.get("detector_summary", "")),
            selected=selected,
            vlm=obj.get("vlm") or {},
            # An edge that predates multi-rule returns only "vlm"; treat that as
            # a single-rule answer rather than failing the inspection.
            vlms=[j for j in (obj.get("vlms") or []) if isinstance(j, dict)]
                 or [obj.get("vlm") or {}],
            metrics={k: float(v) for k, v in (obj.get("metrics") or {}).items()},
            capture=obj.get("capture") or {},
        )

    async def transcribe(self, audio: bytes, filename: str = "speech.wav",
                         language: str = "auto") -> Transcript:
        """Send audio to the DevKit's Whisper ASR (MLA).

        `language` is "en", "ru" or "auto"; auto chooses between those two only.
        Nothing is loaded or restarted when it changes - it is a per-request
        decoding parameter, so the switch takes effect on the next recording.
        """
        try:
            r = await self._client.post(
                "/transcribe",
                params={"language": language},
                files={"file": (filename, audio, "application/octet-stream")},
            )
            r.raise_for_status()
            obj = r.json()
        except httpx.HTTPError as exc:
            raise EdgeError(f"transcribe failed: {exc}") from exc
        except ValueError as exc:
            raise EdgeError(f"transcribe returned invalid JSON: {exc}") from exc

        return Transcript(
            text=str(obj.get("text", "")).strip(),
            language=str(obj.get("language", "unknown")),
            mode=str(obj.get("mode", language)),
            accepted=bool(obj.get("accepted", True)),
            reject_reason=str(obj.get("reject_reason", "")),
            avg_logprob=float(obj.get("avg_logprob", 0.0)),
            no_speech_prob=float(obj.get("no_speech_prob", 0.0)),
            metrics={k: float(v) for k, v in (obj.get("metrics") or {}).items()},
        )
