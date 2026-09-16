"""HTTP client for the Foreman edge agent running on the Modalix DevKit.

The wire contract between Mac and DevKit, in one place:

  GET  /health                 -> {"ok": true, "models": {...}}
  GET  /events                 -> text/event-stream of detection frames
                                  data: {"frame_id":int,"ts":float,
                                         "detections":[{"label","confidence",
                                                        "bbox":[x1,y1,x2,y2],
                                                        "track_id"}]}
  POST /inspect                 {"standard": str}
                               -> {"verdict":"pass"|"fail"|"unclear",
                                   "reason": str, "evidence_jpeg_b64": str,
                                   "metrics": {"ttft_ms","tokens_per_s",
                                               "inference_ms"}}
  POST /transcribe              multipart audio
                               -> {"text": str, "language": str,
                                   "no_speech_prob": float,
                                   "metrics": {"inference_ms"}}

Boxes are normalised to [0,1]. All inference named here happens on the
Modalix MLA; this module only moves bytes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
class Verdict:
    verdict: str          # "pass" | "fail" | "unclear"
    reason: str
    evidence_jpeg_b64: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
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
        """Ask the DevKit to judge the current frame against `standard`.

        This is the VLM call. It runs on the MLA and takes seconds.
        """
        try:
            r = await self._client.post("/inspect", json={"standard": standard})
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

    async def transcribe(self, audio: bytes, filename: str = "speech.wav") -> Transcript:
        """Send audio to the DevKit's Whisper ASR (MLA)."""
        try:
            r = await self._client.post(
                "/transcribe",
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
            no_speech_prob=float(obj.get("no_speech_prob", 0.0)),
            metrics={k: float(v) for k, v in (obj.get("metrics") or {}).items()},
        )
