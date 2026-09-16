#!/usr/bin/env python3
"""OPTIONAL local development backend. Not the demo path.

=====================================================================
This serves the same Edge API as edge/foreman_edge.py, but runs the
models on this Mac instead of the Modalix MLA. It exists so Foreman
can be developed and tested without a DevKit. Every result it produces
is labelled `backend: "local"` all the way into the audit trail, so a
local verdict can never be read back as hardware output.

It is NOT a reference for Modalix behaviour: the board runs quantised
models (whisper-small-a16w8, Qwen3-VL-2B-GPTQ-a16w4) and these are the
upstream weights, so the numbers and sometimes the answers differ.
Never quote a latency measured here as a Modalix benchmark.
=====================================================================

The temporal logic is deliberately *not* reimplemented. The rolling evidence
window, representative-frame selection, nested-box suppression, the model-reply
parser and the speech-language constraint are all imported from foreman_edge,
which keeps them pure of pyneat. Only the three inference calls are swapped, so
a bug fixed in one backend is fixed in both.

Model adapters load lazily and degrade honestly: a capability whose dependency
is absent reports itself unavailable at /health and returns a 503 saying exactly
what to install, rather than pretending or crashing at request time.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import foreman_edge as fe  # noqa: E402  (pure helpers only; pyneat is never loaded)

BACKEND = "local"


# ----------------------------------------------------------- model adapters

class Unavailable(RuntimeError):
    """A capability whose dependency is not installed on this machine."""


class Capability:
    """Base for a lazily-loaded local model.

    `install_hint` is surfaced verbatim to the caller, so a missing dependency
    produces an actionable message instead of a stack trace.
    """

    name = "capability"
    install_hint = ""

    def __init__(self) -> None:
        self._loaded = False
        self._error: str | None = None

    def available(self) -> bool:
        try:
            self.ensure()
        except Unavailable as exc:
            self._error = str(exc)
            return False
        return True

    def status(self) -> dict:
        ok = self.available()
        return {"available": ok, "detail": self._error or "", "hint": "" if ok else self.install_hint}

    def ensure(self) -> None:
        raise NotImplementedError


class LocalDetector(Capability):
    """YOLO26n via onnxruntime, with the CoreML execution provider when present.

    ONNX Runtime rather than the full PyTorch/Ultralytics stack: it is a far
    smaller dependency and needs neither torch nor scipy. The weights are the
    ONNX export of YOLO26n - the same model family
    the board runs as `yolo_26n_mpk`, so the class semantics line up - but these
    are the upstream float weights, not the quantised MLA build, so the boxes
    and scores will not match the board exactly.

    The export is NMS-free: it emits 300 candidate queries, and a candidate is
    kept on score alone. Foreman's own nested-box suppression still runs on top,
    the same code the Modalix path uses.
    """

    name = "detector"
    install_hint = ("uv sync --group local, then put a YOLO .onnx at "
                    "$FOREMAN_LOCAL_DETECTOR (see docs/LOCAL_BACKEND.md)")
    #: resize + rescale only, per the model's own preprocessor_config.json
    SIZE = 640

    def __init__(self, model_path: str, labels: list[str], score_threshold: float) -> None:
        super().__init__()
        self.model_path = model_path
        self.labels = labels
        self.score_threshold = score_threshold
        self.providers: list[str] = []
        self._session = None
        self._last_ms = 0.0

    def ensure(self) -> None:
        if self._loaded:
            return
        try:
            import numpy  # noqa: F401
            import onnxruntime
        except ImportError as exc:
            raise Unavailable(f"onnxruntime/numpy not installed ({exc})") from exc
        if not self.model_path or not Path(self.model_path).is_file():
            raise Unavailable(f"detector model not found at {self.model_path!r}")
        # CPU by default, deliberately. Measured on this model: the CoreML
        # execution provider returns effectively empty output - top score 0.0001
        # against 0.9530 on CPU for the same frame - while being only ~12%
        # faster (8.8 ms vs 10.0 ms). It partitions the graph into 18 pieces and
        # something in that split is wrong. A provider that silently reports no
        # objects is far worse than one that is marginally slower, so CoreML is
        # opt-in via FOREMAN_LOCAL_ORT_COREML=1 and must be re-verified against
        # CPU output before it is trusted.
        available = onnxruntime.get_available_providers()
        if os.environ.get("FOREMAN_LOCAL_ORT_COREML") == "1" and \
                "CoreMLExecutionProvider" in available:
            self.providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
        self._session = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        self._input = self._session.get_inputs()[0].name
        self._loaded = True

    def _letterbox(self, rgb):
        """Resize preserving aspect ratio, padding to a square.

        The model's own preprocessor_config.json says do_pad:false - a plain
        stretch to 640x640 - but that config is a generic YolosImageProcessor
        default and it measurably hurts this model. Squashing a 1280x720 frame
        distorts every object by 1.78x vertically. Measured on the same live
        frames: person 0.115 squashed against 0.293 letterboxed, and on one
        frame squash found nothing at all where letterbox found a person at
        0.354. Letterbox is what YOLO is trained with, so it is what we send.

        Returns the tensor plus the mapping needed to put boxes back into
        original-frame coordinates.
        """
        import numpy as np
        from PIL import Image

        h, w = rgb.shape[:2]
        scale = self.SIZE / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        canvas = Image.new("RGB", (self.SIZE, self.SIZE), (114, 114, 114))
        canvas.paste(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR),
                     ((self.SIZE - nw) // 2, (self.SIZE - nh) // 2))
        arr = (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        return arr, {"pad_x": (self.SIZE - nw) // 2, "pad_y": (self.SIZE - nh) // 2,
                     "nw": nw, "nh": nh}

    def _raw(self, rgb):
        """(labels, confidences, boxes, mapping) for every query, unfiltered."""
        import numpy as np

        self.ensure()
        arr, box_map = self._letterbox(rgb)
        logits, boxes = self._session.run(None, {self._input: arr})
        x = logits[0]
        # Stable sigmoid: exp(-x) overflows float32 on the very negative logits
        # this model emits for rejected queries.
        scores = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                          np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))
        best = scores.argmax(axis=1)
        conf = scores[np.arange(scores.shape[0]), best]
        return best, conf, boxes[0], box_map

    def _to_frame(self, box, m) -> list[float]:
        """cxcywh on the padded canvas -> normalised xyxy on the original frame."""
        cx, cy, bw, bh = (float(v) * self.SIZE for v in box)
        x1 = (cx - bw / 2.0 - m["pad_x"]) / m["nw"]
        y1 = (cy - bh / 2.0 - m["pad_y"]) / m["nh"]
        x2 = (cx + bw / 2.0 - m["pad_x"]) / m["nw"]
        y2 = (cy + bh / 2.0 - m["pad_y"]) / m["nh"]
        return [round(max(0.0, min(1.0, v)), 5) for v in (x1, y1, x2, y2)]

    def detect(self, rgb, threshold: float | None = None) -> list[dict]:
        """Detections for one RGB frame, in Foreman's schema.

        `bbox` is normalised [x1, y1, x2, y2] against the full frame, exactly
        what the Modalix edge emits, so the console, the policy and the evidence
        overlays need no per-backend handling.
        """
        started = time.monotonic()
        best, conf, boxes, box_map = self._raw(rgb)
        cutoff = self.score_threshold if threshold is None else threshold

        out: list[dict] = []
        for cls, score, box in zip(best, conf, boxes, strict=True):
            if score < cutoff:
                continue
            label = self.labels[int(cls)] if int(cls) < len(self.labels) else f"class_{int(cls)}"
            out.append({
                "label": label,
                "confidence": round(float(score), 4),
                "bbox": self._to_frame(box, box_map),
                "track_id": None,
            })
        self._last_ms = (time.monotonic() - started) * 1000.0
        # Same suppression the board applies, imported rather than reimplemented.
        return fe.suppress_nested_duplicates(out)

    def detect_debug(self, rgb, threshold: float) -> list[dict]:
        """Every query above `threshold`, for Camera Check's debug view only.

        Deliberately a separate call on a separate endpoint. Nothing here reaches
        the evidence buffer, the gate or host/policy.py, so a low debug threshold
        can never move a PASS/FAIL. Each detection carries the runner-up class,
        because the useful question is usually "what else did it think this was".
        """
        import numpy as np

        best, conf, boxes, box_map = self._raw(rgb)
        arr, _ = self._letterbox(rgb)
        logits, _ = self._session.run(None, {self._input: arr})
        x = logits[0]
        scores = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                          np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))

        out: list[dict] = []
        for i, (cls, score, box) in enumerate(zip(best, conf, boxes, strict=True)):
            if score < threshold:
                continue
            order = scores[i].argsort()[::-1][:3]
            out.append({
                "label": self.labels[int(cls)] if int(cls) < len(self.labels) else f"class_{int(cls)}",
                "confidence": round(float(score), 4),
                "bbox": self._to_frame(box, box_map),
                "track_id": None,
                "below_threshold": bool(score < self.score_threshold),
                "alternatives": [
                    {"label": self.labels[int(c)], "confidence": round(float(scores[i][int(c)]), 4)}
                    for c in order if int(c) < len(self.labels)
                ],
            })
        out.sort(key=lambda d: -d["confidence"])
        return out[:40]


class LocalASR(Capability):
    """Whisper locally, via faster-whisper (CTranslate2).

    faster-whisper rather than mlx-whisper on purpose: mlx-whisper resolves
    torch, scipy and numba, while CTranslate2 needs none of them and already
    brings the onnxruntime the detector uses. int8 on this machine's CPU is fast
    enough that the Metal path was not worth a 500 MB dependency tree.

    The English/Russian/Auto behaviour is *inherited*, not reimplemented: the
    engine subclasses foreman_edge.GenAI and overrides only the single forced
    decode, so mode handling, the auto early-break, the script-agreement rule
    and the unclear-speech rejection are literally the same code the Modalix
    path runs.
    """

    name = "asr"
    install_hint = "uv sync --group local  (installs faster-whisper)"

    class _Engine(fe.GenAI):
        """foreman_edge.GenAI with the one network call replaced by local Whisper."""

        def __init__(self, model_name: str, compute_type: str) -> None:
            super().__init__("local://", "vlm", "asr")
            self.model_name = model_name
            self.compute_type = compute_type
            self._model = None

        def _load(self):
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_name, device="cpu",
                                           compute_type=self.compute_type)
            return self._model

        def transcribe_forced(self, audio: bytes, filename: str, language: str) -> dict:
            import tempfile

            model = self._load()
            started = time.monotonic()
            suffix = Path(filename or "speech.webm").suffix or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio)
                tmp.flush()
                # condition_on_previous_text=False keeps one clip from dragging a
                # previous hallucination forward, which is what makes Whisper
                # repeat itself on silence.
                segments, _info = model.transcribe(
                    tmp.name, language=language, beam_size=5,
                    condition_on_previous_text=False, vad_filter=False)
                segments = list(segments)

            text = "".join(seg.text for seg in segments).strip()
            # Duration-weighted, so a long clear segment is not outvoted by a
            # short noisy one. Same two signals the board's server reports.
            total = sum(max(seg.end - seg.start, 1e-6) for seg in segments) or 1.0
            avg_logprob = (sum(seg.avg_logprob * max(seg.end - seg.start, 1e-6)
                               for seg in segments) / total) if segments else -99.0
            no_speech = (max(seg.no_speech_prob for seg in segments) if segments else 1.0)
            return {
                "text": text,
                "language": language,
                "avg_logprob": float(avg_logprob),
                "no_speech_prob": float(no_speech),
                "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
            }

    def __init__(self, model_name: str = "small", compute_type: str = "int8") -> None:
        super().__init__()
        self.model_name = model_name
        self.compute_type = compute_type
        self._engine = self._Engine(model_name, compute_type)

    def ensure(self) -> None:
        if self._loaded:
            return
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise Unavailable(f"faster-whisper not installed ({exc})") from exc
        self._loaded = True

    def warm(self) -> None:
        """Load the weights up front so the first spoken standard is not slow."""
        self.ensure()
        self._engine._load()

    def transcribe(self, audio: bytes, filename: str, mode: str) -> dict:
        self.ensure()
        return self._engine.transcribe(audio, filename, mode)


def trace_vlm(path: str, *, model: str, n_images: int, prompt: str,
              raw: str, metrics: dict) -> None:
    """Append one prompt/reply pair to a JSONL trace. Opt-in, never fatal.

    The audit trail stores the *parsed* judgement, which was not enough: when the
    model copied the prompt's own example verbatim, the only record was a clean
    sentence that looked like an observation, and reconstructing what had really
    been emitted took a replay harness. This records the reply untouched, next to
    the prompt that produced it.

    Local backend only, and a failure here must never cost an inspection, so
    every error is swallowed.
    """
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": time.time(),
                "model": model,
                "n_images": n_images,
                "prompt": prompt,
                "raw": raw,
                "metrics": metrics,
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


class LocalVLM(Capability):
    """Qwen3-VL-2B via mlx-vlm, on Metal.

    Native rather than containerised: Docker Desktop on macOS has no Metal
    passthrough, so a containerised 2B model would be CPU-only and far slower
    than the board it stands in for.

    The prompt is `foreman_edge.build_window_prompt` - byte-identical to the one
    the Modalix path sends, pinned by tests - and the reply goes through
    `foreman_edge.parse_window_judgement`, the same parser with the same
    verdict/boolean cross-check and placeholder rejection. Only the transport
    differs: an in-process MLX call instead of an HTTP request to the board.
    """

    name = "vlm"
    install_hint = "uv sync --group local  (installs mlx-vlm; Apple Silicon only)"

    def __init__(self, model_id: str, max_tokens: int = 260) -> None:
        super().__init__()
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model = None
        self._processor = None
        self._config = None
        self.load_ms = 0.0
        self.peak_memory_gb = 0.0

    def ensure(self) -> None:
        if self._loaded:
            return
        try:
            import mlx_vlm  # noqa: F401
        except ImportError as exc:
            raise Unavailable(f"mlx-vlm not installed ({exc})") from exc
        self._loaded = True

    def warm(self) -> None:
        """Load the weights up front. First load is seconds; a demo should not
        pay that on its first inspection."""
        self.ensure()
        if self._model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        started = time.monotonic()
        self._model, self._processor = load(self.model_id)
        self._config = load_config(self.model_id)
        self.load_ms = (time.monotonic() - started) * 1000.0
        print(f"[local-edge] VLM loaded in {self.load_ms / 1000:.1f}s: {self.model_id}", flush=True)

    def judge_window(self, jpegs: list[bytes], standard: str, detector_summary: str,
                     roi_label: str | None = None, roi_count: int = 0,
                     confirmed: list[str] | None = None,
                     relation: tuple[str, str, str, bool] | None = None) -> tuple[dict, dict]:
        """One multi-image judgement, matching the Modalix contract exactly."""
        import tempfile

        from mlx_vlm import apply_chat_template, generate

        self.warm()
        n = len(jpegs)
        # (name, subject, object, expected) or None for a presence-only rule.
        rel_name, rel_subj, rel_obj, rel_want = relation or (None, None, None, None)
        prompt_text = fe.build_window_prompt(
            n, standard, detector_summary, roi_label, roi_count, confirmed,
            relation=rel_name, relation_subject=rel_subj,
            relation_object=rel_obj, relation_expected=rel_want)
        started = time.monotonic()
        # mlx-vlm takes image paths, so the frames are written to a temporary
        # directory that is removed as soon as generation returns.
        with tempfile.TemporaryDirectory(prefix="foreman-vlm-") as tmp:
            paths = []
            for i, jpeg in enumerate(jpegs):
                path = Path(tmp) / f"frame{i:02d}.jpg"
                path.write_bytes(jpeg)
                paths.append(str(path))
            formatted = apply_chat_template(
                self._processor, self._config, prompt_text, num_images=n)
            result = generate(self._model, self._processor, formatted,
                              image=paths, max_tokens=self.max_tokens,
                              temperature=0.0, verbose=False)
        elapsed_ms = (time.monotonic() - started) * 1000.0

        text = getattr(result, "text", "") or ""
        self.peak_memory_gb = float(getattr(result, "peak_memory", 0.0) or 0.0)
        metrics = {
            "inference_ms": round(elapsed_ms, 1),
            "vlm_calls": 1,
            "frames_sent": n,
            # Real measurements from the runtime. No invented confidence.
            "completion_tokens": float(getattr(result, "generation_tokens", 0) or 0),
            "tokens_per_s": round(float(getattr(result, "generation_tps", 0.0) or 0.0), 2),
            "peak_memory_gb": round(self.peak_memory_gb, 2),
        }
        trace_path = os.environ.get("FOREMAN_LOCAL_VLM_TRACE")
        if trace_path:
            trace_vlm(trace_path, model=self.model_id, n_images=n,
                      prompt=prompt_text, raw=text, metrics=metrics)
        return fe.parse_window_judgement(text, n), metrics


# ------------------------------------------------------------------ camera

class MacCamera:
    """Frames from the Mac camera, via ffmpeg rather than an OpenCV dependency.

    ffmpeg is already a prerequisite of the Modalix path (it publishes the RTSP
    stream), so reading raw frames from it here adds nothing new to install.
    avfoundation only accepts a rate the device actually advertises and only
    delivers uyvy422; ffmpeg does the conversion to rgb24 for us.
    """

    #: avfoundation only accepts a capture rate the device advertises exactly.
    #: 1280x720 uyvy422 negotiates at 30 here even though 15 is listed, which is
    #: the same quirk scripts/stream-camera.sh documents. Capture at 30 and let
    #: ffmpeg drop to the target rate on output.
    CAPTURE_FPS = 30

    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.proc = None
        self.error: str | None = None

    def start(self) -> None:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            self.error = "ffmpeg not found (brew install ffmpeg)"
            return
        cmd = [
            # -nostdin is load-bearing. ffmpeg reads stdin for interactive keys
            # ("q" to quit). When this process group is not the terminal's
            # foreground group - which is exactly what job control in the
            # launcher creates - that read raises SIGTTIN and STOPS ffmpeg and
            # every sibling in its group, including this server. The socket
            # stays bound so the port still looks open while nothing answers.
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", str(self.CAPTURE_FPS),
            "-pixel_format", "uyvy422", "-video_size", f"{self.width}x{self.height}",
            "-i", f"{self.device}:none",
            "-r", str(self.fps),
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ]
        try:
            # stdin from /dev/null as well as -nostdin: belt and braces, so no
            # child in this tree can ever touch the controlling terminal.
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=0)
        except OSError as exc:
            self.error = f"could not start ffmpeg: {exc}"

    def _read_exact(self, n: int) -> bytes | None:
        """Exactly n bytes, or None at end of stream.

        A pipe read returns whatever happens to be buffered - typically 64 KB,
        never a whole 2.7 MB frame - so a single read() that comes up short means
        "more is coming", not EOF. Treating short reads as EOF is what silently
        produced zero frames.
        """
        chunks, remaining = [], n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def frames(self):
        """Yield rgb24 frames as numpy arrays until the pipe closes."""
        import numpy as np
        if self.proc is None or self.proc.stdout is None:
            return
        n = self.width * self.height * 3
        while True:
            raw = self._read_exact(n)
            if raw is None:
                err = b""
                if self.proc.stderr is not None:
                    with contextlib.suppress(Exception):
                        err = self.proc.stderr.read() or b""
                if err:
                    self.error = err.decode(errors="replace").strip()[:300]
                return
            yield np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

    def stop(self) -> None:
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.terminate()
                self.proc.wait(timeout=3)
            self.proc = None


def encode_jpeg(rgb, quality: int = 80) -> bytes:
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def capture_loop(state: dict) -> None:
    """Read the camera, detect on every frame, publish, and fill the window.

    Deliberately the same shape as the Modalix pipeline: detections on every
    frame because they are the temporal grounding signal, JPEGs only every Nth
    because encoding is the expensive part.
    """
    args = state["args"]
    cam: MacCamera = state["camera"]
    detector: LocalDetector = state["detector"]
    buf: fe.EvidenceBuffer = state["buffer"]

    cam.start()
    if cam.error:
        print(f"[local-edge] camera unavailable: {cam.error}", flush=True)
        return
    print(f"[local-edge] camera {args.camera} at {args.width}x{args.height}@{args.fps}", flush=True)

    detector_ok = detector.available()
    if not detector_ok:
        print("[local-edge] detector unavailable - frames will stream without boxes", flush=True)

    for frame_id, rgb in enumerate(cam.frames(), start=1):
        detections: list[dict] = []
        if detector_ok:
            try:
                detections = detector.detect(rgb)
            except (Unavailable, RuntimeError, ValueError) as exc:
                detector_ok = False
                print(f"[local-edge] detector stopped: {exc}", flush=True)
        jpeg = None
        if frame_id % max(1, args.jpeg_every) == 0:
            with contextlib.suppress(Exception):
                jpeg = encode_jpeg(rgb)
        ts = time.time()
        # Kept only so the debug endpoint can re-run the detector on exactly the
        # frame the operator is looking at. Never used by the inspection path.
        state["last_rgb"] = rgb
        buf.add(fe.FrameRecord(frame_id=frame_id, ts=ts, detections=detections,
                               jpeg=jpeg, sharpness=0.0), detector._last_ms)
        fe.publish({"frame_id": frame_id, "ts": ts, "detections": detections})
    print(f"[local-edge] camera stream ended{': ' + cam.error if cam.error else ''}", flush=True)


# ------------------------------------------------------------------ server

def make_handler(state: dict):
    detector: LocalDetector = state["detector"]
    asr: LocalASR = state["asr"]
    vlm: LocalVLM = state["vlm"]
    buf: fe.EvidenceBuffer = state["buffer"]
    args = state["args"]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet; the launcher prints what matters
            pass

        def _json(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _unavailable(self, exc: Unavailable, cap: Capability):
            return self._json(503, {"error": str(exc), "backend": BACKEND,
                                    "capability": cap.name, "hint": cap.install_hint})

        def do_GET(self):
            route, _, query = self.path.partition("?")
            if route == "/health":
                stats = buf.stats()
                rec = buf.latest()
                return self._json(200, {
                    "ok": True,
                    "backend": BACKEND,
                    # Said plainly and in the payload itself, not only in a log.
                    "notice": "Local development backend. Results are NOT Modalix "
                              "hardware results and must not be quoted as such.",
                    "models": {"detector": Path(args.detector or "").name or "(none)",
                               "vlm": args.vlm_model, "asr": args.asr_model},
                    "capabilities": {c.name: c.status() for c in (detector, asr, vlm)},
                    "frame_id": rec.frame_id if rec else 0,
                    "last_frame_age_s": round(time.time() - rec.ts, 2) if rec else None,
                    "detector_ms": stats.get("detector_ms", 0.0),
                    "window_s": args.window_s,
                    "evidence_frames": args.evidence_frames,
                    "buffer": stats,
                })
            if route == "/events":
                return self._events()
            if route == "/detect_debug":
                # Camera Check's debug view only. It re-runs the detector on the
                # latest frame at a lower threshold and returns the result
                # directly. Nothing is written to the evidence buffer, nothing is
                # published on /events, so host/policy.py cannot see any of it and
                # a low debug threshold can never move a PASS/FAIL.
                if not detector.available():
                    return self._unavailable(Unavailable(detector.status()["detail"]), detector)
                raw = state.get("last_rgb")
                if raw is None:
                    return self._json(503, {"error": "no frame yet", "backend": BACKEND})
                try:
                    thr = float((urllib.parse.parse_qs(query).get("threshold") or ["0.20"])[0])
                except ValueError:
                    thr = 0.20
                thr = max(0.01, min(0.95, thr))
                started = time.monotonic()
                dets = detector.detect_debug(raw, thr)
                return self._json(200, {
                    "backend": BACKEND,
                    "debug": True,
                    "threshold": thr,
                    "production_threshold": detector.score_threshold,
                    "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
                    "detections": dets,
                    "note": "Debug output only. These detections do not reach the "
                            "evidence window, the gate or the grounding policy.",
                })
            if route == "/frame.jpg":
                # latest_jpeg, not latest: only every Nth frame is encoded, so the
                # most recent frame usually has no pixels attached.
                jpeg, _ts = buf.latest_jpeg()
                if not jpeg:
                    return self._json(503, {"error": "no frame yet", "backend": BACKEND})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(jpeg)
            return self._json(404, {"error": "not found"})

        def _events(self):
            import queue
            q: queue.Queue = queue.Queue(maxsize=64)
            with fe.SUBSCRIBERS_LOCK:
                fe.SUBSCRIBERS.add(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    try:
                        payload = q.get(timeout=15.0)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with fe.SUBSCRIBERS_LOCK:
                    fe.SUBSCRIBERS.discard(q)

        def _inspect_window(self):
            """Judge a rolling evidence window. Same contract as the Modalix edge.

            Returns evidence, never a final verdict: host/policy.py decides, and
            it can still override the model. The frame selection, ROI crops,
            prompt and reply parsing are all foreman_edge functions, so this
            differs from the board only in where the model runs.

            With no local VLM installed, the detector evidence is still real and
            still returned, and the `vlm` block says plainly that the semantic
            judgement did not happen. Nothing is fabricated.
            """
            try:
                body = json.loads(self._read_body() or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "invalid JSON body"})

            standard = str(body.get("standard", "")).strip()
            required = [str(x) for x in (body.get("required_objects")
                                         or body.get("required") or [])]
            prohibited = [str(x) for x in (body.get("prohibited_objects")
                                           or body.get("prohibited") or [])]
            window_s = float(body.get("window_s") or args.window_s)
            want = int(body.get("num_frames") or args.evidence_frames)
            capture_s = float(body.get("capture_s") or 0.0)
            # Every class any rule needs reported, for the shared window summary
            # the response carries. Per-rule scoping happens in rule_inputs();
            # this list is only for the one summary an operator sees.
            summary_objects = list(required)
            for extra in [(body.get("relation") or {}).get("object")] + [
                    (r.get("relation") or {}).get("object") for r in (body.get("rules") or [])]:
                if extra and str(extra) not in summary_objects:
                    summary_objects.append(str(extra))
            for r in (body.get("rules") or []):
                for c in (r.get("required_objects") or []):
                    if str(c) not in summary_objects:
                        summary_objects.append(str(c))


            t_sel = time.monotonic()
            if capture_s > 0:
                # Manual mode, same helper the Modalix edge uses: judge what
                # happens from now on, not what is already buffered.
                frames, cap_start, cap_end = fe.capture_window(buf, capture_s)
                capture = {"mode": "manual", "requested_s": capture_s,
                           "start_ts": cap_start, "end_ts": cap_end,
                           "duration_s": round(cap_end - cap_start, 3)}
            else:
                frames = buf.window(window_s)
                capture = {"mode": "auto", "requested_s": window_s}
            if not frames:
                return self._json(503, {"error": "no frames buffered yet", "backend": BACKEND})
            if time.time() - frames[-1].ts > 5.0:
                return self._json(503, {"error": "camera frames are stale", "backend": BACKEND})
            selected = fe.select_representative(frames, required, want)
            if not selected:
                return self._json(503, {"error": "no encoded frames in the window",
                                        "backend": BACKEND})
            selection_ms = (time.monotonic() - t_sel) * 1000.0
            summary = fe.detector_summary(frames, summary_objects, prohibited)
            t0 = frames[0].ts

            # Same generic ROI idea as the board: the detector grounds the parent
            # object and a padded close-up gives the model detail for sub-parts it
            # has no class for (a cap, a label).
            roi_label, roi_images, roi_meta = None, [], []
            if required and not args.no_roi:
                roi_label, _ = fe.pick_roi_class(frames, required)
            if roi_label:
                fallback = fe.stable_bbox(frames, roi_label)
                for f in selected:
                    bbox, best = None, 0.0
                    for det in f.detections:
                        if det.get("label") == roi_label and float(det.get("confidence", 0)) > best:
                            best, bbox = float(det["confidence"]), det["bbox"]
                    bbox = bbox or fallback
                    if not bbox or not f.jpeg:
                        continue
                    crop = fe.crop_roi(f.jpeg, bbox, pad=args.roi_pad)
                    if crop:
                        roi_images.append(crop)
                        roi_meta.append({"frame_id": f.frame_id,
                                         "rel_ts": round(f.ts - t0, 3),
                                         "label": roi_label,
                                         "bbox": [round(v, 4) for v in bbox],
                                         "bytes": len(crop)})
                roi_images, roi_meta = roi_images[:args.roi_frames], roi_meta[:args.roi_frames]

            payload = {
                "window": {"start_ts": t0, "end_ts": frames[-1].ts,
                           "duration_s": round(frames[-1].ts - t0, 3),
                           "total_frames": len(frames),
                           "image_frames": sum(1 for f in frames if f.jpeg)},
                "detector_summary": summary,
                "capture": capture | {"detector_frames": len(frames)},
                "roi": {"label": roi_label, "frames": roi_meta} if roi_images else None,
                "selected": [{
                    "frame_id": f.frame_id, "ts": f.ts,
                    "rel_ts": round(f.ts - t0, 3),
                    "sharpness": round(f.sharpness, 1),
                    "detections": f.detections,
                    "jpeg_b64": base64.b64encode(f.jpeg).decode() if f.jpeg else "",
                } for f in selected],
                "backend": BACKEND,
            }

            if not vlm.available():
                payload["capabilities"] = {"vlm": False}
                payload["vlm"] = {
                    "verdict": "unclear",
                    "reason": "Local VLM is not installed, so the relationship in "
                              "this standard was not judged. Detector evidence above "
                              "is real and was still applied.",
                    "evidence": [], "missing_evidence": ["local vision-language model"],
                    "per_frame": [None] * len(selected),
                }
                payload["metrics"] = {"inference_ms": 0.0, "vlm_calls": 0,
                                      "frames_sent": len(selected),
                                      "selection_ms": round(selection_ms, 1)}
                return self._json(200, payload)

            wide = [f.jpeg for f in selected if f.jpeg]
            if roi_images:
                wide = wide[:max(1, args.evidence_frames - len(roi_images) + 1)]
            # One judgement per rule, over the SAME `wide` frames. The camera is
            # never re-opened and the detector never re-runs: this loop is the
            # only thing that repeats, which is the trade the design accepts to
            # keep the rules from contaminating each other in one merged prompt.
            specs = fe.rule_inputs(body, frames, standard, required, prohibited,
                                    body.get("relation"))
            vlms, total_ms, calls = [], 0.0, 0
            try:
                for spec in specs:
                    j, m = vlm.judge_window(
                        wide + roi_images, spec["standard"], spec["summary"],
                        roi_label=roi_label if roi_images else None,
                        roi_count=len(roi_images), confirmed=spec["confirmed"],
                        relation=spec["relation"])
                    vlms.append(j)
                    total_ms += float(m.get("inference_ms") or 0.0)
                    calls += 1
                    metrics = m
                # ROI crops are extra views of one moment, not extra moments.
                # Reporting them as agreeing frames would inflate "6/6 frames in
                # agreement" with images that carry no new temporal evidence.
                metrics["temporal_frames"] = len(wide)
                metrics["inference_ms"] = round(total_ms, 1)
                metrics["vlm_calls"] = calls
                metrics["rules_judged"] = len(vlms)
                judgement = vlms[0]
            except (Unavailable, RuntimeError, ValueError, OSError) as exc:
                return self._json(502, {"error": f"local VLM failed: {exc}",
                                        "backend": BACKEND})
            metrics["selection_ms"] = round(selection_ms, 1)
            payload["vlm"] = judgement
            payload["vlms"] = vlms
            payload["metrics"] = metrics
            payload["capabilities"] = {"vlm": True}
            return self._json(200, payload)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_POST(self):
            route, _, query = self.path.partition("?")
            if route == "/inspect":
                return self._inspect_window()
            if route == "/inspect_single":
                return self._json(503, {
                    "error": "the single-frame fallback is not implemented on the local "
                             "backend; temporal inspection is the supported path",
                    "backend": BACKEND, "capability": "vlm"})
            if route == "/transcribe":
                body = self._read_body()
                if not body:
                    return self._json(400, {"error": "empty upload"})
                audio, filename = fe.extract_uploaded_file(
                    body, self.headers.get("Content-Type", ""))
                if not audio:
                    return self._json(400, {"error": "no file part in upload"})
                mode = (urllib.parse.parse_qs(query).get("language") or ["auto"])[0]
                try:
                    return self._json(200, asr.transcribe(audio, filename, mode))
                except Unavailable as exc:
                    return self._unavailable(exc, asr)
            return self._json(404, {"error": "not found"})

    return Handler


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    env = lambda k, d: os.environ.get(k, d)  # noqa: E731
    p.add_argument("--port", type=int, default=int(env("FOREMAN_EDGE_PORT", "8100")))
    # Loopback by default. The Modalix edge binds 0.0.0.0 because the host talks
    # to it across the DevKit link; this one runs on the same machine as the
    # host, so there is no reason to expose a development model server to the
    # local network.
    p.add_argument("--bind", default=env("FOREMAN_LOCAL_BIND", "127.0.0.1"))
    p.add_argument("--camera", default=env("FOREMAN_LOCAL_CAMERA", "0"))
    p.add_argument("--detector", default=env("FOREMAN_LOCAL_DETECTOR", ""))
    p.add_argument("--labels", default=env("FOREMAN_LABELS", str(
        Path(__file__).resolve().parents[1] / "assets" / "coco.txt")))
    p.add_argument("--vlm-model", default=env("FOREMAN_LOCAL_VLM", "mlx-community/Qwen3-VL-2B-Instruct-4bit"))
    p.add_argument("--asr-model", default=env("FOREMAN_LOCAL_ASR", "small"))
    p.add_argument("--score-threshold", type=float, default=float(env("FOREMAN_SCORE_THRESHOLD", "0.55")))
    p.add_argument("--window-s", type=float, default=float(env("FOREMAN_WINDOW_S", "3.0")))
    p.add_argument("--evidence-frames", type=int, default=int(env("FOREMAN_EVIDENCE_FRAMES", "6")))
    p.add_argument("--capture-s", type=float, default=float(env("FOREMAN_CAPTURE_S", "3.0")))
    p.add_argument("--width", type=int, default=int(env("FOREMAN_WIDTH", "1280")))
    p.add_argument("--height", type=int, default=int(env("FOREMAN_HEIGHT", "720")))
    p.add_argument("--fps", type=int, default=int(env("FOREMAN_FPS", "15")))
    p.add_argument("--jpeg-every", type=int, default=int(env("FOREMAN_JPEG_EVERY", "3")))
    p.add_argument("--no-roi", action="store_true", help="disable ROI close-ups")
    p.add_argument("--roi-frames", type=int, default=int(env("FOREMAN_ROI_FRAMES", "2")))
    p.add_argument("--roi-pad", type=float, default=float(env("FOREMAN_ROI_PAD", "0.25")))
    p.add_argument("--no-camera", action="store_true",
                   help="serve the API without opening the camera")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = fe.load_labels(args.labels) if Path(args.labels).is_file() else []
    state = {
        "args": args,
        "buffer": fe.EvidenceBuffer(window_s=args.window_s,
                                    retain_s=max(args.window_s, args.capture_s) + 2.0),
        "detector": LocalDetector(args.detector, labels, args.score_threshold),
        "asr": LocalASR(args.asr_model),
        "vlm": LocalVLM(args.vlm_model),
        "camera": MacCamera(args.camera, args.width, args.height, args.fps),
    }
    print("=" * 66, flush=True)
    print("  Foreman LOCAL development backend - NOT Modalix hardware", flush=True)
    print("  Results are labelled backend=local everywhere, including the", flush=True)
    print("  audit trail. Never quote a latency measured here as a Modalix", flush=True)
    print("  benchmark.", flush=True)
    print("=" * 66, flush=True)
    for cap in (state["detector"], state["asr"], state["vlm"]):
        st = cap.status()
        mark = "ready" if st["available"] else "unavailable"
        print(f"  {cap.name:<9} {mark}" + ("" if st["available"] else f"  - {st['detail']}"), flush=True)
    print(f"[local-edge] API on http://{args.bind}:{args.port}", flush=True)

    server = ThreadingHTTPServer((args.bind, args.port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not args.no_camera:
        threading.Thread(target=capture_loop, args=(state,), daemon=True).start()
    # Load the ASR weights now rather than on the first spoken standard, so the
    # first recording is not several seconds slower than every one after it.
    if state["asr"].available():
        threading.Thread(target=state["asr"].warm, daemon=True).start()
    if state["vlm"].available():
        threading.Thread(target=state["vlm"].warm, daemon=True).start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[local-edge] stopping", flush=True)
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
