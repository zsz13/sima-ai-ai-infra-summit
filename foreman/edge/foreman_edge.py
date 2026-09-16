#!/usr/bin/env python3
"""Foreman edge agent — the [MODALIX] half of the system.

Runs on the Modalix DevKit inside the board's PyNeat environment.

  SEE        YOLO26 object detection on the MLA, continuously, from RTSP.
             Frames and boxes also go to Neat Insight for the live view.
  UNDERSTAND On request, judges the current frame against a stated standard
             using a vision-language model on the MLA.
  ASR        On request, transcribes speech with Whisper on the MLA.

The VLM and Whisper are reached over the Neat GenAI server's OpenAI-compatible
HTTP API on localhost (tutorial 021), which is the pattern the shipped
detection-to-vlm-assistant example uses: separate processes, one MLA.

Only the standard library is used for HTTP, so this needs nothing installed in
the PyNeat venv beyond what Neat already provides (pyneat, numpy, cv2).

Run:
    source ~/pyneat/bin/activate
    python3 foreman_edge.py --source rtsp://<sdk-host>:8554/src1 \
                            --model /path/to/yolo26m-det.tar.gz \
                            --labels coco.txt \
                            --insight-host <sdk-host>
"""

# This module runs only on the Modalix DevKit. pyneat, cv2 and numpy exist in the
# board's PyNeat environment and cannot resolve on a development host, so those
# checks are disabled here rather than left to bury real findings.
# pyright: reportMissingImports=false, reportOptionalMemberAccess=false
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import queue
import signal
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

pyneat = None  # type: ignore
cv2 = None  # type: ignore
np = None  # type: ignore


def load_runtime_dependencies() -> None:
    """Import board-only modules with an error a human can act on."""
    global pyneat, cv2, np
    try:
        import cv2 as cv2_module
        import numpy as np_module
        import pyneat as pyneat_module
    except ImportError as exc:
        raise SystemExit(
            f"missing runtime dependency: {exc}\n"
            "This agent runs on the Modalix DevKit inside the PyNeat environment.\n"
            "Try:  source ~/pyneat/bin/activate"
        ) from exc
    pyneat, cv2, np = pyneat_module, cv2_module, np_module


# ---------------------------------------------------------------- state

class Latest:
    """The most recent frame and detections, shared with the HTTP threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frame_jpeg: bytes | None = None
        self.detections: list[dict] = []
        self.frame_id: int = 0
        self.ts: float = 0.0
        self.detector_ms: float = 0.0

    def set(self, jpeg: bytes | None, detections: list[dict], frame_id: int, detector_ms: float) -> None:
        with self._lock:
            if jpeg is not None:
                self.frame_jpeg = jpeg
            self.detections = detections
            self.frame_id = frame_id
            self.ts = time.time()
            self.detector_ms = detector_ms

    def snapshot(self) -> tuple[bytes | None, list[dict], int, float, float]:
        with self._lock:
            return self.frame_jpeg, list(self.detections), self.frame_id, self.ts, self.detector_ms


LATEST = Latest()
SUBSCRIBERS: set[queue.Queue] = set()
SUBSCRIBERS_LOCK = threading.Lock()
STOPPING = threading.Event()


def publish(event: dict) -> None:
    payload = json.dumps(event, separators=(",", ":"))
    with SUBSCRIBERS_LOCK:
        targets = list(SUBSCRIBERS)
    for q in targets:
        # A slow consumer must never stall the pipeline.
        with contextlib.suppress(queue.Full):
            q.put_nowait(payload)


# ------------------------------------------------------- GenAI over HTTP

class GenAI:
    """Client for the Neat GenAI server (OpenAI-compatible) on this board."""

    def __init__(self, base_url: str, vlm_model: str, asr_model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.vlm_model = vlm_model
        self.asr_model = asr_model
        self.timeout = timeout

    def _post_json(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def judge(self, jpeg: bytes, standard: str) -> tuple[str, str, dict]:
        """Ask the VLM whether this frame meets `standard`. Runs on the MLA."""
        # The model is asked for a boolean AND a verdict string. Asking twice in
        # one call costs nothing and catches the observed failure mode where the
        # reason says "is not wearing a hard hat" while the verdict says "pass";
        # parse_verdict downgrades any disagreement to "unclear".
        # Measured on this board: 24/24 correct over 6 standards x 4 repetitions,
        # 0 self-inconsistencies, median 1423 ms.
        prompt = (
            "You are a strict visual inspection system.\n\n"
            f"REQUIREMENT: {standard}\n\n"
            "Decide whether the image visibly satisfies the requirement in full. "
            "If any part of it is missing, absent, or not visible, it is not satisfied.\n\n"
            "Reply with JSON only, no other text:\n"
            '{"meets_requirement": true or false, '
            '"verdict": "pass" or "fail", '
            '"reason": "<one short sentence>"}\n'
            'Set "verdict" to "pass" only when "meets_requirement" is true.'
        )
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        started = time.monotonic()
        result = self._post_json("/v1/chat/completions", {
            "model": self.vlm_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "max_tokens": 160,
            "temperature": 0.0,
        })
        elapsed_ms = (time.monotonic() - started) * 1000.0

        text = ""
        with contextlib.suppress(KeyError, IndexError, TypeError):
            text = result["choices"][0]["message"]["content"]
        verdict, reason = parse_verdict(text)

        usage = result.get("usage") or {}
        metrics = {"inference_ms": round(elapsed_ms, 1)}
        for src, dst in (("ttft_ms", "ttft_ms"), ("tokens_per_second", "tokens_per_s"),
                         ("completion_tokens", "completion_tokens")):
            if src in usage:
                metrics[dst] = float(usage[src])
        return verdict, reason, metrics

    def transcribe(self, audio: bytes, filename: str) -> dict:
        """Whisper on the MLA, via the server's /v1/audio/transcriptions."""
        body, content_type = encode_multipart({
            "model": self.asr_model,
            "language": "auto",
        }, "file", filename, audio)
        req = urllib.request.Request(
            self.base_url + "/v1/audio/transcriptions",
            data=body, headers={"Content-Type": content_type},
        )
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return {
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "no_speech_prob": float(result.get("no_speech_prob", 0.0) or 0.0),
            "metrics": {"inference_ms": round(elapsed_ms, 1)},
        }


def parse_verdict(text: str) -> tuple[str, str]:
    """Pull a verdict out of the model's reply, tolerating code fences and prose.

    The model returns both a boolean and a verdict string. They normally agree;
    when they do not, the answer is not trustworthy and becomes "unclear" rather
    than a confident verdict a human might act on.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw[3:]
        raw = raw.removeprefix("json").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            verdict = str(obj.get("verdict", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip() or "No reason given."
            meets = obj.get("meets_requirement")
            if verdict in {"pass", "fail"} and isinstance(meets, bool):
                if (meets is True) == (verdict == "pass"):
                    return verdict, reason
                return "unclear", (
                    f"The model contradicted itself, so this needs a human. {reason}")
            if verdict in {"pass", "fail", "unclear"}:
                return verdict, reason
            if isinstance(meets, bool):
                return ("pass" if meets else "fail"), reason

    # The model answered in prose. Fall back to a keyword read rather than
    # inventing a verdict, and say so.
    low = raw.lower()
    if "fail" in low and "pass" not in low:
        return "fail", raw[:240] or "Model replied in prose."
    if "pass" in low and "fail" not in low:
        return "pass", raw[:240] or "Model replied in prose."
    return "unclear", (raw[:240] or "Model returned no usable answer.")


def encode_multipart(fields: dict, file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """Minimal multipart/form-data encoder, so no extra dependency is needed."""
    boundary = "----foreman" + os.urandom(12).hex()
    out = bytearray()
    for key, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    out += f"--{boundary}\r\n".encode()
    out += (f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n').encode()
    out += b"Content-Type: application/octet-stream\r\n\r\n"
    out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


# ----------------------------------------------------------- detection

def parse_boxes(payload: bytes, img_w: int, img_h: int) -> list[dict]:
    """Decode Neat's BBOX tensor payload: uint32 count, then (x,y,w,h,score,class)."""
    if len(payload) < 4:
        return []
    count = struct.unpack_from("<I", payload, 0)[0]
    max_boxes = (len(payload) - 4) // 24
    if count > max_boxes:
        raise RuntimeError("bbox header exceeds payload count")
    boxes, offset = [], 4
    for _ in range(count):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        boxes.append({
            "x1": max(0.0, min(float(x), float(img_w))),
            "y1": max(0.0, min(float(y), float(img_h))),
            "x2": max(0.0, min(float(x + w), float(img_w))),
            "y2": max(0.0, min(float(y + h), float(img_h))),
            "score": float(score),
            "class_id": int(class_id),
        })
    return boxes


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def attr_or_call(obj, name: str, default=None):
    """Several Neat members are methods in C++ but plain attributes in pyneat."""
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def first_tensor(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field in getattr(sample, "fields", []):
        tensor = first_tensor(field)
        if tensor is not None:
            return tensor
    return None


def bbox_payload(sample) -> bytes:
    tensor = first_tensor(sample)
    if tensor is None:
        raise RuntimeError("no tensor in detection sample")
    return tensor.copy_payload_bytes()


def nv12_to_bgr(payload, width: int, height: int, planar_i420: bool = False):
    """Convert a decoded NV12/I420 payload to BGR, honouring plane padding.

    The decoder pads the luma plane's *height* to a 64-row boundary: a 1280x720
    frame arrives as 1,474,560 bytes, not 1,382,400, because Y occupies 768 rows
    and the chroma plane 384. Reshaping naively to (height * 3 // 2, width) reads
    chroma from the wrong offset, which shows up as a green band and a ghosted
    second copy of the image.
    """
    arr = np.frombuffer(payload, dtype=np.uint8)
    exact = width * height * 3 // 2
    if arr.size == exact:
        aligned_h = height
    else:
        aligned_h = int(arr.size // (width * 1.5))
        if aligned_h < height or aligned_h * width * 3 // 2 > arr.size:
            raise ValueError(
                f"cannot map {arr.size} bytes to a {width}x{height} NV12 frame")

    y_size = aligned_h * width
    y = arr[:y_size].reshape(aligned_h, width)[:height, :]
    uv = arr[y_size:y_size + (aligned_h // 2) * width]
    uv = uv.reshape(aligned_h // 2, width)[:height // 2, :]
    stacked = np.vstack((y, uv))
    code = cv2.COLOR_YUV2BGR_I420 if planar_i420 else cv2.COLOR_YUV2BGR_NV12
    return cv2.cvtColor(np.ascontiguousarray(stacked), code)


def frame_to_jpeg(tensor, quality: int) -> bytes | None:
    """Decoded NV12/I420 tensor -> JPEG bytes, for VLM input and evidence."""
    try:
        width, height = tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
        if tensor.is_nv12() or tensor.is_i420():
            bgr = nv12_to_bgr(tensor.copy_payload_bytes(), width, height,
                              planar_i420=tensor.is_i420())
        else:
            frame = np.asarray(tensor.to_numpy(copy=True))
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3:
                return None
            bgr = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None
    except Exception as exc:
        print(f"[foreman-edge] frame encode failed: {exc}", file=sys.stderr, flush=True)
        return None


# ------------------------------------------------------------ pipeline

class Pipeline:
    def __init__(self, args, labels: list[str]):
        self.args = args
        self.labels = labels
        self.frame_w = 0
        self.frame_h = 0
        self.fps = 0

    def build(self):
        a = self.args
        self.frame_w, self.frame_h, self.fps = probe_source(
            a.source, a.width, a.height, a.fps)
        if self.fps <= 0 or self.frame_w <= 0 or self.frame_h <= 0:
            raise RuntimeError("could not resolve source geometry; pass --width/--height/--fps")

        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
        opt.preprocess.input_max_width = self.frame_w
        opt.preprocess.input_max_height = self.frame_h
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = a.score_threshold
        opt.nms_iou_threshold = a.nms_iou
        opt.top_k = a.max_detections
        self.model = pyneat.Model(a.model, opt)

        source = pyneat.groups.rtsp_decoded_input(self._source_options())

        sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
            self.frame_w, self.frame_h, self.fps)
        sender_opt.host = a.insight_host
        sender_opt.channel = a.channel
        sender_opt.video_port_base = a.video_port
        sender_opt.encoder.bitrate_kbps = a.bitrate_kbps

        video_graph = pyneat.Graph("video")
        video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_opt))

        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), self.model)

        detections_graph = pyneat.Graph("detections")
        detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

        frame_graph = pyneat.Graph("frame")
        frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

        # Insight correlates the RTP timestamp with the metadata timestamp, so the
        # encoder and the detections must stay in one Run, on one GStreamer timeline.
        branch = pyneat.graphs.branch("source", ["video", "model", "frame"])
        joined = pyneat.graphs.combine(
            ["frame", "detections"], "inspection", pyneat.CombinePolicy.ByFrame)

        graph = pyneat.Graph()
        graph.connect(source, branch)
        graph.connect(branch, video_graph)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections_graph)
        graph.connect(branch, frame_graph)
        graph.connect(frame_graph, joined)
        graph.connect(detections_graph, joined)

        if self.args.profile:
            print(f"Backend:\n{graph.describe_backend()}", flush=True)

        run_options = pyneat.RunOptions()
        run_options.preset = pyneat.RunPreset.Realtime
        run_options.queue_depth = 3
        run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        run_options.output_memory = pyneat.OutputMemory.ZeroCopy
        self.run = graph.build(run_options)

        meta_opt = pyneat.MetadataSenderOptions()
        meta_opt.host = a.insight_host
        meta_opt.channel = a.channel
        meta_opt.metadata_port_base = a.metadata_port
        self.metadata_sender = pyneat.MetadataSender(meta_opt)

        print(f"[foreman-edge] source={a.source} {self.frame_w}x{self.frame_h}@{self.fps} "
              f"insight={a.insight_host} "
              f"video={attr_or_call(sender_opt, 'video_port', a.video_port + a.channel)} "
              f"metadata={attr_or_call(self.metadata_sender, 'metadata_port', a.metadata_port + a.channel)} "
              f"channel={a.channel}", flush=True)

    def _source_options(self):
        """RTSP source options, mirroring the shipped single-stream example."""
        a = self.args
        opt = pyneat.RtspDecodedInputOptions()
        opt.url = a.source
        opt.latency_ms = a.latency_ms
        opt.tcp = True
        opt.insert_queue = True
        opt.decoder_name = "decoder"
        opt.decoder_raw_output = True
        opt.source_fps = self.fps
        if a.codec == "h264":
            opt.codec = pyneat.RtspCodec.H264
            opt.payload_type = 96
            opt.auto_caps_from_stream = True
            opt.fallback_h264_width = self.frame_w
            opt.fallback_h264_height = self.frame_h
        elif a.codec == "h265":
            opt.codec = pyneat.RtspCodec.H265
            opt.payload_type = 96
            opt.auto_caps_from_stream = True
            opt.dec_width = self.frame_w
            opt.dec_height = self.frame_h
        else:
            opt.codec = pyneat.RtspCodec.MJPEG
            opt.mjpeg_payload_type = 26
            opt.dec_width = self.frame_w
            opt.dec_height = self.frame_h

        caps = opt.output_caps
        caps.enable = True
        caps.format = pyneat.Format.NV12
        caps.width = self.frame_w
        caps.height = self.frame_h
        caps.fps = self.fps
        caps.memory = pyneat.CapsMemory.Any
        return opt

    def pump(self) -> None:
        """Pull samples until stopped. Runs on its own thread."""
        every = max(1, self.args.jpeg_every)
        n = 0
        while not STOPPING.is_set():
            try:
                sample = self.run.pull("inspection", timeout_ms=1000)
            except Exception as exc:
                if not STOPPING.is_set():
                    print(f"[foreman-edge] pull failed: {exc}", file=sys.stderr, flush=True)
                    time.sleep(0.5)
                continue
            if sample is None:
                continue
            n += 1
            started = time.monotonic()

            detections: list[dict] = []
            try:
                det_field = find_field(sample, "detections")
                boxes = parse_boxes(bbox_payload(det_field if det_field is not None else sample),
                                    self.frame_w, self.frame_h)
                detections = [{
                    "label": self.labels[b["class_id"]]
                    if 0 <= b["class_id"] < len(self.labels) else "unknown",
                    "confidence": b["score"],
                    # normalised for the Mac; Insight gets pixels below
                    "bbox": [b["x1"] / self.frame_w, b["y1"] / self.frame_h,
                             b["x2"] / self.frame_w, b["y2"] / self.frame_h],
                    "track_id": None,
                    "_px": [b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]],
                } for b in boxes]
            except Exception as exc:
                print(f"[foreman-edge] box decode failed: {exc}", file=sys.stderr, flush=True)

            jpeg = None
            if n % every == 0:
                frame_field = find_field(sample, "frame")
                tensor = first_tensor(frame_field if frame_field is not None else sample)
                if tensor is not None:
                    jpeg = frame_to_jpeg(tensor, self.args.jpeg_quality)

            detector_ms = (time.monotonic() - started) * 1000.0
            frame_id = int(getattr(sample, "frame_id", n) or n)
            LATEST.set(jpeg, detections, frame_id, detector_ms)

            self._send_metadata(sample, detections)
            publish({
                "frame_id": frame_id,
                "ts": time.time(),
                "detections": [{k: v for k, v in d.items() if k != "_px"} for d in detections],
            })

    def _send_metadata(self, sample, detections: list[dict]) -> None:
        objects = [{
            "id": f"obj_{i}",
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox": [float(v) for v in d["_px"]],
        } for i, d in enumerate(detections, start=1)]
        pts = getattr(sample, "pts_ns", -1)
        timestamp_ms = int(pts // 1_000_000) if pts and pts >= 0 else -1
        frame_id = getattr(sample, "frame_id", -1)
        try:
            self.metadata_sender.send_metadata(
                "object-detection",
                json.dumps({"objects": objects}, separators=(",", ":")),
                timestamp_ms,
                str(frame_id) if frame_id >= 0 else "",
            )
        except Exception as exc:
            print(f"[foreman-edge] metadata send failed: {exc}", file=sys.stderr, flush=True)


def probe_source(url: str, width: int, height: int, fps: int) -> tuple[int, int, int]:
    """Resolve stream geometry.

    The validated DevKit image has **no ffprobe and no ffmpeg** (only
    gst-launch-1.0), so geometry cannot be probed on the board. Pass
    --width/--height/--fps explicitly; scripts/demo.sh probes them on the Mac,
    where ffprobe does exist, and forwards them. ffprobe is still used when it
    happens to be available, so the agent also runs on a developer host.
    """
    if width > 0 and height > 0 and fps > 0:
        return width, height, fps

    import shutil
    import subprocess
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "stream geometry is unknown and ffprobe is not available on this host.\n"
            "Pass --width, --height and --fps explicitly "
            "(scripts/demo.sh probes them on the Mac and forwards them)."
        )
    cmd = ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate",
           "-of", "json", "-i", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=25, check=True).stdout
        stream = json.loads(out)["streams"][0]
        probed_w, probed_h = int(stream["width"]), int(stream["height"])
        rate = stream.get("avg_frame_rate", "0/1")
        num, _, den = rate.partition("/")
        probed_fps = int(round(int(num) / int(den))) if den and int(den) else 0
    except Exception as exc:
        raise RuntimeError(f"could not probe {url}: {exc}") from exc
    return (width or probed_w, height or probed_h, fps or probed_fps)


# ---------------------------------------------------------------- HTTP

def make_handler(genai: GenAI, args):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # quiet; the pipeline owns stdout
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                _, detections, frame_id, ts, detector_ms = LATEST.snapshot()
                return self._json(200, {
                    "ok": True,
                    "models": {"detector": args.model.rsplit("/", 1)[-1],
                               "vlm": args.vlm_model, "asr": args.asr_model},
                    "frame_id": frame_id, "last_frame_age_s": round(time.time() - ts, 2) if ts else None,
                    "detector_ms": round(detector_ms, 2),
                })
            if self.path == "/events":
                return self._events()
            if self.path == "/frame.jpg":
                jpeg, *_ = LATEST.snapshot()
                if jpeg is None:
                    return self._json(503, {"error": "no frame yet"})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                return self.wfile.write(jpeg)
            return self._json(404, {"error": "not found"})

        def _events(self):
            q: queue.Queue = queue.Queue(maxsize=8)
            with SUBSCRIBERS_LOCK:
                SUBSCRIBERS.add(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while not STOPPING.is_set():
                    try:
                        payload = q.get(timeout=10.0)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with SUBSCRIBERS_LOCK:
                    SUBSCRIBERS.discard(q)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def do_POST(self):
            if self.path == "/inspect":
                return self._inspect()
            if self.path == "/transcribe":
                return self._transcribe()
            return self._json(404, {"error": "not found"})

        def _inspect(self):
            try:
                body = json.loads(self._read_body() or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            standard = str(body.get("standard", "")).strip()
            if not standard:
                return self._json(400, {"error": "standard is required"})

            jpeg, _, _, ts, _ = LATEST.snapshot()
            if jpeg is None:
                return self._json(503, {"error": "no frame available from the camera yet"})
            if time.time() - ts > 5.0:
                return self._json(503, {"error": "camera frames are stale; check the RTSP source"})

            try:
                verdict, reason, metrics = genai.judge(jpeg, standard)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                return self._json(502, {"error": f"vision-language model unavailable: {exc}"})

            return self._json(200, {
                "verdict": verdict,
                "reason": reason,
                "evidence_jpeg_b64": base64.b64encode(jpeg).decode(),
                "metrics": metrics,
            })

        def _transcribe(self):
            body = self._read_body()
            if not body:
                return self._json(400, {"error": "empty upload"})
            audio, filename = extract_uploaded_file(body, self.headers.get("Content-Type", ""))
            if not audio:
                return self._json(400, {"error": "no file part in upload"})
            try:
                return self._json(200, genai.transcribe(audio, filename))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                return self._json(502, {"error": f"speech recognition unavailable: {exc}"})

    return Handler


def extract_uploaded_file(body: bytes, content_type: str) -> tuple[bytes, str]:
    """Pull the single file part out of a multipart body, without cgi (removed in 3.13)."""
    marker = "boundary="
    if marker not in content_type:
        return body, "speech.wav"  # raw upload
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    sep = ("--" + boundary).encode()
    for part in body.split(sep):
        head, _, data = part.partition(b"\r\n\r\n")
        if b"filename=" not in head:
            continue
        filename = "speech.wav"
        try:
            after = head.decode("utf-8", "replace").split("filename=", 1)[1]
            filename = after.split('"')[1] if '"' in after else filename
        except (IndexError, ValueError):
            pass
        return data.rstrip(b"\r\n--"), filename
    return b"", "speech.wav"


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    env = os.environ.get
    p = argparse.ArgumentParser(description="Foreman edge agent (Modalix)")
    p.add_argument("--source", default=env("FOREMAN_SOURCE", ""),
                   help="RTSP URL, e.g. rtsp://<sdk-host>:8554/src1")
    p.add_argument("--model", default=env("FOREMAN_MODEL", ""),
                   help="compiled detector .tar.gz")
    p.add_argument("--labels", default=env("FOREMAN_LABELS", ""), help="label file, one per line")
    p.add_argument("--insight-host", default=env("FOREMAN_INSIGHT_HOST", "127.0.0.1"))
    p.add_argument("--channel", type=int, default=int(env("FOREMAN_CHANNEL", "0")))
    p.add_argument("--video-port", type=int, default=9000)
    p.add_argument("--metadata-port", type=int, default=9100)
    p.add_argument("--bitrate-kbps", type=int, default=1000)
    p.add_argument("--genai-url", default=env("FOREMAN_GENAI_URL", "http://127.0.0.1:9998"))
    p.add_argument("--vlm-model", default=env("FOREMAN_VLM_MODEL", "vlm"))
    p.add_argument("--asr-model", default=env("FOREMAN_ASR_MODEL", "asr"))
    p.add_argument("--listen", default=env("FOREMAN_LISTEN", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(env("FOREMAN_EDGE_PORT", "8100")))
    p.add_argument("--score-threshold", type=float, default=0.52)
    p.add_argument("--nms-iou", type=float, default=0.45)
    p.add_argument("--max-detections", type=int, default=20)
    p.add_argument("--fps", type=int, default=int(env("FOREMAN_FPS", "0")),
                   help="source fps (required on the DevKit: no ffprobe there)")
    p.add_argument("--width", type=int, default=int(env("FOREMAN_WIDTH", "0")))
    p.add_argument("--height", type=int, default=int(env("FOREMAN_HEIGHT", "0")))
    p.add_argument("--codec", default=env("FOREMAN_CODEC", "h264"),
                   choices=["h264", "h265", "mjpeg"])
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--jpeg-every", type=int, default=3, help="keep 1 in N frames as JPEG")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--profile", action="store_true", help="print the backend pipeline")
    args = p.parse_args(argv)
    missing = [n for n in ("source", "model") if not getattr(args, n)]
    if missing:
        p.error("missing required argument(s): " + ", ".join("--" + m for m in missing))
    return args


def load_labels(path: str) -> list[str]:
    if not path:
        return []
    try:
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]
    except OSError as exc:
        print(f"[foreman-edge] could not read labels: {exc}", file=sys.stderr)
        return []


def main(argv=None) -> int:
    args = parse_args(argv)
    load_runtime_dependencies()

    pipeline = Pipeline(args, load_labels(args.labels))
    pipeline.build()

    pump = threading.Thread(target=pipeline.pump, name="pipeline", daemon=True)
    pump.start()

    genai = GenAI(args.genai_url, args.vlm_model, args.asr_model)
    server = ThreadingHTTPServer((args.listen, args.port), make_handler(genai, args))
    server.daemon_threads = True

    def shutdown(signum, _frame):
        print(f"\n[foreman-edge] signal {signum}, shutting down", flush=True)
        STOPPING.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, shutdown)

    print(f"[foreman-edge] API on http://{args.listen}:{args.port} "
          f"(GenAI at {args.genai_url})", flush=True)
    try:
        server.serve_forever()
    finally:
        STOPPING.set()
        pump.join(timeout=3)
        with contextlib.suppress(Exception):
            pipeline.run.close()
        server.server_close()
        print("[foreman-edge] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
