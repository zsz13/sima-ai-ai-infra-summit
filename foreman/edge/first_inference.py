#!/usr/bin/env python3
"""First real MLA inference on Modalix: YOLO26-n object detection.

A deliberately minimal end-to-end check that the compiled model actually runs on
the accelerator and returns sensible boxes, before any of the Foreman pipeline is
involved. Also reports measured per-inference latency.

Run from the SDK container:
    dk foreman/edge/first_inference.py \
       --model /media/nvme/foreman/models/yolo_26n_mpk.tar.gz \
       --labels /media/nvme/foreman/coco.txt \
       --image /workspace-rsync/assets/images/000000081061.jpg \
       --iterations 50
"""

# pyneat, cv2 and numpy only exist in the DevKit's PyNeat environment.
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import statistics
import struct
import sys
import time
from pathlib import Path

try:
    import pyneat
except ImportError:
    sys.exit(
        "pyneat is not importable. This script runs on the Modalix DevKit.\n"
        "Run it via `dk`, or on the board with: source ~/pyneat/bin/activate"
    )

import cv2
import numpy as np

SIZE = 640


def build_options() -> pyneat.ModelOptions:
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = SIZE
    opt.preprocess.input_max_height = SIZE
    opt.preprocess.input_max_depth = 3
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = 0.40
    opt.nms_iou_threshold = 0.45
    opt.top_k = 20
    return opt


def load_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"failed to read image: {path}")
    if bgr.shape[0] != SIZE or bgr.shape[1] != SIZE:
        bgr = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def first_tensor(sample):
    """Model.run() may hand back a Sample, a list of Samples, or a Tensor."""
    if sample is None:
        return None
    if isinstance(sample, (list, tuple)):
        for item in sample:
            found = first_tensor(item)
            if found is not None:
                return found
        return None
    if not hasattr(sample, "kind"):
        return sample if hasattr(sample, "copy_payload_bytes") else None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field in getattr(sample, "fields", []):
        found = first_tensor(field)
        if found is not None:
            return found
    return None


def parse_boxes(payload: bytes) -> list[dict]:
    """uint32 count, then (x, y, w, h, score, class_id) as <iiiifi>."""
    if len(payload) < 4:
        return []
    count = struct.unpack_from("<I", payload, 0)[0]
    if count > (len(payload) - 4) // 24:
        raise RuntimeError("bbox header exceeds payload")
    out, offset = [], 4
    for _ in range(count):
        x, y, w, h, score, cid = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        out.append({"x": x, "y": y, "w": w, "h": h, "score": score, "class_id": cid})
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv[1:])

    labels: list[str] = []
    if args.labels and args.labels.is_file():
        labels = [ln.strip() for ln in args.labels.read_text().splitlines() if ln.strip()]

    print(f"model  : {args.model}")
    print(f"image  : {args.image}")

    t0 = time.monotonic()
    model = pyneat.Model(args.model, build_options())
    print(f"loaded : {(time.monotonic() - t0) * 1000:.0f} ms")

    image = load_image(args.image)
    # Image-mode inputs need explicit pixel-format metadata; a bare ndarray is refused.
    tensor_in = pyneat.Tensor.from_numpy(image, copy=True, image_format=pyneat.PixelFormat.RGB)

    # Warm-up is excluded from the reported numbers, and said to be.
    for _ in range(args.warmup):
        model.run([tensor_in], timeout_ms=10000)

    timings: list[float] = []
    sample = None
    for _ in range(args.iterations):
        started = time.monotonic()
        sample = model.run([tensor_in], timeout_ms=10000)
        timings.append((time.monotonic() - started) * 1000.0)

    tensor = first_tensor(sample)
    if tensor is None:
        print("no output tensor", file=sys.stderr)
        return 1
    boxes = parse_boxes(tensor.copy_payload_bytes())

    print(f"\ndetections ({len(boxes)}):")
    for b in boxes:
        name = labels[b["class_id"]] if 0 <= b["class_id"] < len(labels) else f"class_{b['class_id']}"
        print(f"  {name:<16} {b['score']:.3f}  "
              f"[{b['x']},{b['y']} {b['w']}x{b['h']}]")

    timings.sort()
    print(f"\nlatency over {len(timings)} runs, {args.warmup} warm-up excluded:")
    print(f"  median  {statistics.median(timings):7.2f} ms")
    print(f"  min     {timings[0]:7.2f} ms")
    print(f"  max     {timings[-1]:7.2f} ms")
    print(f"  p95     {timings[int(len(timings) * 0.95) - 1]:7.2f} ms")
    print(f"  implied {1000.0 / statistics.median(timings):7.1f} fps single-stream")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
