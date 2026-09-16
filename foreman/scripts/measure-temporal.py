#!/usr/bin/env python3
"""Cost of temporal grounding, measured from the Mac against the real DevKit.

Reports detector throughput idle and under load, VLM latency per inspection, the
number of VLM calls, host-side processing time, and evidence-buffer memory.
"""
import contextlib
import json
import statistics
import threading
import time
import urllib.request

STATE = "http://127.0.0.1:8800/api/state"
EDGE = "http://192.168.2.2:8100"


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def fps_samples(seconds, interval=1.0):
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        with contextlib.suppress(Exception):
            out.append(get(STATE)["fps"])
        time.sleep(interval)
    return [f for f in out if f > 0]


def inspect_once(standard="A person must be visible"):
    body = json.dumps({
        "standard": standard, "required_objects": ["person"],
        "prohibited_objects": [], "window_s": 3.0, "num_frames": 3,
    }).encode()
    req = urllib.request.Request(EDGE + "/inspect", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    return out, (time.monotonic() - t0) * 1000


def stat(xs, unit=""):
    xs = sorted(xs)
    return (f"median {statistics.median(xs):.1f}{unit}  min {xs[0]:.1f}  "
            f"max {xs[-1]:.1f}  n={len(xs)}")


print("warming up...")
fps_samples(6)

print("detector fps, idle (25 s)...")
idle = fps_samples(25)

print("running 10 temporal inspections...")
edge_ms, round_ms, calls, frames_sent, sel_ms, window_frames = [], [], [], [], [], []
for _ in range(10):
    out, rtt = inspect_once()
    m = out["metrics"]
    edge_ms.append(m["inference_ms"])
    round_ms.append(rtt)
    calls.append(m.get("vlm_calls", 1))
    frames_sent.append(m.get("frames_sent", 0))
    sel_ms.append(m.get("selection_ms", 0.0))
    window_frames.append(out["window"]["total_frames"])

print("detector fps, under temporal load (25 s)...")
stop = threading.Event()


def hammer():
    while not stop.is_set():
        with contextlib.suppress(Exception):
            inspect_once()


t = threading.Thread(target=hammer, daemon=True)
t.start()
time.sleep(4)
loaded = fps_samples(25)
stop.set()
t.join(timeout=190)

health = get(EDGE + "/health")
buf = health["buffer"]

print()
print(f"detector fps, idle        : {stat(idle)}")
print(f"detector fps, under load  : {stat(loaded)}")
drop = 100 * (statistics.median(idle) - statistics.median(loaded)) / statistics.median(idle)
print(f"throughput change         : {-drop:+.1f}%")
print()
print(f"VLM calls per inspection  : {statistics.median(calls):.0f}")
print(f"frames sent per inspection: {statistics.median(frames_sent):.0f}")
print(f"frames in window          : {statistics.median(window_frames):.0f}")
print(f"VLM latency (edge-side)   : {stat(edge_ms, ' ms')}")
print(f"round trip from the Mac   : {stat(round_ms, ' ms')}")
print(f"frame selection cost      : {stat(sel_ms, ' ms')}")
print(f"host overhead (rtt - vlm) : "
      f"{statistics.median([r - e for r, e in zip(round_ms, edge_ms, strict=False)]):.1f} ms median")
print()
print(f"evidence buffer           : {buf['frames']} frames, {buf['image_frames']} images, "
      f"{buf['buffer_bytes'] / 1e6:.2f} MB")
