#!/usr/bin/env python3
"""Detector throughput idle vs under VLM load, measured from the Mac."""
import json
import statistics
import threading
import time
import urllib.request

STATE = "http://127.0.0.1:8800/api/state"
INSPECT = "http://192.168.2.2:8100/inspect"

def fps_samples(seconds, interval=1.0):
    out = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(STATE, timeout=5) as r:
                out.append(json.load(r)["fps"])
        except Exception:
            pass
        time.sleep(interval)
    return [f for f in out if f > 0]

def hammer(stop, latencies):
    body = json.dumps({"standard": "a person is visible in the image"}).encode()
    while not stop.is_set():
        try:
            req = urllib.request.Request(INSPECT, data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            latencies.append((time.monotonic() - t0) * 1000)
        except Exception:
            pass

print("warming up 8s...")
fps_samples(8)

print("measuring IDLE detector throughput for 30s...")
idle = fps_samples(30)

print("measuring detector throughput UNDER VLM LOAD for 40s...")
stop = threading.Event()
lat = []
t = threading.Thread(target=hammer, args=(stop, lat), daemon=True)
t.start()
time.sleep(5)                      # let the first judgement start
loaded = fps_samples(35)
stop.set()
t.join(timeout=125)

def stat(xs):
    xs = sorted(xs)
    return f"median {statistics.median(xs):.1f}  min {xs[0]:.1f}  max {xs[-1]:.1f}  n={len(xs)}"

print()
print(f"detector fps, idle       : {stat(idle)}")
print(f"detector fps, under load : {stat(loaded)}")
if lat:
    print(f"VLM judgements during load: n={len(lat)}  {stat(lat)} ms")
drop = 100 * (statistics.median(idle) - statistics.median(loaded)) / statistics.median(idle)
print(f"throughput drop under VLM load: {drop:.1f}%")
