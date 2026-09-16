#!/usr/bin/env python3
"""Measure verdict stability and the value of a self-consistency guard.

The model is asked for BOTH a boolean and a verdict string in one call. When the
two disagree the answer is untrustworthy and is reported as 'unclear' rather than
as a confident verdict.
"""
import base64
import json
import re
import sys
import time
import urllib.request

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FRAME = urllib.request.urlopen("http://127.0.0.1:8100/frame.jpg", timeout=15).read()
DATA_URL = "data:image/jpeg;base64," + base64.b64encode(FRAME).decode()

CASES = [
    ("the person is wearing glasses", "pass"),
    ("the person is wearing a lanyard", "pass"),
    ("the person is wearing a hard hat", "fail"),
    ("there is a dog in the picture", "fail"),
    ("the person is wearing a bright yellow safety vest", "fail"),
    ("a person is visible in the image", "pass"),
]

PROMPT = ("You are a strict visual inspection system.\n\n"
          "REQUIREMENT: {s}\n\n"
          "Decide whether the image visibly satisfies the requirement in full. "
          "If any part of it is missing, absent, or not visible, it is not satisfied.\n\n"
          "Reply with JSON only, no other text:\n"
          '{{"meets_requirement": true or false, '
          '"verdict": "pass" or "fail", '
          '"reason": "<one short sentence>"}}\n'
          'Set "verdict" to "pass" only when "meets_requirement" is true.')


def ask(standard):
    body = {"model": "vlm", "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(s=standard)},
        {"type": "image_url", "image_url": {"url": DATA_URL}}]}], "max_tokens": 160}
    req = urllib.request.Request("http://127.0.0.1:9998/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read().decode())
    return out["choices"][0]["message"]["content"], (time.monotonic() - t0) * 1000


def judge(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return "unclear", "no JSON", False
    try:
        o = json.loads(m.group(0))
    except ValueError:
        return "unclear", "bad JSON", False
    v = str(o.get("verdict", "")).lower()
    b = o.get("meets_requirement")
    reason = str(o.get("reason", ""))
    if v not in ("pass", "fail") or not isinstance(b, bool):
        return "unclear", reason, False
    agree = (b is True) == (v == "pass")
    return (v if agree else "unclear"), reason, agree


total = correct = disagreements = 0
lat = []
for rep in range(REPS):
    for standard, expected in CASES:
        text, ms = ask(standard)
        got, reason, agree = judge(text)
        lat.append(ms)
        total += 1
        correct += (got == expected)
        disagreements += (not agree)
        flag = "OK " if got == expected else ("UNC" if got == "unclear" else "BAD")
        if got != expected:
            print(f"  [{flag}] rep{rep+1} want={expected} got={got} :: {standard} :: {reason[:90]}")
lat.sort()
print(f"\ncases      : {len(CASES)} standards x {REPS} repetitions = {total} judgements")
print(f"correct    : {correct}/{total}  ({100*correct/total:.1f}%)")
print(f"self-inconsistent (guard fired): {disagreements}/{total}")
print(f"latency    : median {lat[len(lat)//2]:.0f} ms  min {lat[0]:.0f}  max {lat[-1]:.0f}")
