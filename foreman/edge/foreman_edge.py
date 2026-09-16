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
import re
import signal
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
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

@dataclass
class FrameRecord:
    """One pulled sample: detections always, pixels only for encoded frames."""
    frame_id: int
    ts: float
    detections: list  # list[dict]
    jpeg: bytes | None = None
    sharpness: float = 0.0


class EvidenceBuffer:
    """Rolling window of recent frames, shared with the HTTP threads.

    Detections are kept for every frame - they are tiny and they are the temporal
    grounding signal. JPEGs are kept only for every Nth frame, which at 15 fps and
    the default N=3 leaves about 15 candidate images across a 3 s window: enough
    to pick 3 well-separated representatives without paying to encode every frame.
    """

    def __init__(self, window_s: float = 3.0, max_frames: int = 300,
                 retain_s: float | None = None) -> None:
        self._lock = threading.Lock()
        self._frames: deque[FrameRecord] = deque(maxlen=max_frames)
        self.window_s = window_s
        #: How far back frames are kept, which is not the same as the default
        #: judging window. A manual capture asks for a longer span than the
        #: rolling window, and pruning to window_s would throw away the start of
        #: it before the capture finished.
        self.retain_s = max(window_s, retain_s or window_s)
        self.detector_ms: float = 0.0

    def add(self, record: FrameRecord, detector_ms: float) -> None:
        with self._lock:
            self._frames.append(record)
            self.detector_ms = detector_ms
            cutoff = record.ts - self.retain_s
            while self._frames and self._frames[0].ts < cutoff:
                self._frames.popleft()

    def window(self, seconds: float | None = None) -> list[FrameRecord]:
        """Every frame within the last `seconds`, oldest first."""
        with self._lock:
            if not self._frames:
                return []
            span = self.window_s if seconds is None else seconds
            cutoff = self._frames[-1].ts - span
            return [f for f in self._frames if f.ts >= cutoff]

    def between(self, start_ts: float, end_ts: float) -> list[FrameRecord]:
        """Frames captured within an explicit span, oldest first.

        Manual inspection judges what happened *after* the operator pressed the
        button, so it needs an absolute span rather than "the last N seconds".
        """
        with self._lock:
            return [f for f in self._frames if start_ts <= f.ts <= end_ts]

    def latest(self) -> FrameRecord | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def latest_jpeg(self) -> tuple[bytes | None, float]:
        """Most recent encoded frame and its timestamp."""
        with self._lock:
            for rec in reversed(self._frames):
                if rec.jpeg is not None:
                    return rec.jpeg, rec.ts
        return None, 0.0

    def stats(self) -> dict:
        with self._lock:
            n = len(self._frames)
            imgs = sum(1 for f in self._frames if f.jpeg is not None)
            bytes_held = sum(len(f.jpeg) for f in self._frames if f.jpeg)
            return {"frames": n, "image_frames": imgs, "buffer_bytes": bytes_held,
                    "detector_ms": round(self.detector_ms, 2)}


#: Retention is set from the CLI at startup (see main); the default here only
#: matters before then.
LATEST = EvidenceBuffer()
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


# ----------------------------------------------------- speech language control

#: Foreman accepts spoken standards in these two languages only.
#:
#: whisper-small is multilingual and its compiled `language_detect` stage picks
#: freely from ~99 languages. On real microphone audio that stage returned
#: Bulgarian for the English sentence "This person must be holding a phone",
#: yielding "Това пързина не ме ме ме обгърваме." Constraining the decode is the
#: only reliable fix: the detector head itself is what fails.
SPEECH_LANGUAGES = ("en", "ru")
SPEECH_MODES = ("en", "ru", "auto")

#: Measured on this board with whisper-small-a16w8 (see docs/BENCHMARKS.md).
#: Real speech scores no_speech_prob 0.002-0.005; silence, pink noise and speech
#: buried in noise all score 0.83-0.94. The gap is three orders of magnitude, so
#: this threshold is not finely tuned - anything in 0.3-0.8 behaves identically.
NO_SPEECH_MAX = 0.60
#: A forced decode in the wrong language degrades to about -1.2; a correct one
#: sits near -0.05. Rejects a decode that is grammatically incoherent.
LOGPROB_MIN = -1.00
#: Shorter than this there is nothing an inspection standard could be built from.
MIN_TRANSCRIPT_CHARS = 4

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


def transcript_script(text: str) -> str:
    """Which alphabet the transcript is actually written in."""
    cyr = len(_CYRILLIC.findall(text or ""))
    lat = len(_LATIN.findall(text or ""))
    if cyr == 0 and lat == 0:
        return "none"
    return "cyrillic" if cyr > lat else "latin"


#: The alphabet each forced decode is expected to produce.
_EXPECTED_SCRIPT = {"en": "latin", "ru": "cyrillic"}
_SCRIPT_LANGUAGE = {"latin": "en", "cyrillic": "ru"}


def language_from_script(text: str, fallback: str) -> str:
    """The language this transcript is actually written in.

    A forced decode is not always honoured, so the language Foreman *asked* for
    is not proof of what came back. Reporting the requested language regardless
    would send Latin text to the Russian lexicon in host/standard_parser.py,
    which matches nothing - and a standard that grounds nothing hands the verdict
    to the vision-language model alone, which is the failure this whole layer
    exists to prevent.
    """
    return _SCRIPT_LANGUAGE.get(transcript_script(text), fallback)


def script_agrees(language: str, text: str) -> bool:
    """True when a forced decode produced the alphabet that language is written in.

    Whisper does not always honour a forced language: asked for Russian on the
    English clip "There must be no phone in view" it returned that English
    sentence verbatim. Such a decode is evidence *against* the language it was
    forced into, so it must not be allowed to win on likelihood alone.
    """
    expected = _EXPECTED_SCRIPT.get(language)
    if expected is None:
        return True
    script = transcript_script(text)
    return script in (expected, "none")


def choose_transcript(candidates: list[dict]) -> dict | None:
    """Pick the best of several forced decodes of the same audio.

    Candidates whose alphabet contradicts the language they were forced into are
    discarded first; the most likely of what remains wins. If that rule discards
    everything, fall back to plain likelihood so the caller always gets an answer.
    """
    usable = [c for c in candidates if c.get("text", "").strip()]
    if not usable:
        return candidates[0] if candidates else None
    agreeing = [c for c in usable if script_agrees(c.get("language", ""), c.get("text", ""))]
    pool = agreeing or usable
    return max(pool, key=lambda c: float(c.get("avg_logprob", -99.0)))


def assess_transcript(text: str, no_speech_prob: float, avg_logprob: float) -> tuple[bool, str]:
    """Decide whether a transcript is good enough to become an inspection standard.

    Returns (accepted, reason). A rejected transcript must never replace the
    standard in force: silently adopting a garbled rule is the failure mode that
    makes an inspection system untrustworthy.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "no speech was detected"
    if no_speech_prob >= NO_SPEECH_MAX:
        # Whisper hallucinates confidently on silence - it produced
        # "СПОКОЙНАЯ МУЗЫКА" for pink noise at a healthy -0.53 logprob - so
        # likelihood alone cannot catch this and no_speech_prob must.
        return False, "the recording does not appear to contain speech"
    if avg_logprob <= LOGPROB_MIN:
        return False, "the speech could not be recognised clearly"
    if len(cleaned) < MIN_TRANSCRIPT_CHARS or not re.search(r"[^\W\d_]", cleaned, re.UNICODE):
        return False, "the transcript was too short to be a standard"
    return True, ""


# ------------------------------------------------------- GenAI over HTTP

def capture_window(buf, capture_s: float, timeout_pad: float = 3.0):
    """Collect a fresh evidence window starting now, for manual inspection.

    Blocks until `capture_s` of NEW frames have arrived, then returns exactly
    those frames. Deliberately not `buf.window(capture_s)`: that would return the
    seconds *before* the call, and a manual inspection is a statement about what
    happens after the operator presses the button.

    Returns (frames, start_ts, end_ts). Gives up after capture_s + timeout_pad so
    a stalled camera cannot hang the request forever.
    """
    # The press itself is the boundary, not the last frame already buffered:
    # using that frame's timestamp would let one pre-press frame into the window,
    # and a manual inspection is a claim about what happened afterwards.
    start_ts = time.time()
    deadline = time.monotonic() + capture_s + timeout_pad
    end_ts = start_ts + capture_s
    while time.monotonic() < deadline:
        newest = buf.latest()
        if newest is not None and newest.ts >= end_ts:
            break
        time.sleep(0.05)
    frames = buf.between(start_ts, end_ts)
    return frames, start_ts, (frames[-1].ts if frames else end_ts)


#: The prose values in the prompt's JSON example, in the order they appear:
#: evidence, missing_evidence, reason. A small model shown six images returns the
#: right structured answer but sometimes reproduces these word for word, which
#: put "No phone is visible in any frame." on a correct PASS and then tripped the
#: detector-contradiction guard. They live here so build_window_prompt() writes
#: them and parse_window_judgement() can recognise them; the two cannot drift.
_EXAMPLE_PROSE = (
    "hands are visible and empty",
    "any phone in the hand",
    "No phone is visible in any frame.",
)


def build_window_prompt(n: int, standard: str, detector_summary: str,
                        roi_label: str | None = None, roi_count: int = 0,
                        confirmed: list[str] | None = None,
                        relation: str | None = None,
                        relation_subject: str | None = None,
                        relation_object: str | None = None,
                        relation_expected: bool | None = None) -> str:
    """The judging prompt for one evidence window.

    Shared by both backends so the instructions cannot drift apart;
    tests/test_edge_pure.py pins the exact text.

    `confirmed` names objects the detector measured as reliably present. Those
    are stated as settled fact and the model is told its job is the relationship,
    not presence. Without it the model answered a question it had not been asked:
    with a phone detected in 42 of 45 frames at 85% confidence it still replied
    "No phone is visible in any frame."
    """
    if roi_label and roi_count:
        wide = n - roi_count
        layout = (
            f"You are shown {n} images from about three seconds of video. "
            f"The first {wide} show the whole scene, in time order. "
            f"The remaining {roi_count} are close-up crops of the {roi_label} "
            f"from the same window, enlarged so you can see detail. "
            f"Use the close-ups to judge small features, and the wide shots for "
            f"context and relationships.\n\n")
    else:
        layout = (f"You are shown {n} still frames captured over about three "
                  "seconds, in time order.\n\n")
    if confirmed:
        listed = "".join(f"- {c}\n" for c in confirmed)
        grounded = (
            "DETECTOR-CONFIRMED FACTS. These were measured across the window and "
            "are authoritative for whether an object is present:\n"
            f"{listed}"
            "Treat the presence of those objects as settled. Do NOT say any of "
            "them is absent, missing or not visible - that question is already "
            "answered. Judge only the remaining part of the requirement: the "
            "relationship, position or attribute.\n\n")
    else:
        grounded = ""

    # A relationship rule turns on a value, not on presence: "must be holding"
    # and "must NOT be holding" name the same relationship and disagree only
    # about what it should be. The prompt therefore states the expected value and
    # asks for the OBSERVED one. Deciding whether the observation passes is
    # policy, and policy is applied deterministically in host/policy.py rather
    # than left to the model.
    if relation and relation_subject and relation_object:
        want = "true" if relation_expected else "false"
        rel_block = (
            "RELATIONSHIP TO JUDGE:\n"
            f"- relationship: {relation}({relation_subject}, {relation_object})\n"
            f"- the standard is met when {relation} = {want}\n"
            f"Decide only one thing: in these frames, is the {relation_subject} "
            f"actually {relation} the {relation_object}?\n"
            "Report what you OBSERVE. Do not report whether the rule passes - "
            "that is decided elsewhere from your observation.\n"
            f'Put it in "observed_relationship" as exactly one of: '
            f'"{relation}", "not_{relation}", "unclear".\n'
            f"The {relation_object} being somewhere in the frame is NOT enough: "
            f'answer "{relation}" only if the {relation_subject} is holding it.\n\n')
        rel_field = f'"observed_relationship":"{relation}|not_{relation}|unclear",'
    else:
        rel_block = ""
        rel_field = ""

    return (
        "You are a careful visual inspection system. " + layout +
        f"REQUIREMENT: {standard}\n\n"
        f"DETECTOR EVIDENCE (measured, not your opinion):\n{detector_summary}\n\n"
        + grounded + rel_block +
        "Rules you must follow:\n"
        "- Judge ONLY what is directly visible in these frames.\n"
        "- Never state that an object is present unless you can actually see it.\n"
        "- Never infer an object from context, from the scene type, or from what "
        "would usually be there.\n"
        "- Keep what you SEE separate from what you ASSUME. Report only what you see.\n"
        "- Use all the frames together: an object briefly hidden in one frame but "
        "clearly visible in others is still present.\n"
        "- If the frames disagree with each other, answer UNCLEAR.\n"
        "- If you cannot see enough to decide, answer UNCLEAR.\n"
        "- If the detector evidence says a required object was never found, you "
        "may not claim it is present.\n"
        "- Equally, if an object is listed as detector-confirmed above, you may "
        "not claim it is absent. Judge the relationship instead.\n"
        "- Write about the objects this requirement names. Do not describe "
        "unrelated things in the scene.\n\n"
        "Answer with JSON only, no other text, and keep it SHORT - every "
        "string under 15 words, at most two items per list:\n"
        "Follow this shape exactly, but write your own values - never copy the "
        "example text:\n"
        # These three strings are the example prose, and they are also listed in
        # _EXAMPLE_PROSE so the parser can recognise them if they come back
        # verbatim. Keeping a realistic example measurably produces better
        # wording than an angle-bracket placeholder, which the small model simply
        # copies instead, leaving no usable reason at all. The echo is therefore
        # detected rather than prevented.
        '{"meets_requirement":false,"verdict":"FAIL",'
        f'"per_frame":[{",".join(["false"] * n)}],'
        + rel_field +
        f'"evidence":["{_EXAMPLE_PROSE[0]}"],'
        f'"missing_evidence":["{_EXAMPLE_PROSE[1]}"],'
        f'"reason":"{_EXAMPLE_PROSE[2]}"}}\n'
        f'"per_frame" must have exactly {n} entries, one per frame in order: '
        "true if that frame supports the requirement, false if it contradicts "
        "it, null if you cannot tell from that frame. Do not describe the "
        "frames individually.\n"
        'Set "verdict" to "PASS" only when "meets_requirement" is true, and to '
        '"FAIL" when it is false. The two must agree.'
    )


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

    def judge_window(self, jpegs: list[bytes], standard: str,
                     detector_summary: str, roi_label: str | None = None,
                     roi_count: int = 0,
                     confirmed: list[str] | None = None,
                     relation: tuple[str, str, str, bool] | None = None) -> tuple[dict, dict]:
        """Judge a short evidence window from several frames in ONE request.

        Multi-image support was verified on this hardware: three images in one
        request are received and described individually (see docs/BENCHMARKS.md,
        "Multi-image support"). One call is preferred over N calls because it
        lets the model reason across time and costs ~3.1 s instead of ~4 s.

        The detector's own findings are put in the prompt so the model is told
        what was actually measured rather than left to guess - but the binding
        constraint is enforced on the Mac, in host/policy.py, not here.
        """
        n = len(jpegs)
        # (name, subject, object, expected) or None for a presence-only rule.
        rel_name, rel_subj, rel_obj, rel_want = relation or (None, None, None, None)
        prompt = build_window_prompt(
            n, standard, detector_summary, roi_label, roi_count, confirmed,
            relation=rel_name, relation_subject=rel_subj,
            relation_object=rel_obj, relation_expected=rel_want)
        content = [{"type": "text", "text": prompt}]
        for jpeg in jpegs:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})

        started = time.monotonic()
        result = self._post_json("/v1/chat/completions", {
            "model": self.vlm_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 260,
        })
        elapsed_ms = (time.monotonic() - started) * 1000.0

        text = ""
        with contextlib.suppress(KeyError, IndexError, TypeError):
            text = result["choices"][0]["message"]["content"]

        usage = result.get("usage") or {}
        metrics = {"inference_ms": round(elapsed_ms, 1), "vlm_calls": 1, "frames_sent": n}
        for src, dst in (("ttft_ms", "ttft_ms"), ("tokens_per_second", "tokens_per_s"),
                         ("completion_tokens", "completion_tokens")):
            if src in usage:
                metrics[dst] = float(usage[src])
        return parse_window_judgement(text, n), metrics

    def transcribe_forced(self, audio: bytes, filename: str, language: str) -> dict:
        """One Whisper decode on the MLA with the language pinned.

        The server honours an explicit ISO code and skips its own language
        detection entirely, which is the whole point: the detection stage is what
        chose Bulgarian for English speech.
        """
        body, content_type = encode_multipart({
            "model": self.asr_model,
            "language": language,
        }, "file", filename, audio)
        req = urllib.request.Request(
            self.base_url + "/v1/audio/transcriptions",
            data=body, headers={"Content-Type": content_type},
        )
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode())
        return {
            "text": (result.get("text") or "").strip(),
            # Trust what we asked for, not what the server echoes back.
            "language": language,
            "avg_logprob": float(result.get("avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(result.get("no_speech_prob", 0.0) or 0.0),
            "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
        }

    def transcribe(self, audio: bytes, filename: str, mode: str = "auto") -> dict:
        """Transcribe speech, constrained to English or Russian.

        `mode` is "en" or "ru" to pin the language, or "auto" to let Foreman
        choose between exactly those two. Auto never asks the server to detect a
        language: it decodes the clip both ways and keeps the better reading, so
        a third language cannot be returned however the audio sounds.
        """
        mode = mode if mode in SPEECH_MODES else "auto"
        if mode in SPEECH_LANGUAGES:
            best = self.transcribe_forced(audio, filename, mode)
            # Pinning Russian and then speaking English returns the English
            # sentence verbatim (measured - see docs/BENCHMARKS.md 4.1). Label it
            # for what it is rather than for what was asked.
            best = dict(best, language=language_from_script(best["text"], mode))
            candidates = [best]
        else:
            candidates = []
            for lang in SPEECH_LANGUAGES:
                cand = self.transcribe_forced(audio, filename, lang)
                candidates.append(cand)
                # Measured on this server, no_speech_prob is byte-identical
                # across both forced languages for the same clip - it is derived
                # from the audio, not from the decode. If the clip has no speech
                # in it a second pass cannot change that, so stop and save the
                # latency.
                if cand["no_speech_prob"] >= NO_SPEECH_MAX:
                    break
            best = choose_transcript(candidates) or candidates[0]
            # If both decodes contradicted their forced language, the winner
            # still carries a label its own alphabet disagrees with. Correct it.
            best = dict(best, language=language_from_script(best["text"],
                                                            best["language"]))

        accepted, why = assess_transcript(
            best["text"], best["no_speech_prob"], best["avg_logprob"])
        return {
            "text": best["text"],
            "language": best["language"],
            "mode": mode,
            "accepted": accepted,
            "reject_reason": why,
            "avg_logprob": best["avg_logprob"],
            "no_speech_prob": best["no_speech_prob"],
            "candidates": [
                {"language": c["language"], "text": c["text"],
                 "avg_logprob": c["avg_logprob"]} for c in candidates
            ],
            "metrics": {
                "inference_ms": round(sum(c["inference_ms"] for c in candidates), 1),
                "asr_calls": len(candidates),
            },
        }

def _strip_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw[3:]
        raw = raw.removeprefix("json").strip()
    return raw


def _clean_text(value, fallback: str) -> str:
    """Trim a model string, rejecting schema placeholders it copied verbatim.

    Observed on hardware: the model returned the literal "<one short sentence>"
    from the prompt's own schema. A placeholder is worse than nothing, because it
    looks like a real explanation in the UI.
    """
    text = str(value or "").strip()[:400]
    if not text:
        return fallback
    if text.startswith("<") and text.endswith(">"):
        return fallback
    if text.lower() in {"<one short sentence>", "string", "reason", "n/a"}:
        return fallback
    # A verbatim copy of the prompt's own example is not an observation. It reads
    # exactly like one, which is what makes it dangerous: it reached the operator
    # as "No phone is visible in any frame." on an inspection whose detector had
    # the phone in 45 of 45 frames, and the grounding guard then overturned a
    # correct PASS on the strength of a sentence the model had merely copied.
    if text.lower() in {s.lower() for s in _EXAMPLE_PROSE}:
        return fallback
    return text


def parse_window_judgement(text: str, frames_sent: int) -> dict:
    """Parse the multi-frame reply into the shape host/policy.py expects.

    Anything unparseable becomes UNCLEAR with an explanation rather than a
    guessed verdict - an inspection result must never be invented here.
    """
    raw = _strip_fence(text)
    start, end = raw.find("{"), raw.rfind("}")
    obj = None
    if start != -1 and end > start:
        with contextlib.suppress(ValueError):
            obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        return {"verdict": "unclear", "reason": "The model did not return usable JSON.",
                "evidence": [], "missing_evidence": [], "per_frame": [None] * frames_sent}

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "fail", "unclear"}:
        verdict = "unclear"

    # The model is asked for a boolean and a verdict string. On this model they
    # sometimes disagree - observed: verdict "pass" with the reason "the person is
    # not holding a smartphone". A disagreement means the answer is not
    # trustworthy, so it becomes UNCLEAR rather than a confident verdict.
    meets = obj.get("meets_requirement")
    contradiction = False
    if isinstance(meets, bool) and verdict in {"pass", "fail"}:
        if (meets is True) != (verdict == "pass"):
            contradiction = True
            verdict = "unclear"
    elif isinstance(meets, bool) and verdict == "unclear":
        pass

    per_frame: list[bool | None] = []
    for entry in (obj.get("per_frame") or [])[:frames_sent]:
        supports = entry.get("supports") if isinstance(entry, dict) else entry
        per_frame.append(supports if isinstance(supports, bool) else None)
    while len(per_frame) < frames_sent:
        per_frame.append(None)

    def strlist(key):
        value = obj.get(key) or []
        if isinstance(value, str):
            value = [value]
        out, seen = [], set()
        for item in value:
            text = _clean_text(item, "")
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out[:4]

    reason = _clean_text(obj.get("reason"), "")
    if not reason:
        # The model sometimes omits the sentence but still lists what it saw.
        # Reuse that rather than showing an empty explanation next to a verdict.
        ev = strlist("evidence")
        reason = (ev[0][0].upper() + ev[0][1:]) if ev else "No reason given."
    if contradiction:
        reason = f"The model contradicted itself, so this needs a human. {reason}"

    # The observed relationship is an observation, not a verdict: the policy
    # compares it against the polarity the standard asks for. Normalised here so
    # "NOT_HOLDING", "not holding" and "not_holding" all mean the same thing.
    observed = obj.get("observed_relationship")
    if isinstance(observed, str) and observed.strip():
        observed = observed.strip().lower().replace(" ", "_").replace("-", "_")
    else:
        observed = None

    return {
        "verdict": verdict,
        "reason": reason,
        "evidence": strlist("evidence"),
        "missing_evidence": strlist("missing_evidence"),
        "per_frame": per_frame,
        "observed_relationship": observed,
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

def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def box_containment(a, b) -> float:
    """Fraction of the SMALLER box that lies inside the larger one."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(aa, ab)
    return (iw * ih) / smaller if smaller > 0 else 0.0


def suppress_nested_duplicates(detections, containment_min=0.92, area_ratio_max=0.12):
    """Drop a small same-class box almost entirely swallowed by a much larger one.

    On-device NMS uses IoU, which is structurally blind to nesting: a box that is
    95% inside a box ten times its size still scores a low IoU, because the union
    is dominated by the large box. Measured here: same-class nested pairs appeared
    in ~2.7% of frames (10 pairs over 375 frames) with IoU 0.05-0.10 but
    containment 0.73-0.97, so the 0.45 IoU threshold never touched them.

    The thresholds are deliberately strict. A second person standing behind the
    main subject was verified by eye in this scene at containment 0.65 with an
    area ratio of 0.013 - a real person, not a duplicate - so containment alone
    is NOT sufficient evidence. Requiring containment >= 0.92 keeps that case,
    and every other genuinely separate person observed, intact.

    Only the lower-confidence box of a suppressed pair is dropped, and only
    within the same class: two different classes may legitimately nest
    (a cell phone inside a person).
    """
    keep = []
    for det in sorted(detections, key=lambda d: -float(d.get("confidence", 0.0))):
        box = det["bbox"]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        swallowed = False
        for kept in keep:
            if kept["label"] != det["label"]:
                continue
            kbox = kept["bbox"]
            karea = max(0.0, kbox[2] - kbox[0]) * max(0.0, kbox[3] - kbox[1])
            if karea <= 0 or area > karea:
                continue
            if (box_containment(box, kbox) >= containment_min
                    and (area / karea) <= area_ratio_max):
                swallowed = True
                break
        if not swallowed:
            keep.append(det)
    return keep


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


def luma_sharpness(payload, width: int, height: int) -> float:
    """Variance of the Laplacian over a downscaled luma plane.

    NV12 stores luma first, so this needs no colour conversion. Downscaling to
    320 px wide keeps the cost negligible at 15 fps while still separating a
    motion-blurred frame from a sharp one. Higher is sharper; the absolute value
    is scene-dependent and is only ever compared within one evidence window.
    """
    try:
        arr = np.frombuffer(payload, dtype=np.uint8)
        aligned_h = height if arr.size == width * height * 3 // 2 else int(arr.size // (width * 1.5))
        y = arr[:aligned_h * width].reshape(aligned_h, width)[:height, :]
        small = cv2.resize(y, (320, max(1, int(320 * height / width))),
                           interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(small, cv2.CV_64F).var())
    except Exception:
        return 0.0


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


def pick_roi_class(frames, required):
    """Choose which required object is worth zooming into.

    The smallest one, by median area across the window. The detector grounds the
    parent object; the details that decide a verdict - a cap, a label, glasses, a
    hand gripping a phone - are usually a small part of it, and are the first
    thing lost when a 1280x720 frame is resized for the vision encoder. Picking
    the smallest required object is generic: it needs no per-object rules and no
    list of attributes.
    """
    best, best_area = None, None
    for label in required:
        areas = []
        for f in frames:
            for d in f.detections:
                if d.get("label") == label:
                    b = d["bbox"]
                    areas.append(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
        if not areas:
            continue
        areas.sort()
        median = areas[len(areas) // 2]
        if best_area is None or median < best_area:
            best, best_area = label, median
    return best, (best_area or 0.0)


def stable_bbox(frames, label):
    """Median bbox for a class across the window, resistant to a single bad frame."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for f in frames:
        best, bbox = 0.0, None
        for d in f.detections:
            if d.get("label") == label and float(d.get("confidence", 0)) > best:
                best, bbox = float(d["confidence"]), d["bbox"]
        if bbox:
            xs1.append(bbox[0])
            ys1.append(bbox[1])
            xs2.append(bbox[2])
            ys2.append(bbox[3])
    if not xs1:
        return None
    def med(values):
        return sorted(values)[len(values) // 2]

    return (med(xs1), med(ys1), med(xs2), med(ys2))


def crop_roi(jpeg: bytes, bbox, pad: float = 0.25, min_side: int = 224) -> bytes | None:
    """Crop a padded region of interest and return it as JPEG.

    Padding keeps context around the object - a cap is only meaningful relative to
    the bottle neck it sits on. Very small crops are upscaled to `min_side` so the
    vision encoder receives usable detail instead of a handful of pixels.
    """
    try:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = (x2 - x1) * w, (y2 - y1) * h
        px, py = bw * pad, bh * pad
        cx1 = max(0, int(x1 * w - px))
        cy1 = max(0, int(y1 * h - py))
        cx2 = min(w, int(x2 * w + px))
        cy2 = min(h, int(y2 * h + py))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        roi = img[cy1:cy2, cx1:cx2]
        side = min(roi.shape[0], roi.shape[1])
        if side < min_side:
            scale = min_side / side
            roi = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                             interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def detector_summary(frames, required, prohibited) -> str:
    """Plain-language summary of what the detector measured across the window.

    Goes into the VLM prompt so the model is told the measurement instead of
    guessing. The binding constraint is still applied on the Mac.
    """
    total = len(frames)
    lines = []
    for label in [*required, *prohibited]:
        present = sum(1 for f in frames
                      if any(d.get("label") == label for d in f.detections))
        pct = round(100 * present / total) if total else 0
        if present == 0:
            lines.append(f"- {label}: NEVER detected in any of {total} frames.")
        else:
            lines.append(f"- {label}: detected in {present}/{total} frames ({pct}%).")
    if not lines:
        lines.append("- no detector-supported objects were named in this requirement.")
    return "\n".join(lines)


def confirmed_present(frames, required, present_ratio_min: float = 0.60) -> list[str]:
    """Required objects the detector saw often enough to call settled.

    The same threshold host/policy.py uses to treat an object as reliably
    present, so the prompt and the policy agree about what counts as confirmed.
    These are handed to the model as fact, which is what stops it answering a
    presence question it was not asked.
    """
    total = len(frames)
    if not total:
        return []
    out = []
    for label in required:
        present = sum(1 for f in frames
                      if any(d.get("label") == label for d in f.detections))
        if present / total >= present_ratio_min:
            pct = round(100 * present / total)
            out.append(f"{label} is present in {present}/{total} frames ({pct}%)")
    return out



def rule_inputs(body: dict, frames, default_standard: str,
                default_required: list, default_prohibited: list,
                default_relation) -> list[dict]:
    """One judging spec per rule, all against the SAME captured frames.

    A request without "rules" yields exactly one spec built from the top-level
    fields, so a single-rule inspection is byte-identical to what it was before
    multi-rule existed.

    Each spec carries only the objects ITS rule names. That scoping is what keeps
    rule 1's answer from discussing rule 2's objects: the model is never told
    about a class the rule it is judging does not mention.
    """
    raw_rules = body.get("rules") or []
    if not raw_rules:
        raw_rules = [{"standard": default_standard,
                      "required_objects": default_required,
                      "prohibited_objects": default_prohibited,
                      "relation": default_relation}]
    specs = []
    for r in raw_rules:
        req = [str(x) for x in (r.get("required_objects") or [])]
        proh = [str(x) for x in (r.get("prohibited_objects") or [])]
        rel = r.get("relation") or None
        name = str(rel.get("name") or "") if rel else ""
        subj = str(rel.get("subject") or "") if rel else ""
        obj = str(rel.get("object") or "") if rel else ""
        want = bool(rel.get("expected")) if rel and name else None
        scoped = list(req)
        if obj and obj not in scoped:
            scoped.append(obj)
        specs.append({
            "standard": str(r.get("standard") or default_standard),
            "summary": detector_summary(frames, scoped, proh),
            "confirmed": confirmed_present(frames, scoped),
            "required": req,
            "relation": (name, subj, obj, want) if name and obj else None,
        })
    return specs

def select_representative(frames, required, count: int, min_gap_s: float | None = None):
    """Pick `count` sharp, well-separated frames that show the required objects.

    Greedy by score, but a candidate is rejected if it sits within `min_gap_s` of
    an already-chosen frame. Bucketing alone was not enough: with three buckets
    over three seconds, frames at +0.9 s and +1.0 s land in different buckets yet
    are effectively the same instant, which defeats the point of temporal
    evidence. The gap defaults to 60% of an even split.

    If the gap cannot be satisfied (short or sparse window) it is relaxed rather
    than returning fewer frames, because three near frames still beat one.

    Score combines: how many required objects the frame shows, the detector
    confidence for them, and sharpness normalised against this window - the
    absolute Laplacian variance is scene-dependent and only comparable within it.
    """
    candidates = [f for f in frames if f.jpeg is not None]
    if not candidates or count <= 0:
        return []

    span = max(candidates[-1].ts - candidates[0].ts, 1e-6)
    if min_gap_s is None:
        min_gap_s = (span / count) * 0.6
    sharp_max = max((f.sharpness for f in candidates), default=0.0) or 1.0

    def score(rec) -> float:
        hits, conf = 0, 0.0
        for label in required:
            best = max((float(d.get("confidence", 0.0)) for d in rec.detections
                        if d.get("label") == label), default=0.0)
            if best > 0:
                hits += 1
                conf += best
        object_score = (hits / len(required)) if required else 0.0
        conf_score = (conf / len(required)) if required else 0.0
        return 2.0 * object_score + 1.0 * conf_score + 1.0 * (rec.sharpness / sharp_max)

    ranked = sorted(candidates, key=score, reverse=True)
    chosen: list = []

    # Pass 1: one frame per equal slice of the window, best-scoring within each.
    # Greedy-by-score alone clusters wherever the sharpest frames happen to fall -
    # measured on a real 3 s capture it put all six frames between +0.88 s and
    # +2.87 s and left the first second unrepresented, which is not temporal
    # evidence so much as a burst. Slicing guarantees the window is covered; the
    # gap check below still rejects two picks that land either side of a boundary
    # and are effectively the same instant.
    t0 = candidates[0].ts
    slice_s = span / count
    for i in range(count):
        lo, hi = t0 + i * slice_s, t0 + (i + 1) * slice_s
        in_slice = [f for f in ranked
                    if lo <= f.ts < hi or (i == count - 1 and f.ts >= lo)]
        for rec in in_slice:
            if all(abs(rec.ts - r.ts) >= min_gap_s for r in chosen):
                chosen.append(rec)
                break

    # Pass 2: top up from anywhere if slices were empty or the gap blocked them,
    # relaxing the gap rather than returning fewer frames - six near frames still
    # beat three.
    gap = min_gap_s
    while len(chosen) < count and gap >= 0:
        for rec in ranked:
            if len(chosen) >= count:
                break
            if any(r is rec for r in chosen):
                continue
            if all(abs(rec.ts - r.ts) >= gap for r in chosen):
                chosen.append(rec)
        gap = gap / 2 if gap > 0.02 else -1

    return sorted(chosen, key=lambda r: r.ts)


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
                if self.args.suppress_nested:
                    detections = suppress_nested_duplicates(
                        detections,
                        containment_min=self.args.nested_containment,
                        area_ratio_max=self.args.nested_area_ratio)
            except Exception as exc:
                print(f"[foreman-edge] box decode failed: {exc}", file=sys.stderr, flush=True)

            jpeg, sharpness = None, 0.0
            if n % every == 0:
                frame_field = find_field(sample, "frame")
                tensor = first_tensor(frame_field if frame_field is not None else sample)
                if tensor is not None:
                    jpeg = frame_to_jpeg(tensor, self.args.jpeg_quality)
                    if jpeg is not None:
                        sharpness = luma_sharpness(tensor.copy_payload_bytes(),
                                                   tensor_dim(tensor, "width"),
                                                   tensor_dim(tensor, "height"))

            detector_ms = (time.monotonic() - started) * 1000.0
            frame_id = int(getattr(sample, "frame_id", n) or n)
            LATEST.add(FrameRecord(frame_id=frame_id, ts=time.time(),
                                   detections=detections, jpeg=jpeg,
                                   sharpness=sharpness), detector_ms)

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
                rec = LATEST.latest()
                stats = LATEST.stats()
                return self._json(200, {
                    "ok": True,
                    # Which inference backend answered. The console and the audit
                    # trail carry this through so a local development result can
                    # never be mistaken for one produced on the Modalix MLA.
                    "backend": "modalix",
                    "models": {"detector": args.model.rsplit("/", 1)[-1],
                               "vlm": args.vlm_model, "asr": args.asr_model},
                    "frame_id": rec.frame_id if rec else 0,
                    "last_frame_age_s": round(time.time() - rec.ts, 2) if rec else None,
                    "detector_ms": stats["detector_ms"],
                    "window_s": args.window_s,
                    "evidence_frames": args.evidence_frames,
                    "buffer": stats,
                })
            if self.path == "/events":
                return self._events()
            if self.path == "/frame.jpg":
                jpeg, _ = LATEST.latest_jpeg()
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
            route, _, query = self.path.partition("?")
            if route == "/inspect":
                return self._inspect()
            if route == "/inspect_single":
                return self._inspect_single()
            if route == "/transcribe":
                return self._transcribe(query)
            return self._json(404, {"error": "not found"})

        def _inspect(self):
            """Judge a rolling evidence window, not a single frame.

            Returns raw evidence - the window bounds, the selected frames and the
            model's structured reply. It deliberately does NOT return a final
            verdict: the grounding policy that can override the model lives on
            the Mac, in host/policy.py, where it is unit-testable without hardware.
            """
            try:
                body = json.loads(self._read_body() or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            standard = str(body.get("standard", "")).strip()
            if not standard:
                return self._json(400, {"error": "standard is required"})

            required = [str(x) for x in (body.get("required_objects") or [])]
            prohibited = [str(x) for x in (body.get("prohibited_objects") or [])]
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
                # Manual mode: judge what happens from now on, not what is
                # already in the rolling buffer.
                frames, cap_start, cap_end = capture_window(LATEST, capture_s)
                capture = {"mode": "manual", "requested_s": capture_s,
                           "start_ts": cap_start, "end_ts": cap_end,
                           "duration_s": round(cap_end - cap_start, 3)}
            else:
                frames = LATEST.window(window_s)
                capture = {"mode": "auto", "requested_s": window_s}
            if not frames:
                return self._json(503, {"error": "no frames buffered yet"})
            if time.time() - frames[-1].ts > 5.0:
                return self._json(503, {"error": "camera frames are stale; check the RTSP source"})

            selected = select_representative(frames, required, want)
            if not selected:
                return self._json(503, {"error": "no encoded frames in the evidence window"})
            selection_ms = (time.monotonic() - t_sel) * 1000.0

            summary = detector_summary(frames, summary_objects, prohibited)

            # Generic attribute inspection: the detector grounds the parent
            # object, and a padded close-up of it gives the model the detail it
            # needs for sub-parts it has no class for (a cap, a label, glasses).
            roi_label, roi_images, roi_meta = None, [], []
            if required and args.roi_enabled:
                roi_label, _ = pick_roi_class(frames, required)
            if roi_label:
                fallback = stable_bbox(frames, roi_label)
                for f in selected:
                    bbox = None
                    best = 0.0
                    for det in f.detections:
                        if det.get("label") == roi_label and float(det.get("confidence", 0)) > best:
                            best, bbox = float(det["confidence"]), det["bbox"]
                    bbox = bbox or fallback
                    if not bbox:
                        continue
                    crop = crop_roi(f.jpeg, bbox, pad=args.roi_pad)
                    if crop:
                        roi_images.append(crop)
                        roi_meta.append({"frame_id": f.frame_id,
                                         "rel_ts": round(f.ts - frames[0].ts, 3),
                                         "label": roi_label,
                                         "bbox": [round(v, 4) for v in bbox],
                                         "bytes": len(crop)})
                roi_images = roi_images[:args.roi_frames]
                roi_meta = roi_meta[:args.roi_frames]

            wide = [f.jpeg for f in selected]
            if roi_images:
                # Keep total images bounded: fewer wide shots when ROI crops are added.
                wide = wide[:max(1, args.evidence_frames - len(roi_images) + 1)]
            # One judgement per rule, over the SAME `wide` frames. The camera is
            # never re-opened and the detector never re-runs: this loop is the
            # only thing that repeats, which is the trade the design accepts to
            # keep the rules from contaminating each other in one merged prompt.
            specs = rule_inputs(body, frames, standard, required, prohibited,
                                    body.get("relation"))
            vlms, total_ms, calls = [], 0.0, 0
            try:
                for spec in specs:
                    j, m = genai.judge_window(
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
            except (urllib.error.URLError, OSError, ValueError) as exc:
                return self._json(502, {"error": f"vision-language model unavailable: {exc}"})

            metrics["selection_ms"] = round(selection_ms, 1)
            t0 = frames[0].ts
            return self._json(200, {
                "window": {
                    "start_ts": t0,
                    "end_ts": frames[-1].ts,
                    "duration_s": round(frames[-1].ts - t0, 3),
                    "total_frames": len(frames),
                    "image_frames": sum(1 for f in frames if f.jpeg is not None),
                },
                "detector_summary": summary,
                "capture": capture | {"detector_frames": len(frames)},
                "roi": {"label": roi_label, "frames": roi_meta} if roi_images else None,
                "selected": [{
                    "frame_id": f.frame_id,
                    "ts": f.ts,
                    "rel_ts": round(f.ts - t0, 3),
                    "sharpness": round(f.sharpness, 1),
                    "detections": [{k: v for k, v in d.items() if k != "_px"}
                                   for d in f.detections],
                    "jpeg_b64": base64.b64encode(f.jpeg).decode(),
                } for f in selected],
                "vlm": judgement,
                "vlms": vlms,
                "metrics": metrics,
            })

        def _inspect_single(self):
            """The pre-temporal single-frame path, kept as a fallback.

            Judges the most recent encoded frame with no temporal evidence and no
            detector grounding - i.e. the behaviour that allowed a hallucinated
            object to PASS. Reachable only when the Mac is started with
            FOREMAN_TEMPORAL=0, and retained so the previously demonstrated
            pipeline stays runnable while the temporal one is being proven.
            """
            try:
                body = json.loads(self._read_body() or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON"})
            standard = str(body.get("standard", "")).strip()
            if not standard:
                return self._json(400, {"error": "standard is required"})

            jpeg, ts = LATEST.latest_jpeg()
            if jpeg is None:
                return self._json(503, {"error": "no frame available from the camera yet"})
            if time.time() - ts > 5.0:
                return self._json(503, {"error": "camera frames are stale; check the RTSP source"})
            try:
                verdict, reason, metrics = genai.judge(jpeg, standard)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                return self._json(502, {"error": f"vision-language model unavailable: {exc}"})
            return self._json(200, {
                "verdict": verdict, "reason": reason,
                "evidence_jpeg_b64": base64.b64encode(jpeg).decode(),
                "metrics": metrics,
            })

        def _transcribe(self, query: str = ""):
            body = self._read_body()
            if not body:
                return self._json(400, {"error": "empty upload"})
            audio, filename = extract_uploaded_file(body, self.headers.get("Content-Type", ""))
            if not audio:
                return self._json(400, {"error": "no file part in upload"})
            mode = (urllib.parse.parse_qs(query).get("language") or ["auto"])[0]
            try:
                return self._json(200, genai.transcribe(audio, filename, mode))
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
    p.add_argument("--no-suppress-nested", dest="suppress_nested", action="store_false",
                   help="disable nested duplicate-box suppression")
    p.add_argument("--nested-containment", type=float, default=0.92,
                   help="min containment for a nested box to be suppressed")
    p.add_argument("--nested-area-ratio", type=float, default=0.12,
                   help="max small/large area ratio for nested suppression")
    p.add_argument("--no-roi", dest="roi_enabled", action="store_false",
                   help="disable region-of-interest attribute inspection")
    p.add_argument("--roi-frames", type=int, default=int(env("FOREMAN_ROI_FRAMES", "2")),
                   help="ROI close-ups added to a judgement")
    p.add_argument("--roi-pad", type=float, default=0.25,
                   help="padding around the ROI, as a fraction of the box")
    p.add_argument("--window-s", type=float, default=float(env("FOREMAN_WINDOW_S", "3.0")),
                   help="rolling evidence window in seconds")
    p.add_argument("--evidence-frames", type=int, default=int(env("FOREMAN_EVIDENCE_FRAMES", "6")),
                   help="representative frames sent to the VLM per inspection")
    p.add_argument("--capture-s", type=float, default=float(env("FOREMAN_CAPTURE_S", "3.0")),
                   help="manual 'Inspect now' capture length; sets buffer retention")
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

    LATEST.window_s = args.window_s
    # Keep enough history for a manual capture, which asks for a longer span
    # than the rolling window and would otherwise be pruned mid-capture.
    LATEST.retain_s = max(args.window_s, args.capture_s) + 2.0
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
