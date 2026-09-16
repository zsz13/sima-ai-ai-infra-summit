"""Integration test: orchestrator <-> a real HTTP edge (the test harness).

Exercises the actual wire contract over a real socket - SSE parsing, gating,
the inspect round trip, audit persistence and reconnection - with no DevKit.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host.edge_client import EdgeClient, EdgeError  # noqa: E402
from host.gate import GateConfig  # noqa: E402
from host.orchestrator import Orchestrator  # noqa: E402
from tests.fake_edge import STATE, app  # noqa: E402

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def edge_url():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        threading.Event().wait(0.05)
    else:
        pytest.fail("fake edge did not start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_harness():
    STATE.update({
        "window_frames": 45,
        "per_frame": [True, True, True],
        "vlm_evidence": ["a box is visible"],
        "vlm_missing": [],
        "detector_summary": "- box: detected in 45/45 frames (100%).",
        "present": False,
        "bbox": [0.30, 0.30, 0.60, 0.60],
        "verdict": "pass",
        "reason": "Synthetic verdict from the test harness. Not a real inference.",
        "inspect_delay": 0.05,
        "inspect_calls": 0,
        "fps": 60.0,
    })
    yield


async def _wait_for(predicate, timeout=10.0, interval=0.05):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def _orch(edge_url, tmp_path, mode: str = "auto", **gate_kwargs) -> Orchestrator:
    """An orchestrator for tests.

    Defaults to auto because most of these tests exercise the gate; the product
    default is manual, which is covered explicitly below.
    """
    cfg = GateConfig(stable_frames=3, absent_frames=3, **gate_kwargs)
    return Orchestrator(edge=EdgeClient(edge_url), audit_dir=tmp_path,
                        gate_config=cfg, inspection_mode=mode)


# --- client-level ------------------------------------------------------

async def test_health_reports_ok(edge_url):
    client = EdgeClient(edge_url)
    try:
        assert (await client.health())["ok"] is True
    finally:
        await client.aclose()


async def test_health_raises_edge_error_when_unreachable():
    client = EdgeClient(f"http://127.0.0.1:{_free_port()}")
    try:
        with pytest.raises(EdgeError):
            await client.health()
    finally:
        await client.aclose()


async def test_event_stream_parses_detections(edge_url):
    STATE["present"] = True
    client = EdgeClient(edge_url)
    try:
        seen = []
        async for frame in client.events():
            seen.append(frame)
            if len(seen) >= 3:
                break
        assert all(f.detections[0].label == "box" for f in seen)
        assert seen[0].detections[0].bbox == (0.30, 0.30, 0.60, 0.60)
        assert [f.frame_id for f in seen] == sorted(f.frame_id for f in seen)
    finally:
        await client.aclose()


async def test_unknown_verdict_string_is_coerced_to_unclear(edge_url):
    STATE["verdict"] = "definitely-maybe"
    client = EdgeClient(edge_url)
    try:
        assert (await client.inspect("anything")).verdict == "unclear"
    finally:
        await client.aclose()


async def test_transcribe_returns_text(edge_url):
    client = EdgeClient(edge_url)
    try:
        t = await client.transcribe(b"RIFF....fake wav")
        assert "label facing up" in t.text
        assert t.language == "en"
    finally:
        await client.aclose()


# --- orchestrator end to end -------------------------------------------

async def test_inspects_once_when_item_settles(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1), "no inspection recorded"
        await asyncio.sleep(0.5)
        assert orch.counters.total == 1, "item must be inspected exactly once"
        assert orch.counters.passed == 1
        rec = orch.recent[-1]
        assert rec.verdict == "pass"
        assert rec.trigger_label == "box"
        assert rec.metrics["inference_ms"] == 3050.0   # temporal multi-image call
        assert "end_to_end_ms" in rec.metrics
        assert rec.decided_by == "vlm"
        assert len(rec.evidence_paths) == orch.evidence_frames, \
            "one evidence image per representative frame"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_does_not_inspect_without_a_standard(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        await asyncio.sleep(1.0)
        assert orch.counters.total == 0
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_second_item_inspected_after_first_leaves(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("check the lid")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total == 1)
        STATE["present"] = False
        await asyncio.sleep(0.4)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total == 2), "second item not inspected"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_failing_verdict_is_counted_and_persisted(edge_url, tmp_path):
    STATE["verdict"] = "fail"
    STATE["reason"] = "The lid is open."
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("lid must be closed")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1)
        assert orch.counters.failed == 1

        audit = tmp_path / "inspections.jsonl"
        assert audit.exists()
        rows = [json.loads(line) for line in audit.read_text().splitlines()]
        assert rows[-1]["verdict"] == "fail"
        assert rows[-1]["reason"] == "The lid is open."
        assert rows[-1]["standard"] == "lid must be closed"

        evidence = tmp_path / "evidence"
        assert list(evidence.glob("*.jpg")), "evidence frame not written"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_setting_a_new_standard_rearms_the_gate(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("first standard")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total == 1)
        # Same item still in frame, but the standard changed: re-inspect it.
        orch.set_standard("second standard")
        assert await _wait_for(lambda: orch.counters.total == 2), "new standard did not re-arm"
        assert orch.recent[-1].standard == "second standard"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_snapshot_reports_live_state(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("s")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.frames_seen > 10)
        snap = orch.snapshot()
        assert snap["connected"] is True
        assert snap["fps"] > 0
        assert snap["detections_last_frame"] == 1
        assert snap["counters"]["total"] == snap["counters"]["passed"] + \
            snap["counters"]["failed"] + snap["counters"]["unclear"]
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_reports_disconnected_when_edge_is_down(tmp_path):
    orch = Orchestrator(
        edge=EdgeClient(f"http://127.0.0.1:{_free_port()}"),
        audit_dir=tmp_path,
        gate_config=GateConfig(stable_frames=3),
    )
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.last_error is not None, timeout=5.0)
        assert orch.connected is False
        assert orch.snapshot()["connected"] is False
    finally:
        await orch.stop()
        await orch.edge.aclose()


# --- temporal grounding, over the real HTTP contract -------------------

async def test_detector_grounding_overrides_a_hallucinating_vlm(edge_url, tmp_path):
    """Case 1/7 end to end: the edge's model says PASS, the detector never saw a
    phone, and the Mac must still refuse to pass it."""
    STATE["verdict"] = "pass"
    STATE["reason"] = "The person is clearly holding a smartphone."
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("the person must be holding a smartphone")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True          # SSE emits "box", never person/cell phone
        assert await _wait_for(lambda: orch.counters.total >= 1)
        rec = orch.recent[-1]
        assert rec.verdict == "fail", "a hallucinated phone must never pass"
        assert rec.decided_by == "detector-absent"
        assert any("Detector evidence wins" in n for n in rec.notes)
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_window_and_frame_metadata_are_recorded(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("the lid must be closed")     # ungrounded -> VLM decides
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1)
        rec = orch.recent[-1]
        assert rec.window["total_frames"] == 45
        assert rec.window["duration_s"] > 0
        assert len(rec.frames) == orch.evidence_frames
        assert [f["rel_ts"] for f in rec.frames] == sorted(f["rel_ts"] for f in rec.frames)
        assert all(f["path"] for f in rec.frames)
        # Agreement is counted over distinct temporal frames, never ROI crops.
        assert rec.vlm["frames_judged"] == orch.evidence_frames
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_contradictory_per_frame_evidence_is_unclear_end_to_end(edge_url, tmp_path):
    STATE["verdict"] = "pass"
    STATE["per_frame"] = [True, False, True]
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("the lid must be closed")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1)
        assert orch.recent[-1].verdict == "unclear"
        assert orch.recent[-1].decided_by == "vlm-contradictory"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_single_frame_fallback_still_works(edge_url, tmp_path):
    """FOREMAN_TEMPORAL=0 must keep the previously demonstrated pipeline runnable."""
    orch = await _orch(edge_url, tmp_path)
    orch.temporal = False
    orch.set_standard("the lid must be closed")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1)
        rec = orch.recent[-1]
        assert rec.decided_by == "single-frame-fallback"
        assert rec.metrics["inference_ms"] == 1234.5
        assert len(rec.evidence_paths) == 1
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_audit_record_keeps_the_original_fields(edge_url, tmp_path):
    """Existing readers of inspections.jsonl must not break."""
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("the lid must be closed")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1)
        row = json.loads((tmp_path / "inspections.jsonl").read_text().splitlines()[-1])
        for key in ("id", "ts", "verdict", "reason", "standard", "trigger_label",
                    "trigger_confidence", "metrics", "evidence_path"):
            assert key in row, f"backward-compatible field {key} missing"
        assert row["evidence_path"].startswith("evidence/")
        for key in ("decided_by", "evidence_paths", "window", "required_objects",
                    "prohibited_objects", "vlm", "notes", "frames"):
            assert key in row, f"temporal field {key} missing"
    finally:
        await orch.stop()
        await orch.edge.aclose()


# --- Camera Check ------------------------------------------------------
#
# Camera Check is a detector-only debugging view. It must observe without
# consuming: pausing is not the same as inspecting.

async def test_camera_check_does_not_consume_the_item(edge_url, tmp_path):
    """Leaving Camera Check must not leave the console idle.

    The gate fires once per item and then latches until that item leaves frame.
    If it were allowed to advance while paused, the item in view would already be
    marked inspected on the way out, and the operator would see nothing happen
    for no visible reason.
    """
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    orch.paused = True
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        await asyncio.sleep(1.0)
        assert orch.counters.total == 0, "Camera Check must not inspect"

        orch.paused = False
        assert await _wait_for(lambda: orch.counters.total >= 1), \
            "the item in view was consumed while paused and never inspected"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_camera_check_writes_no_verdict_records(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    orch.paused = True
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        await asyncio.sleep(1.2)
        assert orch.counters.total == 0
        assert orch.recent == [] or len(orch.recent) == 0
        assert not (tmp_path / "inspections.jsonl").exists()
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_camera_check_keeps_the_standard(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("Человек должен держать телефон.")
    await orch.start()
    try:
        orch.paused = True
        await asyncio.sleep(0.3)
        assert orch.standard == "Человек должен держать телефон."
        orch.paused = False
        assert orch.standard == "Человек должен держать телефон."
        assert orch.parsed_standard()["required"] == ["person", "cell phone"]
    finally:
        await orch.stop()
        await orch.edge.aclose()


# --- bilingual speech --------------------------------------------------

async def test_transcribe_honours_a_forced_language(edge_url):
    client = EdgeClient(edge_url)
    try:
        ru = await client.transcribe(b"RIFF....fake wav", language="ru")
        assert ru.language == "ru"
        assert ru.accepted
        assert ru.metrics["asr_calls"] == 1.0
        en = await client.transcribe(b"RIFF....fake wav", language="en")
        assert en.language == "en"
    finally:
        await client.aclose()


async def test_auto_mode_decodes_twice(edge_url):
    """Auto EN/RU decodes the clip both ways rather than asking Whisper to guess."""
    client = EdgeClient(edge_url)
    try:
        t = await client.transcribe(b"RIFF....fake wav", language="auto")
        assert t.mode == "auto"
        assert t.language in ("en", "ru")
        assert t.metrics["asr_calls"] == 2.0
    finally:
        await client.aclose()


async def test_unclear_speech_does_not_replace_the_standard(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("the lid must be closed")
    STATE["speech_unclear"] = True
    try:
        t = await orch.edge.transcribe(b"RIFF....silence", language="auto")
        assert not t.accepted
        assert t.reject_reason
        # the caller must not adopt it
        if t.accepted:
            orch.set_standard(t.text)
        assert orch.standard == "the lid must be closed"
    finally:
        STATE["speech_unclear"] = False
        await orch.edge.aclose()


async def test_russian_standard_is_detector_grounded(edge_url, tmp_path):
    """A Russian rule must ground exactly like its English twin.

    If it parsed to nothing, host/policy.py would have no detector evidence to
    weigh and would defer entirely to the vision-language model - removing the
    protection temporal grounding exists to provide.
    """
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("Человек должен держать телефон.", language="ru")
    parsed = orch.parsed_standard()
    assert parsed["language"] == "ru"
    assert parsed["grounded"] is True
    assert parsed["required"] == ["person", "cell phone"]
    await orch.edge.aclose()


@pytest.mark.parametrize(("standard", "language"), [
    ("The person must be holding a phone.", "en"),
    ("Человек должен держать телефон.", "ru"),
])
async def test_a_missing_object_fails_in_either_language(edge_url, tmp_path,
                                                         standard, language):
    """Drive a whole inspection, not just the parse.

    The detector sees a person and never a phone, and the harness's model is
    rigged to say PASS. The verdict must still be FAIL, decided by the detector,
    in both languages - which is only possible if the Russian standard reached
    `build_evidence` with the same COCO classes as the English one.
    """
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard(standard, language=language)
    STATE["label"] = "person"
    STATE["verdict"] = "pass"
    STATE["reason"] = "The person is holding a smartphone."
    STATE["detector_summary"] = "- person: detected in 45/45 frames (100%)."
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1), "no inspection recorded"
        rec = orch.recent[-1]
        assert rec.verdict == "fail", "the model must not conjure a phone the detector never saw"
        assert rec.decided_by == "detector-absent"
        labels = {o["label"] for o in rec.required_objects}
        assert labels == {"person", "cell phone"}
        absent = next(o for o in rec.required_objects if o["label"] == "cell phone")
        assert absent["frames_present"] == 0
    finally:
        STATE.update(present=False, label="box", verdict="pass",
                     reason="Synthetic verdict from the test harness. Not a real inference.",
                     detector_summary="- box: detected in 45/45 frames (100%).")
        await orch.stop()
        await orch.edge.aclose()


# --- Clear session must really clear ------------------------------------
#
# The bug: clear_session() reset the counters and the gate but left the standard
# in force, so the gate re-armed and the next object in view was immediately
# inspected against the rule the operator had just tried to discard.

async def test_clear_session_removes_the_standard(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    assert orch.standard
    orch.clear_session()
    assert orch.standard == "", "the standard survived Clear session"
    assert orch.parsed_standard()["required"] == []


async def test_no_inspection_happens_after_clear_with_objects_in_view(edge_url, tmp_path):
    """Clear, then leave an object sitting in frame. Nothing may be inspected."""
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1), "no baseline inspection"

        orch.clear_session()
        before = orch.counters.total
        assert before == 0, "counters were not reset"
        # the object stays in view for several gate cycles
        await asyncio.sleep(1.5)
        assert orch.counters.total == 0, "an inspection fired with no standard set"
        assert orch.standard == "", "the old standard came back"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_a_new_typed_standard_works_after_clear(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("first standard")
    orch.clear_session()
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1), \
            "inspection did not resume after setting a new standard"
        assert orch.recent[-1].standard == "every box must have a label facing up"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_a_spoken_standard_works_after_clear(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("first standard")
    orch.clear_session()
    t = await orch.edge.transcribe(b"RIFF....fake wav", language="en")
    assert t.accepted
    assert orch.set_standard(t.text, language=t.language, epoch=orch.session_epoch)
    assert orch.standard == t.text
    await orch.edge.aclose()


async def test_a_transcript_that_arrives_after_a_clear_is_discarded(edge_url, tmp_path):
    """Recognition takes about a second. A clear during that window must win."""
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("first standard")
    epoch = orch.session_epoch                 # what the request captured
    orch.clear_session()                       # operator clears while ASR runs
    applied = orch.set_standard("a late transcript", epoch=epoch)
    assert applied is False, "a stale transcript reinstated a standard"
    assert orch.standard == "", "the cleared standard came back"


async def test_a_standard_set_within_the_current_session_still_applies(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    orch.clear_session()
    assert orch.set_standard("fresh", epoch=orch.session_epoch) is True
    assert orch.standard == "fresh"


async def test_an_inspection_finishing_after_a_clear_is_not_recorded(edge_url, tmp_path):
    """A verdict must not be counted against a standard no longer in force."""
    orch = await _orch(edge_url, tmp_path)
    orch.set_standard("every box must have a label facing up")
    STATE["inspect_delay"] = 1.0               # long enough to clear mid-flight
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.inspecting, timeout=6.0), "never started"
        orch.clear_session()                   # cleared while the VLM is running
        await asyncio.sleep(2.0)
        assert orch.counters.total == 0, "a stale verdict was recorded after clear"
        assert orch.standard == ""
    finally:
        STATE["present"] = False
        STATE["inspect_delay"] = 0.05
        await orch.stop()
        await orch.edge.aclose()


# --- manual vs auto -----------------------------------------------------
#
# Manual is the product default. Setting a standard arms the system; only
# "Inspect now" starts an inspection. The two modes never run at once.

async def test_manual_is_the_default_mode(edge_url, tmp_path):
    orch = Orchestrator(edge=EdgeClient(edge_url), audit_dir=tmp_path)
    assert orch.inspection_mode == "manual"
    await orch.edge.aclose()


async def test_setting_a_standard_in_manual_mode_starts_nothing(edge_url, tmp_path):
    """The reported UX problem: typing a rule should arm, not fire."""
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        STATE["present"] = True
        await asyncio.sleep(2.0)                 # several gate cycles
        assert orch.counters.total == 0, "an inspection started without Inspect now"
        assert orch.standard, "the standard should still be armed"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_inspect_now_works_immediately_in_manual_mode(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        started = await orch.inspect_now(capture_s=0.5)
        assert started["started"] is True, "Inspect now was not available straight away"
        assert await _wait_for(lambda: orch.counters.total >= 1, timeout=15.0)
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_inspect_now_is_refused_without_a_standard(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.connected = True
    result = await orch.inspect_now()
    assert result["started"] is False
    assert "no standard" in result["reason"]


async def test_inspect_now_is_refused_while_disconnected(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("x")
    orch.connected = False
    result = await orch.inspect_now()
    assert result["started"] is False
    assert "not connected" in result["reason"]


async def test_inspect_now_is_refused_while_one_is_running(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("x")
    orch.connected = True
    orch.inspecting = True
    result = await orch.inspect_now()
    assert result["started"] is False
    assert "already running" in result["reason"]


async def test_switching_auto_to_manual_stops_triggering_but_keeps_the_standard(
        edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="auto")
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        assert await _wait_for(lambda: orch.counters.total >= 1), "auto never fired"

        orch.set_inspection_mode("manual")
        assert orch.standard, "switching mode must not clear the standard"
        STATE["present"] = False
        await asyncio.sleep(0.6)
        STATE["present"] = True
        before = orch.counters.total
        await asyncio.sleep(2.0)
        assert orch.counters.total == before, "auto kept firing after switching to manual"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_switching_manual_to_auto_begins_monitoring(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("every box must have a label facing up")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        STATE["present"] = True
        await asyncio.sleep(1.5)
        assert orch.counters.total == 0
        orch.set_inspection_mode("auto")
        assert await _wait_for(lambda: orch.counters.total >= 1, timeout=10.0), \
            "auto did not start monitoring with the existing standard"
    finally:
        STATE["present"] = False
        await orch.stop()
        await orch.edge.aclose()


async def test_an_invalid_mode_is_rejected(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path)
    with pytest.raises(ValueError):
        orch.set_inspection_mode("sometimes")
    await orch.edge.aclose()


# --- at most two rules -------------------------------------------------------

async def test_a_single_rule_is_stored_as_one_rule(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("The person must be holding a phone.")
    assert orch.rules == ["The person must be holding a phone."]
    assert orch.standard == "The person must be holding a phone.", "backward compatible"
    await orch.edge.aclose()


async def test_two_rules_are_kept_separate_and_verbatim(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["The person must be holding a phone.",
                    "The person must NOT be holding a bottle."])
    assert orch.rules == ["The person must be holding a phone.",
                          "The person must NOT be holding a bottle."]
    assert orch.standard == "The person must be holding a phone."
    await orch.edge.aclose()


async def test_a_third_rule_is_refused(edge_url, tmp_path):
    """Two is the supported maximum and the limit is enforced, not assumed."""
    orch = await _orch(edge_url, tmp_path, mode="manual")
    with pytest.raises(ValueError, match="at most two"):
        orch.set_rules(["a person must be visible", "no phone in view", "no bottle in view"])
    await orch.edge.aclose()


async def test_blank_rules_are_dropped_not_stored(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["A person must be visible.", "   "])
    assert orch.rules == ["A person must be visible."]
    await orch.edge.aclose()


async def test_each_rule_is_parsed_independently(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["The person must be holding a phone.",
                    "The person must NOT be holding a bottle."])
    parsed = orch.parsed_rules()
    assert len(parsed) == 2
    assert parsed[0]["relation_object"] == "cell phone"
    assert parsed[0]["relation_expected"] is True
    assert parsed[1]["relation_object"] == "bottle"
    assert parsed[1]["relation_expected"] is False
    await orch.edge.aclose()


async def test_the_detector_tracks_the_union_of_both_rules(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["The person must be holding a phone.",
                    "The person must NOT be holding a bottle."])
    assert set(orch.tracked_objects()) == {"person", "cell phone", "bottle"}
    await orch.edge.aclose()


async def test_rules_survive_a_language_mix(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["A person must be visible.", "Человек не должен держать телефон."])
    parsed = orch.parsed_rules()
    assert parsed[0]["language"] == "en"
    assert parsed[1]["language"] == "ru"
    await orch.edge.aclose()


async def test_clear_session_removes_both_rules(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["A person must be visible.", "There must be no bottle in view."])
    orch.clear_session()
    assert orch.rules == []
    assert orch.standard == ""
    assert orch.parsed_rules() == []
    await orch.edge.aclose()


async def test_the_snapshot_exposes_both_rules_for_the_console(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(["A person must be visible.", "There must be no bottle in view."])
    snap = orch.snapshot()
    assert snap["rules"] == ["A person must be visible.", "There must be no bottle in view."]
    assert len(snap["parsed_rules"]) == 2
    await orch.edge.aclose()


# --- one capture, two rule evaluations ---------------------------------------

async def _two_rule_inspection(edge_url, tmp_path, rules, rule_results,
                               detector_summary=None, present=True,
                               labels=("person", "cell phone", "bottle")):
    """Run one manual inspection against two rules and return the record."""
    import tests.fake_edge as fake
    fake.STATE["rule_results"] = rule_results
    fake.STATE["present"] = present
    fake.STATE["labels"] = labels
    if detector_summary:
        fake.STATE["detector_summary"] = detector_summary
    before = fake.STATE["inspect_calls"]
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_rules(rules)
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected), "never connected"
        # The policy needs a real window before it will judge anything, so let
        # the ring fill the way it does in production.
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20), "ring never filled"
        await orch.inspect_now()
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=20), "no verdict"
        rec = orch.recent[0]
        calls = fake.STATE["inspect_calls"] - before
    finally:
        await orch.stop()
        await orch.edge.aclose()
        fake.STATE["rule_results"] = None
        fake.STATE["labels"] = None
    return rec, calls


async def test_two_rules_cost_exactly_one_capture(edge_url, tmp_path):
    """The requirement that matters: one window, not one per rule."""
    rec, calls = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("pass", "No bottle held.", "not_holding")])
    assert calls == 1, f"two rules triggered {calls} captures"


async def test_both_rules_share_one_evidence_set(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("pass", "No bottle held.", "not_holding")])
    assert len(rec.rules) == 2
    # One window, one set of images, shared by both rules.
    assert len(rec.frames) == rec.window["image_frames"] or len(rec.frames) > 0
    assert rec.window["total_frames"] > 0
    paths = rec.evidence_paths
    assert paths, "evidence frames were not stored"
    assert len(set(paths)) == len(paths), "evidence images duplicated per rule"


async def test_case_A_pass_and_pass_is_overall_pass(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("pass", "No bottle held.", "not_holding")])
    assert [r["verdict"] for r in rec.rules] == ["pass", "pass"], \
        [(r["verdict"], r["decided_by"], r["reason"][:70]) for r in rec.rules]
    assert rec.verdict == "pass"


async def test_case_B_pass_and_fail_is_overall_fail(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("fail", "Bottle is held.", "holding")])
    assert [r["verdict"] for r in rec.rules] == ["pass", "fail"]
    assert rec.verdict == "fail"


async def test_case_D_pass_and_unclear_is_overall_unclear(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("unclear", "Cannot tell.", "unclear")])
    assert [r["verdict"] for r in rec.rules] == ["pass", "unclear"]
    assert rec.verdict == "unclear"


async def test_each_rule_keeps_its_own_text_and_reason(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("fail", "Bottle is held.", "holding")])
    assert rec.rules[0]["text"] == "The person must be holding a phone."
    assert rec.rules[1]["text"] == "The person must NOT be holding a bottle."
    # Rule 1 must not talk about the bottle, and rule 2 must not talk about the phone.
    assert "bottle" not in rec.rules[0]["reason"].lower()
    assert "phone" not in rec.rules[1]["reason"].lower()


async def test_each_rule_records_its_normalised_semantics(edge_url, tmp_path):
    rec, _ = await _two_rule_inspection(
        edge_url, tmp_path,
        ["The person must be holding a phone.", "The person must NOT be holding a bottle."],
        [("pass", "Phone is held.", "holding"), ("pass", "No bottle.", "not_holding")])
    assert rec.rules[0]["parsed"]["relation_expected"] is True
    assert rec.rules[1]["parsed"]["relation_expected"] is False
    assert rec.rules[0]["parsed"]["relation_object"] == "cell phone"
    assert rec.rules[1]["parsed"]["relation_object"] == "bottle"


async def test_a_single_rule_record_is_unchanged(edge_url, tmp_path):
    """One rule must produce exactly the record shape it always did."""
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        await orch.inspect_now()
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=20)
        rec = orch.recent[0]
    finally:
        await orch.stop()
        await orch.edge.aclose()
    assert rec.standard == "A person must be visible."
    assert rec.verdict in ("pass", "fail", "unclear")
    assert rec.reason
    assert len(rec.rules) == 1, "a single rule is still one rule in the record"
    assert rec.rules[0]["text"] == "A person must be visible."


# --- manual preparation delay ------------------------------------------------
#
# Pressing "Inspect now" gives the operator a moment to get into position before
# anything is recorded. The preparation period is deliberately NOT evidence: the
# detector keeps running for the live view, but none of those frames may reach
# the inspection.

async def test_manual_has_a_preparation_delay_before_capture(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        press = time.time()
        await orch.inspect_now()
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=25)
        rec = orch.recent[0]
    finally:
        await orch.stop()
        await orch.edge.aclose()
    started = rec.capture["start_ts"]
    delay = started - press
    assert delay >= orch.prepare_s * 0.8, (
        f"capture began {delay:.3f}s after the press, before preparation finished")
    assert rec.capture["prepare_s"] == orch.prepare_s
    assert rec.capture["prepared_at"] >= press


async def test_the_preparation_period_is_not_part_of_the_evidence(edge_url, tmp_path):
    """The window must begin after preparation, never at the button press."""
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        press = time.time()
        await orch.inspect_now()
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=25)
        rec = orch.recent[0]
    finally:
        await orch.stop()
        await orch.edge.aclose()
    assert rec.window["start_ts"] >= press, "evidence predates the button press"
    assert rec.capture["start_ts"] >= rec.capture["prepared_at"], \
        "capture started before preparation finished"


async def test_capture_is_still_three_seconds_and_six_frames(edge_url, tmp_path):
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        await orch.inspect_now()
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=25)
        rec = orch.recent[0]
    finally:
        await orch.stop()
        await orch.edge.aclose()
    assert rec.capture["requested_s"] == 3.0, "the capture length must not change"
    assert len(rec.frames) == 6, f"expected 6 evidence frames, got {len(rec.frames)}"


async def test_the_phase_is_preparing_before_it_is_capturing(edge_url, tmp_path):
    """The console must be able to tell the two apart."""
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        await orch.inspect_now()
        assert await _wait_for(lambda: orch.capture_phase == "preparing", timeout=5), \
            "never reported a preparing phase"
        assert await _wait_for(lambda: orch.capture_phase == "manual", timeout=10), \
            "never moved on to capturing"
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_clear_session_during_preparation_cancels_the_inspection(edge_url, tmp_path):
    import tests.fake_edge as fake
    before = fake.STATE["inspect_calls"]
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        await orch.inspect_now()
        assert await _wait_for(lambda: orch.capture_phase == "preparing", timeout=5)
        orch.clear_session()
        await asyncio.sleep(orch.prepare_s + 1.5)
        assert len(orch.recent) == 0, "a verdict was recorded after Clear session"
        assert orch.counters.total == 0
        assert fake.STATE["inspect_calls"] == before, \
            "capture was started despite the session being cleared"
        assert orch.capture_phase == "idle"
        assert not orch.inspecting
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_a_disconnect_during_preparation_does_not_start_a_stale_capture(edge_url, tmp_path):
    import tests.fake_edge as fake
    before = fake.STATE["inspect_calls"]
    orch = await _orch(edge_url, tmp_path, mode="manual")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        await orch.inspect_now()
        assert await _wait_for(lambda: orch.capture_phase == "preparing", timeout=5)
        orch.connected = False          # the edge went away mid-preparation
        await asyncio.sleep(orch.prepare_s + 1.5)
        assert fake.STATE["inspect_calls"] == before, \
            "a capture was started after the edge disconnected"
        assert len(orch.recent) == 0
        assert orch.capture_phase == "idle"
        assert not orch.inspecting
    finally:
        await orch.stop()
        await orch.edge.aclose()


async def test_auto_mode_has_no_preparation_delay(edge_url, tmp_path):
    """Auto is unchanged: the gate fires and the rolling window is judged."""
    STATE["present"] = True          # the gate needs something to settle on
    orch = await _orch(edge_url, tmp_path, mode="auto")
    orch.set_standard("A person must be visible.")
    await orch.start()
    try:
        assert await _wait_for(lambda: orch.connected)
        assert await _wait_for(lambda: orch.frames_seen >= 20, timeout=20)
        assert await _wait_for(lambda: len(orch.recent) > 0, timeout=25), "auto never fired"
        rec = orch.recent[0]
    finally:
        await orch.stop()
        await orch.edge.aclose()
    assert rec.capture.get("mode") != "manual"
    assert "prepare_s" not in rec.capture, "auto must not carry a preparation delay"
