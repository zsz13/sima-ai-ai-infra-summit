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


async def _orch(edge_url, tmp_path, **gate_kwargs) -> Orchestrator:
    cfg = GateConfig(stable_frames=3, absent_frames=3, **gate_kwargs)
    return Orchestrator(edge=EdgeClient(edge_url), audit_dir=tmp_path, gate_config=cfg)


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
        assert rec.metrics["inference_ms"] == 1234.5
        assert "end_to_end_ms" in rec.metrics
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
