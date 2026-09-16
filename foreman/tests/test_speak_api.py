"""HTTP contract for POST /api/standard/speak.

The safety property under test is at the route level, not one layer down: a
recording Whisper could not read must leave the standard in force untouched. A
handler that adopted the transcript regardless of `accepted` would still pass the
orchestrator-level tests, so it is checked here against the real FastAPI app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Point the module-level client at nothing reachable; every test stubs the call.
os.environ.setdefault("FOREMAN_EDGE_URL", "http://127.0.0.1:1")

from fastapi.testclient import TestClient  # noqa: E402

from host import app as app_module  # noqa: E402
from host.edge_client import EdgeError, Transcript  # noqa: E402

# Not used as a context manager on purpose: that would run the lifespan and try
# to connect to a DevKit. The route is what is under test.
client = TestClient(app_module.app)


def _transcript(**kw) -> Transcript:
    base = {"text": "человек должен держать телефон", "language": "ru", "mode": "auto",
            "accepted": True, "reject_reason": "", "avg_logprob": -0.05,
            "no_speech_prob": 0.004,
            "metrics": {"inference_ms": 490.0, "asr_calls": 2.0}}
    base.update(kw)
    return Transcript(**base)


@pytest.fixture(autouse=True)
def _reset():
    app_module.orchestrator.set_standard("the lid must be closed")
    yield


def _stub(monkeypatch, result):
    async def fake(_audio, _filename="speech.wav", language="auto"):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(app_module.orchestrator.edge, "transcribe", fake)


def _post(language="auto", audio=b"RIFFfake"):
    return client.post(f"/api/standard/speak?language={language}",
                       files={"file": ("speech.webm", audio, "audio/webm")})


def test_accepted_speech_becomes_the_standard(monkeypatch):
    _stub(monkeypatch, _transcript())
    r = _post()
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["standard"] == "человек должен держать телефон"
    assert body["language"] == "ru"
    assert app_module.orchestrator.standard == "человек должен держать телефон"


def test_the_decoded_language_is_stored_not_the_requested_one(monkeypatch):
    """Asking for "auto" and decoding Russian must store Russian, or the standard
    would later be re-parsed with the English lexicon and ground nothing."""
    _stub(monkeypatch, _transcript(mode="auto", language="ru"))
    r = _post(language="auto")
    assert r.status_code == 200
    assert app_module.orchestrator.standard_language == "ru"
    assert r.json()["parsed"]["required"] == ["person", "cell phone"]


def test_rejected_speech_leaves_the_standard_in_force(monkeypatch):
    _stub(monkeypatch, _transcript(
        text="СПОКОЙНАЯ МУЗЫКА", accepted=False, no_speech_prob=0.905,
        avg_logprob=-0.529, reject_reason="the recording does not appear to contain speech"))
    r = _post()
    assert r.status_code == 200, "a rejection is a result, not a transport error"
    body = r.json()
    assert body["accepted"] is False
    assert body["reason"]
    assert body["standard"] == "the lid must be closed"
    assert app_module.orchestrator.standard == "the lid must be closed"


def test_rejected_speech_reports_no_parse(monkeypatch):
    """Nothing was adopted, so there is no new parse to show."""
    _stub(monkeypatch, _transcript(accepted=False, reject_reason="unclear"))
    assert "parsed" not in _post().json()


def test_empty_upload_is_refused(monkeypatch):
    _stub(monkeypatch, _transcript())
    assert _post(audio=b"").status_code == 400
    assert app_module.orchestrator.standard == "the lid must be closed"


@pytest.mark.parametrize("language", ["bg", "uk", "de", "", "EN"])
def test_only_en_ru_auto_are_accepted(monkeypatch, language):
    """A third language must be refused at the door rather than passed through
    to Whisper, which would happily decode Bulgarian."""
    _stub(monkeypatch, _transcript())
    assert _post(language=language).status_code == 400
    assert app_module.orchestrator.standard == "the lid must be closed"


def test_asr_failure_is_a_503_and_keeps_the_standard(monkeypatch):
    _stub(monkeypatch, EdgeError("transcribe failed: connection refused"))
    r = _post()
    assert r.status_code == 503
    assert app_module.orchestrator.standard == "the lid must be closed"


def test_typed_standard_reports_its_parse():
    r = client.post("/api/standard", json={"text": "The person must be holding a phone."})
    assert r.status_code == 200
    assert r.json()["parsed"]["required"] == ["person", "cell phone"]


def test_typed_standard_rejects_an_empty_string():
    assert client.post("/api/standard", json={"text": "   "}).status_code == 400
