"""HTTP contract for POST /api/standard/speak.

The safety property under test is at the route level, not one layer down: a
recording Whisper could not read must leave the standard in force untouched. A
handler that adopted the transcript regardless of `accepted` would still pass the
orchestrator-level tests, so it is checked here against the real FastAPI app.
"""

from __future__ import annotations

import io as _io
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


# --- selected export over HTTP -------------------------------------------

def test_export_endpoints_accept_an_ids_filter(tmp_path, monkeypatch):
    """The ids filter must reach the export, and an unknown id must not 500."""
    import json as _json

    trail = tmp_path / "inspections.jsonl"
    recs = [{"id": f"rec{i}", "ts": 1789527560.0 + i, "verdict": "pass",
             "reason": "ok", "standard": "человек должен держать телефон.",
             "metrics": {}, "frames": []} for i in range(3)]
    trail.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                     encoding="utf-8")
    monkeypatch.setattr(app_module, "AUDIT_FILE", trail)
    monkeypatch.setattr(app_module, "EVIDENCE_DIR", tmp_path / "evidence")

    import csv as _csv
    import io as _io

    all_rows = list(_csv.DictReader(_io.StringIO(client.get("/api/export.csv").text)))
    assert len(all_rows) == 3

    one = list(_csv.DictReader(_io.StringIO(client.get("/api/export.csv?ids=rec1").text)))
    assert [r["inspection_id"] for r in one] == ["rec1"]

    two = list(_csv.DictReader(_io.StringIO(client.get("/api/export.csv?ids=rec0,rec2").text)))
    assert [r["inspection_id"] for r in two] == ["rec0", "rec2"]

    ghost = list(_csv.DictReader(_io.StringIO(client.get("/api/export.csv?ids=nope").text)))
    assert ghost == []

    # a path-shaped id must be sanitised away, not resolved
    r = client.get("/api/export.csv?ids=../../etc/passwd")
    assert r.status_code == 200
    assert list(_csv.DictReader(_io.StringIO(r.text))) == []


def test_selected_zip_downloads_and_is_named_as_a_selection(tmp_path, monkeypatch):
    import json as _json
    import zipfile as _zip

    trail = tmp_path / "inspections.jsonl"
    trail.write_text(_json.dumps({"id": "abc", "ts": 1.0, "verdict": "pass",
                                  "reason": "ok", "standard": "s", "metrics": {}}) + "\n")
    monkeypatch.setattr(app_module, "AUDIT_FILE", trail)
    monkeypatch.setattr(app_module, "EVIDENCE_DIR", tmp_path / "evidence")
    r = client.get("/api/export.zip?ids=abc")
    assert r.status_code == 200
    assert "-selected-" in r.headers["content-disposition"]
    with _zip.ZipFile(_io.BytesIO(r.content)) as z:
        assert z.testzip() is None
        assert "report.csv" in z.namelist()


# --- setting one or two rules through the API --------------------------------

def test_the_api_still_accepts_a_single_text():
    r = client.post("/api/standard", json={"text": "A person must be visible."})
    assert r.status_code == 200
    body = r.json()
    assert body["standard"] == "A person must be visible."
    assert body["rules"] == ["A person must be visible."]


def test_the_api_accepts_two_rules():
    r = client.post("/api/standard", json={
        "rules": ["The person must be holding a phone.",
                  "The person must NOT be holding a bottle."]})
    assert r.status_code == 200
    body = r.json()
    assert body["rules"] == ["The person must be holding a phone.",
                             "The person must NOT be holding a bottle."]
    assert len(body["parsed_rules"]) == 2
    assert body["parsed_rules"][1]["relation_expected"] is False


def test_the_api_refuses_three_rules():
    r = client.post("/api/standard", json={
        "rules": ["A person must be visible.", "There must be no phone in view.",
                  "There must be no bottle in view."]})
    assert r.status_code == 400
    assert "two" in r.text.lower()


def test_the_api_refuses_an_empty_rule_list():
    assert client.post("/api/standard", json={"rules": []}).status_code == 400
    assert client.post("/api/standard", json={"rules": ["  "]}).status_code == 400


def test_state_exposes_the_rules_for_the_console():
    client.post("/api/standard", json={
        "rules": ["A person must be visible.", "There must be no bottle in view."]})
    body = client.get("/api/state").json()
    assert body["rules"] == ["A person must be visible.",
                             "There must be no bottle in view."]
    assert len(body["parsed_rules"]) == 2


def test_clear_session_empties_the_rules():
    client.post("/api/standard", json={"rules": ["A person must be visible.",
                                                 "There must be no bottle in view."]})
    client.post("/api/session/clear")
    body = client.get("/api/state").json()
    assert body["rules"] == []
    assert body["standard"] == ""
