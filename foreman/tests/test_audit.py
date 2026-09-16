"""The history view and the export are read-only views over the audit trail.

The property that matters: reading must never alter the trail, and a torn line
(a crash mid-write) must not make the whole history unreadable.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host.audit import (  # noqa: E402
    CSV_COLUMNS,
    MAX_LIMIT,
    read_records,
    summarise,
    to_csv,
)


def _record(**kw) -> dict:
    base = {
        "id": "abc123", "ts": 1789527569.9022238, "verdict": "fail",
        "reason": "No cell phone was found during the inspection window.",
        "standard": "человек должен держать телефон.",
        "trigger_label": "person", "trigger_confidence": 0.69,
        "metrics": {"inference_ms": 2395.2, "end_to_end_ms": 2499.0},
        "evidence_path": "evidence/1-1.jpg",
        "decided_by": "detector-absent",
        "evidence_paths": ["evidence/1-1.jpg"],
        "window": {"total_frames": 49, "duration_s": 2.988},
        "required_objects": [
            {"label": "person", "frames_present": 49, "frames_total": 49, "presence_ratio": 1.0},
            {"label": "cell phone", "frames_present": 0, "frames_total": 49, "presence_ratio": 0.0},
        ],
        "prohibited_objects": [],
        "vlm": {"frames_judged": 4},
        "notes": [],
        "frames": [{"path": "evidence/1-1.jpg", "rel_ts": 0.0, "detections": []}],
    }
    base.update(kw)
    return base


@pytest.fixture
def trail(tmp_path) -> Path:
    p = tmp_path / "inspections.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps(_record(id=f"rec{i}", ts=1789527560.0 + i),
                                ensure_ascii=False) + "\n")
    return p


# --- reading ------------------------------------------------------------

def test_reads_newest_first(trail):
    rows = read_records(trail)
    assert [r["id"] for r in rows] == ["rec4", "rec3", "rec2", "rec1", "rec0"]


def test_limit_keeps_the_most_recent(trail):
    rows = read_records(trail, limit=2)
    assert [r["id"] for r in rows] == ["rec4", "rec3"]


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert read_records(tmp_path / "nope.jsonl") == []


def test_a_torn_final_line_does_not_break_the_history(trail):
    """A crash mid-write leaves a partial line. One bad line must not cost the
    other records."""
    with trail.open("a", encoding="utf-8") as fh:
        fh.write('{"id":"torn","verdict":"fa')
    rows = read_records(trail)
    assert len(rows) == 5
    assert "torn" not in [r["id"] for r in rows]


def test_blank_lines_are_skipped(trail):
    with trail.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert len(read_records(trail)) == 5


def test_reading_never_modifies_the_trail(trail):
    before = trail.read_bytes()
    read_records(trail)
    to_csv(read_records(trail))
    assert trail.read_bytes() == before, "the audit trail must be read-only"


def test_limit_is_clamped(trail):
    assert read_records(trail, limit=0) is not None
    assert len(read_records(trail, limit=10**9)) == 5
    assert MAX_LIMIT >= 1


# --- summarising --------------------------------------------------------

def test_summary_carries_the_grounding_story():
    row = summarise(_record())
    assert row["verdict"] == "fail"
    assert row["decided_by"] == "detector-absent"
    assert row["required_summary"] == "person 49/49 (100%); cell phone 0/49 (0%)"
    assert row["prohibited_summary"] == ""
    assert row["vlm_ms"] == 2395.2
    assert row["end_to_end_ms"] == 2499.0
    assert row["iso"].startswith("2026-")


def test_summary_survives_a_sparse_record():
    """Pre-temporal records have none of the grounding fields."""
    row = summarise({"id": "x", "ts": 1789527560.0, "verdict": "pass", "reason": "ok"})
    assert row["required"] == []
    assert row["vlm_ms"] is None
    assert row["frames"] == []


def test_single_frame_record_still_offers_its_evidence():
    row = summarise({"id": "x", "ts": 1.0, "verdict": "pass",
                     "evidence_path": "evidence/old.jpg"})
    assert row["frames"] == [{"path": "evidence/old.jpg", "rel_ts": None, "detections": []}]


# --- export -------------------------------------------------------------

def test_csv_has_a_header_and_one_row_per_record(trail):
    rows = list(csv.DictReader(io.StringIO(to_csv(read_records(trail)))))
    assert len(rows) == 5
    assert list(rows[0].keys()) == list(CSV_COLUMNS)


def test_csv_carries_the_fields_that_matter():
    row = next(csv.DictReader(io.StringIO(to_csv([_record()]))))
    assert row["verdict"] == "fail"
    assert row["decided_by"] == "detector-absent"
    assert row["standard"] == "человек должен держать телефон."
    assert row["required_objects"] == "person 49/49 (100%); cell phone 0/49 (0%)"
    assert row["vlm_ms"] == "2395.2"
    assert row["end_to_end_ms"] == "2499.0"
    assert row["window_frames"] == "49"
    assert row["inspection_id"] == "abc123"


def test_csv_quotes_a_reason_containing_a_comma():
    text = to_csv([_record(reason="No phone, and no person either.")])
    row = next(csv.DictReader(io.StringIO(text)))
    assert row["reason"] == "No phone, and no person either."


def test_csv_of_nothing_is_still_a_valid_header():
    text = to_csv([])
    assert text.strip() == ",".join(CSV_COLUMNS)


def test_csv_leaves_missing_numbers_blank_not_zero():
    """A blank cell means "not recorded"; a 0 would read as "took no time"."""
    row = next(csv.DictReader(io.StringIO(to_csv([{"id": "x", "ts": 1.0, "verdict": "pass"}]))))
    assert row["vlm_ms"] == ""
    assert row["end_to_end_ms"] == ""
    assert row["window_frames"] == ""


# --- evidence export ----------------------------------------------------

import io as _io  # noqa: E402
import zipfile  # noqa: E402

from host.audit import (  # noqa: E402
    build_package,
    evidence_manifest,
    jpeg_size,
    package_name,
    safe_evidence_path,
    to_json,
)

#: The smallest JPEG that still carries a real SOF0 header, so jpeg_size has
#: something honest to read. 1x1, greyscale.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    + "08" * 64
    + "ffc0000b080001000101011100ffc40014000100000000000000000000000000000000"
    "03ffda0008010100003f00d2cf20ffd9"
)


@pytest.fixture
def evidence(tmp_path) -> Path:
    d = tmp_path / "evidence"
    d.mkdir()
    (d / "1-1.jpg").write_bytes(_TINY_JPEG)
    return d


def test_package_names_are_deterministic_and_collision_safe():
    rec = _record(id="abc123", ts=1789527569.0)
    a = package_name(rec, 0, "evidence/1-1.jpg")
    assert a == package_name(rec, 0, "evidence/1-1.jpg"), "must be stable across exports"
    assert a != package_name(rec, 1, "evidence/1-2.jpg"), "index must disambiguate"
    other = _record(id="def456", ts=1789527569.0)
    assert a != package_name(other, 0, "evidence/1-1.jpg"), "id must disambiguate"
    assert a.endswith(".jpg")


def test_package_name_sanitises_a_hostile_id():
    name = package_name(_record(id="../../etc/passwd"), 0, "x.jpg")
    assert "/" not in name and ".." not in name


# --- path traversal -----------------------------------------------------

def test_evidence_outside_the_directory_is_refused(evidence, tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"not yours")
    for hostile in ("../secret.jpg", "../../etc/passwd",
                    "evidence/../../secret.jpg", "/etc/passwd"):
        assert safe_evidence_path(evidence, hostile) is None, hostile


def test_a_legitimate_reference_resolves(evidence):
    assert safe_evidence_path(evidence, "evidence/1-1.jpg") is not None
    assert safe_evidence_path(evidence, "1-1.jpg") is not None


def test_a_reference_to_a_missing_file_resolves_to_none(evidence):
    assert safe_evidence_path(evidence, "evidence/gone.jpg") is None


# --- manifest -----------------------------------------------------------

def test_manifest_reports_real_evidence(evidence):
    entries = evidence_manifest(_record(), evidence)
    assert len(entries) == 1
    e = entries[0]
    assert e["missing"] is False
    assert e["source"] == "evidence/1-1.jpg"
    assert e["bytes"] > 0
    assert (e["width"], e["height"]) == (1, 1)


def test_manifest_flags_missing_evidence_without_failing(evidence):
    rec = _record(frames=[{"path": "evidence/gone.jpg", "rel_ts": 0.0, "detections": []}])
    e = evidence_manifest(rec, evidence)[0]
    assert e["missing"] is True
    assert "bytes" not in e


def test_jpeg_size_reads_the_header(evidence):
    assert jpeg_size(evidence / "1-1.jpg") == (1, 1)


def test_jpeg_size_of_a_non_jpeg_is_none(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"definitely not a jpeg")
    assert jpeg_size(p) is None


# --- json ---------------------------------------------------------------

def test_json_carries_structured_evidence_not_base64(evidence):
    payload = json.loads(to_json([_record()], evidence))
    ev = payload["records"][0]["evidence"][0]
    assert set(ev) >= {"file", "source", "rel_ts", "detections", "missing"}
    assert "data" not in ev and "base64" not in ev
    assert "base64" not in json.dumps(payload).lower()


def test_json_preserves_cyrillic(evidence):
    text = to_json([_record()], evidence)
    assert "человек должен держать телефон." in text
    assert json.loads(text)["records"][0]["standard"] == "человек должен держать телефон."


# --- zip package --------------------------------------------------------

def test_package_contains_reports_and_evidence(evidence):
    with zipfile.ZipFile(_io.BytesIO(build_package([_record()], evidence))) as z:
        names = z.namelist()
        assert "report.csv" in names
        assert "report.json" in names
        assert any(n.startswith("evidence/") and n.endswith(".jpg") for n in names)
        assert z.read("evidence/" + evidence_manifest(_record(), evidence)[0]["file"]) == _TINY_JPEG


def test_package_reports_agree_with_each_other(evidence):
    recs = [_record(id="a", ts=1.0), _record(id="b", ts=2.0)]
    with zipfile.ZipFile(_io.BytesIO(build_package(recs, evidence))) as z:
        rows = list(csv.DictReader(_io.StringIO(z.read("report.csv").decode("utf-8"))))
        payload = json.loads(z.read("report.json").decode("utf-8"))
    assert len(rows) == len(payload["records"]) == 2
    packaged = {r["evidence_files"] for r in rows}
    from_json = {";".join(e["file"] for e in rec["evidence"]) for rec in payload["records"]}
    assert packaged == from_json, "CSV and JSON must name the same evidence files"


def test_package_only_includes_evidence_of_exported_records(evidence):
    (evidence / "unrelated.jpg").write_bytes(_TINY_JPEG)
    with zipfile.ZipFile(_io.BytesIO(build_package([_record()], evidence))) as z:
        assert not any("unrelated" in n for n in z.namelist())


def test_missing_evidence_is_recorded_not_fatal(evidence):
    rec = _record(id="gone", frames=[{"path": "evidence/nope.jpg", "rel_ts": 0.0, "detections": []}])
    with zipfile.ZipFile(_io.BytesIO(build_package([rec, _record()], evidence))) as z:
        names = z.namelist()
        assert "evidence/MISSING.txt" in names
        assert "nope.jpg" in z.read("evidence/MISSING.txt").decode()
        rows = list(csv.DictReader(_io.StringIO(z.read("report.csv").decode("utf-8"))))
    assert len(rows) == 2, "the inspection stays in the report"
    assert [r for r in rows if r["inspection_id"] == "gone"][0]["evidence_missing"] == "1"


def test_export_never_writes_to_the_audit_data(trail, evidence):
    trail_before = trail.read_bytes()
    files_before = {p.name: p.read_bytes() for p in evidence.iterdir()}
    build_package(read_records(trail), evidence)
    to_json(read_records(trail), evidence)
    to_csv(read_records(trail), evidence)
    assert trail.read_bytes() == trail_before
    assert {p.name: p.read_bytes() for p in evidence.iterdir()} == files_before


def test_package_is_a_valid_readable_zip(evidence):
    blob = build_package([_record()], evidence)
    with zipfile.ZipFile(_io.BytesIO(blob)) as z:
        assert z.testzip() is None, "no corrupt members"


# --- exporting a selection ----------------------------------------------
#
# The safety property: a caller names inspections, never files. An id that is not
# in the audit trail selects nothing, so these endpoints cannot be turned into an
# arbitrary file reader.

from host.audit import parse_ids, select_by_ids  # noqa: E402


@pytest.fixture
def five(trail) -> list[dict]:
    return read_records(trail)


def test_one_selected_record(five):
    got = select_by_ids(five, ["rec2"])
    assert [r["id"] for r in got] == ["rec2"]


def test_two_selected_records(five):
    got = select_by_ids(five, ["rec1", "rec3"])
    assert [r["id"] for r in got] == ["rec1", "rec3"]


def test_selection_follows_the_order_asked_for(five):
    assert [r["id"] for r in select_by_ids(five, ["rec3", "rec0"])] == ["rec3", "rec0"]


def test_no_selection_means_all_records(five):
    assert len(select_by_ids(five, None)) == 5
    assert len(select_by_ids(five, [])) == 5


def test_all_records_selected_explicitly(five):
    ids = [r["id"] for r in five]
    assert [r["id"] for r in select_by_ids(five, ids)] == ids


def test_a_nonexistent_id_selects_nothing(five):
    assert select_by_ids(five, ["does-not-exist"]) == []
    assert [r["id"] for r in select_by_ids(five, ["rec1", "nope"])] == ["rec1"]


def test_duplicate_ids_are_collapsed(five):
    assert [r["id"] for r in select_by_ids(five, ["rec1", "rec1"])] == ["rec1"]


def test_parse_ids_strips_anything_path_shaped():
    assert parse_ids("abc123, def456") == ["abc123", "def456"]
    for hostile in ("../../etc/passwd", "a/../b", "/etc/passwd"):
        assert all("/" not in i and ".." not in i for i in parse_ids(hostile)), hostile
    assert parse_ids(None) == []
    assert parse_ids("") == []


def test_selected_export_carries_only_those_rows(five, evidence):
    text = to_csv(select_by_ids(five, ["rec1", "rec3"]), evidence)
    rows = list(csv.DictReader(_io.StringIO(text)))
    assert [r["inspection_id"] for r in rows] == ["rec1", "rec3"]


def test_selected_zip_contains_only_matching_evidence(tmp_path):
    """Each record gets its own evidence file; a selected package must contain
    exactly the selected inspections' images and nothing else."""
    ev = tmp_path / "evidence"
    ev.mkdir()
    records = []
    for i in range(3):
        (ev / f"shot{i}.jpg").write_bytes(_TINY_JPEG)
        records.append(_record(id=f"rec{i}", ts=1789527560.0 + i,
                               frames=[{"path": f"evidence/shot{i}.jpg",
                                        "rel_ts": 0.0, "detections": []}]))
    chosen = select_by_ids(records, ["rec0", "rec2"])
    with zipfile.ZipFile(_io.BytesIO(build_package(chosen, ev))) as z:
        jpgs = [n for n in z.namelist() if n.endswith(".jpg")]
        assert len(jpgs) == 2
        assert any("rec0" in n for n in jpgs)
        assert any("rec2" in n for n in jpgs)
        assert not any("rec1" in n for n in jpgs), "an unselected inspection leaked in"
        rows = list(csv.DictReader(_io.StringIO(z.read("report.csv").decode("utf-8"))))
    assert [r["inspection_id"] for r in rows] == ["rec0", "rec2"]


def test_selecting_a_single_record_packages_cleanly(evidence):
    with zipfile.ZipFile(_io.BytesIO(build_package([_record(id="only")], evidence))) as z:
        assert z.testzip() is None
        assert len(list(csv.DictReader(_io.StringIO(z.read("report.csv").decode("utf-8"))))) == 1


def test_selected_export_preserves_cyrillic(five, evidence):
    rows = list(csv.DictReader(_io.StringIO(to_csv(select_by_ids(five, ["rec0"]), evidence))))
    assert rows[0]["standard"] == "человек должен держать телефон."
    payload = json.loads(to_json(select_by_ids(five, ["rec0"]), evidence))
    assert payload["records"][0]["standard"] == "человек должен держать телефон."


def test_selecting_nothing_real_produces_an_empty_but_valid_export(five, evidence):
    chosen = select_by_ids(five, ["ghost"])
    assert to_csv(chosen, evidence).strip() == ",".join(CSV_COLUMNS)
    with zipfile.ZipFile(_io.BytesIO(build_package(chosen, evidence))) as z:
        assert z.testzip() is None
        assert "report.csv" in z.namelist()


# --- raw VLM trace (local backend only) -------------------------------------
#
# The audit stores the parsed judgement, so when the model copied the prompt's
# example verbatim there was no record of what it actually emitted and the
# failure had to be reconstructed with a replay harness. The trace is opt-in and
# writes the prompt and the untouched reply next to each other.

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "edge"))


def test_vlm_trace_records_the_untouched_reply(tmp_path):
    import local_edge
    path = tmp_path / "vlm-trace.jsonl"
    local_edge.trace_vlm(str(path), model="m-4bit", n_images=6,
                         prompt="PROMPT TEXT", raw='{"verdict":"PASS"}  ',
                         metrics={"inference_ms": 12.5})
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["raw"] == '{"verdict":"PASS"}  ', "the reply must be stored byte-for-byte"
    assert rec["prompt"] == "PROMPT TEXT"
    assert rec["model"] == "m-4bit"
    assert rec["n_images"] == 6
    assert rec["metrics"]["inference_ms"] == 12.5
    assert isinstance(rec["ts"], float)


def test_vlm_trace_appends_rather_than_overwriting(tmp_path):
    import local_edge
    path = tmp_path / "vlm-trace.jsonl"
    for i in range(3):
        local_edge.trace_vlm(str(path), model="m", n_images=1,
                             prompt="p", raw=f"reply {i}", metrics={})
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["raw"] for line in lines] == ["reply 0", "reply 1", "reply 2"]


def test_vlm_trace_never_breaks_an_inspection(tmp_path):
    """A diagnostic that can fail an inspection is worse than no diagnostic."""
    import local_edge
    unwritable = tmp_path / "no-such-dir" / "deep" / "trace.jsonl"
    local_edge.trace_vlm(str(unwritable), model="m", n_images=1,
                         prompt="p", raw="r", metrics={})  # must not raise


# --- two-rule export ---------------------------------------------------------

def _two_rule_record():
    return {
        "id": "abc123", "ts": 1789560000.0, "verdict": "fail",
        "decided_by": "relationship-violated",
        "standard": "The person must be holding a phone.",
        "reason": "Rule 1: PASS. ... Rule 2: FAIL. ...",
        "backend": "local",
        "rules": [
            {"index": 1, "text": "The person must be holding a phone.",
             "verdict": "pass", "reason": "The person is holding the cell phone.",
             "decided_by": "relationship-met"},
            {"index": 2, "text": "The person must NOT be holding a bottle.",
             "verdict": "fail", "reason": "The person is holding the bottle.",
             "decided_by": "relationship-violated"},
        ],
        "evidence_paths": ["evidence/a.jpg", "evidence/b.jpg"],
        "frames": [{"path": "evidence/a.jpg"}, {"path": "evidence/b.jpg"}],
        "metrics": {}, "window": {}, "required_objects": [], "prohibited_objects": [],
    }


def test_csv_has_explicit_columns_for_both_rules():
    from host.audit import CSV_COLUMNS, to_csv
    for col in ("rule_1", "rule_1_verdict", "rule_1_reason",
                "rule_2", "rule_2_verdict", "rule_2_reason", "overall_verdict"):
        assert col in CSV_COLUMNS, col
    out = to_csv([_two_rule_record()])
    header, row = out.splitlines()[0], out.splitlines()[1]
    assert "rule_2_verdict" in header
    assert "The person must NOT be holding a bottle." in row
    assert "relationship" not in header, "attribution jargon belongs in its own column"


def test_csv_for_a_single_rule_leaves_rule_2_empty():
    from host.audit import to_csv
    rec = _two_rule_record()
    rec["rules"] = rec["rules"][:1]
    rec["verdict"] = "pass"
    out = to_csv([rec])
    cols = out.splitlines()[0].split(",")
    values = next(csv.reader(io.StringIO(out.splitlines()[1])))
    row = dict(zip(cols, values, strict=False))
    assert row["rule_1"] == "The person must be holding a phone."
    assert row["rule_2"] == ""
    assert row["rule_2_verdict"] == ""
    assert row["overall_verdict"] == "pass"


def test_csv_stays_parseable_with_two_rules():
    from host.audit import to_csv
    out = to_csv([_two_rule_record()])
    rows = list(csv.reader(io.StringIO(out)))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1]), "ragged CSV"


def test_a_record_with_no_rules_list_still_exports():
    """Records written before multi-rule existed must keep exporting."""
    from host.audit import to_csv
    rec = _two_rule_record()
    del rec["rules"]
    out = to_csv([rec])
    cols = out.splitlines()[0].split(",")
    values = next(csv.reader(io.StringIO(out.splitlines()[1])))
    row = dict(zip(cols, values, strict=False))
    assert row["rule_1"] == "The person must be holding a phone."
    assert row["overall_verdict"] == "fail"


def test_evidence_is_not_duplicated_per_rule():
    from host.audit import evidence_manifest
    manifest = evidence_manifest(_two_rule_record(), None)
    files = [e["file"] for e in manifest]
    assert len(files) == len(set(files)) == 2, "one shared evidence set, listed once"


def test_the_history_summary_carries_the_per_rule_breakdown():
    """The console expands a row to show Rule 1 / Rule 2, so summarise must
    carry them; without this the detail view would silently show nothing."""
    from host.audit import summarise
    row = summarise(_two_rule_record())
    assert len(row["rules"]) == 2
    assert row["rules"][1]["verdict"] == "fail"
    assert row["rules"][0]["text"] == "The person must be holding a phone."


def test_the_history_summary_of_an_old_record_has_no_rules():
    from host.audit import summarise
    rec = _two_rule_record()
    del rec["rules"]
    assert summarise(rec)["rules"] == []
