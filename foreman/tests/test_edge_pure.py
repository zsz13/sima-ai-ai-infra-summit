"""Tests for the edge agent's host-testable logic.

foreman_edge imports pyneat/cv2/numpy lazily, so everything here runs on the Mac.
The pipeline itself needs the DevKit and is covered by scripts/smoke-test.sh.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))

import foreman_edge as fe  # noqa: E402

# --- parse_verdict: the model's reply is untrusted text ----------------

def test_clean_json_reply():
    assert fe.parse_verdict('{"verdict":"pass","reason":"Label is legible."}') == \
        ("pass", "Label is legible.")


def test_json_wrapped_in_a_fenced_block():
    reply = '```json\n{"verdict": "fail", "reason": "The lid is open."}\n```'
    assert fe.parse_verdict(reply) == ("fail", "The lid is open.")


def test_json_with_surrounding_prose():
    reply = 'Here is my assessment:\n{"verdict":"unclear","reason":"Too blurry."}\nHope that helps.'
    assert fe.parse_verdict(reply) == ("unclear", "Too blurry.")


def test_invalid_verdict_value_falls_through_to_keywords():
    verdict, _ = fe.parse_verdict('{"verdict":"perfect","reason":"looks fine"}')
    assert verdict in {"pass", "fail", "unclear"}


def test_prose_mentioning_only_fail():
    verdict, reason = fe.parse_verdict("This item does not meet the standard and should fail.")
    assert verdict == "fail"
    assert "fail" in reason.lower()


def test_prose_mentioning_only_pass():
    assert fe.parse_verdict("The item passes the stated standard.")[0] == "pass"


def test_ambiguous_prose_is_unclear_not_a_guess():
    assert fe.parse_verdict("It might pass or it might fail.")[0] == "unclear"


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_empty_reply_is_unclear_and_says_so(reply):
    verdict, reason = fe.parse_verdict(reply)
    assert verdict == "unclear"
    assert reason, "an unclear verdict must still explain itself"


def test_reason_is_truncated_not_unbounded():
    assert len(fe.parse_verdict("fail " + "x" * 5000)[1]) <= 240


# --- parse_boxes: binary tensor payload from the MLA -------------------

def pack(boxes):
    out = struct.pack("<I", len(boxes))
    for x, y, w, h, score, cid in boxes:
        out += struct.pack("<iiiifi", x, y, w, h, score, cid)
    return out


def test_parses_a_single_box():
    got = fe.parse_boxes(pack([(10, 20, 30, 40, 0.9, 3)]), 640, 480)
    assert got == [{"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": pytest.approx(0.9),
                    "class_id": 3}]


def test_parses_multiple_boxes():
    assert len(fe.parse_boxes(pack([(0, 0, 5, 5, 0.5, 1)] * 4), 640, 480)) == 4


def test_empty_payload_returns_no_boxes():
    assert fe.parse_boxes(pack([]), 640, 480) == []


def test_truncated_payload_returns_no_boxes():
    assert fe.parse_boxes(b"\x01", 640, 480) == []


def test_boxes_are_clamped_to_the_frame():
    got = fe.parse_boxes(pack([(-50, -50, 5000, 5000, 0.8, 0)]), 640, 480)[0]
    assert (got["x1"], got["y1"]) == (0.0, 0.0)
    assert (got["x2"], got["y2"]) == (640.0, 480.0)


def test_lying_header_is_rejected_rather_than_read_out_of_bounds():
    payload = struct.pack("<I", 99) + struct.pack("<iiiifi", 1, 1, 1, 1, 0.5, 0)
    with pytest.raises(RuntimeError, match="exceeds payload count"):
        fe.parse_boxes(payload, 640, 480)


# --- multipart round trip ---------------------------------------------

def test_multipart_encodes_and_extracts_the_same_bytes():
    audio = b"RIFF\x00\x00\x00\x00WAVEfmt " + bytes(range(256))
    body, content_type = fe.encode_multipart(
        {"model": "asr", "language": "auto"}, "file", "speech.wav", audio)
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body and b"asr" in body
    extracted, filename = fe.extract_uploaded_file(body, content_type)
    assert extracted == audio
    assert filename == "speech.wav"


def test_multipart_preserves_a_webm_filename():
    body, ct = fe.encode_multipart({}, "file", "speech.webm", b"\x1a\x45\xdf\xa3")
    assert fe.extract_uploaded_file(body, ct)[1] == "speech.webm"


def test_raw_body_without_boundary_is_treated_as_the_file():
    assert fe.extract_uploaded_file(b"rawaudio", "application/octet-stream") == \
        (b"rawaudio", "speech.wav")


def test_multipart_with_no_file_part_yields_nothing():
    body, ct = fe.encode_multipart({"model": "asr"}, "file", "x.wav", b"")
    # strip the file part entirely
    stripped = body.split(b'name="file"')[0] + b"--\r\n"
    assert fe.extract_uploaded_file(stripped, ct)[0] == b""


# --- argument handling -------------------------------------------------

def test_requires_source_and_model():
    with pytest.raises(SystemExit):
        fe.parse_args([])


def test_accepts_minimal_arguments():
    args = fe.parse_args(["--source", "rtsp://h/src1", "--model", "/m.tar.gz"])
    assert args.source == "rtsp://h/src1"
    assert args.port == 8100
    assert args.codec == "h264"
    assert args.score_threshold == pytest.approx(0.52)


def test_labels_file_missing_does_not_crash(tmp_path, capsys):
    assert fe.load_labels(str(tmp_path / "nope.txt")) == []
    assert "could not read labels" in capsys.readouterr().err


def test_labels_file_is_read_and_stripped(tmp_path):
    f = tmp_path / "coco.txt"
    f.write_text("person\n bicycle \n\ncar\n")
    assert fe.load_labels(str(f)) == ["person", "bicycle", "car"]


# --- self-consistency guard -------------------------------------------

def test_agreeing_boolean_and_verdict_is_trusted():
    reply = '{"meets_requirement": false, "verdict": "fail", "reason": "No hard hat."}'
    assert fe.parse_verdict(reply) == ("fail", "No hard hat.")


def test_agreeing_pass_is_trusted():
    reply = '{"meets_requirement": true, "verdict": "pass", "reason": "Lanyard is visible."}'
    assert fe.parse_verdict(reply) == ("pass", "Lanyard is visible.")


def test_contradiction_is_downgraded_to_unclear():
    # The observed real failure: reason and boolean say "not wearing", verdict says pass.
    reply = ('{"meets_requirement": false, "verdict": "pass", '
             '"reason": "The person is not wearing a hard hat."}')
    verdict, reason = fe.parse_verdict(reply)
    assert verdict == "unclear"
    assert "contradicted itself" in reason
    assert "not wearing a hard hat" in reason


def test_contradiction_the_other_way_is_also_caught():
    reply = '{"meets_requirement": true, "verdict": "fail", "reason": "It is fine."}'
    assert fe.parse_verdict(reply)[0] == "unclear"


def test_boolean_alone_is_used_when_verdict_string_is_missing():
    assert fe.parse_verdict('{"meets_requirement": false, "reason": "Absent."}') == \
        ("fail", "Absent.")


def test_verdict_alone_still_works_without_the_boolean():
    assert fe.parse_verdict('{"verdict": "pass", "reason": "Fine."}') == ("pass", "Fine.")


def test_fenced_dual_field_reply():
    reply = ('```json\n{"meets_requirement": false, "verdict": "fail", '
             '"reason": "There is no dog."}\n```')
    assert fe.parse_verdict(reply) == ("fail", "There is no dog.")


# --- multi-frame window judgement ------------------------------------

def test_window_judgement_parses_a_clean_reply():
    reply = ('{"meets_requirement":false,"verdict":"FAIL",'
             '"per_frame":[false,false,null],'
             '"evidence":["hands are empty"],"missing_evidence":["any phone"],'
             '"reason":"No phone is visible."}')
    j = fe.parse_window_judgement(reply, 3)
    assert j["verdict"] == "fail"
    assert j["per_frame"] == [False, False, None]
    assert j["evidence"] == ["hands are empty"]
    assert j["missing_evidence"] == ["any phone"]


def test_window_judgement_contradiction_becomes_unclear():
    """Observed on hardware: verdict PASS with a reason saying the opposite."""
    reply = ('{"meets_requirement":false,"verdict":"PASS","per_frame":[null,null,null],'
             '"reason":"The person is not holding a smartphone in any of the frames."}')
    j = fe.parse_window_judgement(reply, 3)
    assert j["verdict"] == "unclear"
    assert "contradicted itself" in j["reason"]


def test_window_judgement_accepts_per_frame_dicts_too():
    reply = ('{"verdict":"PASS","per_frame":[{"index":1,"supports":true},'
             '{"index":2,"supports":false}],"reason":"Mixed."}')
    assert fe.parse_window_judgement(reply, 3)["per_frame"] == [True, False, None]


def test_window_judgement_pads_per_frame_to_the_number_sent():
    j = fe.parse_window_judgement('{"verdict":"PASS","per_frame":[true],"reason":"x"}', 3)
    assert len(j["per_frame"]) == 3


def test_window_judgement_truncates_extra_per_frame_entries():
    j = fe.parse_window_judgement(
        '{"verdict":"PASS","per_frame":[true,true,true,true,true],"reason":"x"}', 3)
    assert len(j["per_frame"]) == 3


def test_window_judgement_unparseable_is_unclear_not_a_guess():
    j = fe.parse_window_judgement("I believe the phone is there somewhere", 3)
    assert j["verdict"] == "unclear"
    assert "did not return usable JSON" in j["reason"]
    assert j["per_frame"] == [None, None, None]


def test_window_judgement_handles_a_fenced_reply():
    reply = '```json\n{"meets_requirement":true,"verdict":"PASS","per_frame":[true,true,true],"reason":"Held."}\n```'
    assert fe.parse_window_judgement(reply, 3)["verdict"] == "pass"


def test_window_judgement_caps_list_lengths():
    reply = ('{"verdict":"PASS","per_frame":[true,true,true],'
             '"evidence":["a","b","c","d","e","f","g","h"],"reason":"x"}')
    assert len(fe.parse_window_judgement(reply, 3)["evidence"]) <= 6


def test_schema_placeholder_reason_is_rejected():
    """Observed on hardware: the model echoed the prompt's own placeholder."""
    reply = '{"meets_requirement":true,"verdict":"PASS","per_frame":[true,true,true],"reason":"<one short sentence>"}'
    j = fe.parse_window_judgement(reply, 3)
    assert j["verdict"] == "pass"
    assert j["reason"] == "No reason given."


def test_placeholder_evidence_items_are_dropped():
    reply = ('{"verdict":"PASS","per_frame":[true,true,true],'
             '"evidence":["<short visible fact>","a real observation"],"reason":"ok"}')
    assert fe.parse_window_judgement(reply, 3)["evidence"] == ["a real observation"]


def test_duplicate_evidence_items_are_collapsed():
    reply = ('{"verdict":"PASS","per_frame":[true,true,true],'
             '"evidence":["person visible","person visible","person visible"],"reason":"ok"}')
    assert fe.parse_window_judgement(reply, 3)["evidence"] == ["person visible"]


# --- representative frame selection -----------------------------------

class _Rec:
    def __init__(self, ts, sharpness=100.0, detections=None, jpeg=b"x", frame_id=0):
        self.ts, self.sharpness = ts, sharpness
        self.detections = detections or []
        self.jpeg, self.frame_id = jpeg, frame_id


def _window(n=45, fps=15.0, sharp=lambda i: 100.0, dets=lambda i: None):
    return [_Rec(ts=i / fps, sharpness=sharp(i), detections=dets(i), frame_id=i)
            for i in range(n) if i % 3 == 0]


def test_selection_returns_the_requested_number():
    assert len(fe.select_representative(_window(), [], 3)) == 3


def test_selected_frames_are_temporally_separated():
    """Regression: +0.9 s and +1.0 s were chosen together from adjacent buckets."""
    chosen = fe.select_representative(_window(), [], 3)
    times = [c.ts for c in chosen]
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert all(g >= 0.5 for g in gaps), f"frames too close together: {times}"


def test_selected_frames_are_returned_in_time_order():
    chosen = fe.select_representative(_window(), [], 3)
    assert [c.ts for c in chosen] == sorted(c.ts for c in chosen)


def test_selection_prefers_frames_showing_the_required_object():
    def dets(i):
        return [{"label": "cell phone", "confidence": 0.66}] if i >= 30 else []
    chosen = fe.select_representative(_window(dets=dets), ["cell phone"], 2)
    assert any(c.detections for c in chosen), "a frame with the object should be chosen"


def test_selection_prefers_sharper_frames_when_all_else_is_equal():
    frames = _window(sharp=lambda i: float(i))
    sharpest = max(f.sharpness for f in frames)
    chosen = fe.select_representative(frames, [], 1)
    assert chosen[0].sharpness == sharpest


def test_selection_skips_frames_with_no_jpeg():
    frames = [_Rec(ts=i / 15.0, jpeg=None) for i in range(45)]
    assert fe.select_representative(frames, [], 3) == []


def test_selection_relaxes_the_gap_rather_than_returning_too_few():
    frames = [_Rec(ts=0.0), _Rec(ts=0.05), _Rec(ts=0.1)]
    assert len(fe.select_representative(frames, [], 3)) == 3


def test_selection_handles_a_window_smaller_than_the_request():
    assert len(fe.select_representative([_Rec(ts=0.0)], [], 3)) == 1


def test_selection_of_zero_frames_is_empty():
    assert fe.select_representative(_window(), [], 0) == []
