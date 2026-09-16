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


# --- nested duplicate suppression --------------------------------------
#
# Root cause context: on-device NMS uses IoU, which cannot see nesting. A box
# 95% inside a box ten times its size still has a low IoU because the union is
# dominated by the large box. Measured on hardware: 10 same-class nested pairs
# over 375 frames, IoU 0.05-0.10, containment 0.73-0.97.
#
# The real bottom-left "duplicate" in this venue was verified by eye to be a
# SECOND PERSON (containment 0.65, area ratio 0.013). The thresholds must spare it.

def d(label, bbox, conf=0.7):
    return {"label": label, "confidence": conf, "bbox": list(bbox)}


def test_nested_small_box_inside_a_much_larger_one_is_suppressed():
    big = d("person", (0.10, 0.10, 0.90, 0.95), 0.69)
    tiny = d("person", (0.40, 0.20, 0.50, 0.35), 0.58)   # fully inside, ~2% of area
    out = fe.suppress_nested_duplicates([big, tiny])
    assert len(out) == 1
    assert out[0]["confidence"] == 0.69, "the higher-confidence box must survive"


def test_two_genuinely_separate_people_are_both_kept():
    a = d("person", (0.05, 0.10, 0.45, 0.95), 0.69)
    b = d("person", (0.55, 0.10, 0.95, 0.95), 0.66)
    assert len(fe.suppress_nested_duplicates([a, b])) == 2


def test_the_verified_real_background_person_is_not_suppressed():
    """Regression from real hardware.

    Stored evidence frame audit/evidence/1789519179373-4.jpg contained two person
    boxes: the main subject filling the frame, and a small box at the bottom left.
    Cropping and enlarging that box showed a genuinely different person - curly
    hair, headphones - sitting behind the subject. Measured containment 0.65 at an
    area ratio of 0.011. If the containment threshold ever drops near 0.65, this
    real person gets deleted.
    """
    main = d("person", (0.125, 0.02, 0.98, 0.99), 0.69)     # containment 0.66
    background = d("person", (0.102, 0.882, 0.183, 0.999), 0.59)
    assert fe.box_containment(background["bbox"], main["bbox"]) == pytest.approx(0.66, abs=0.02)
    out = fe.suppress_nested_duplicates([main, background])
    assert len(out) == 2, "a real second person must never be merged away"


def test_partially_overlapping_people_are_both_kept():
    a = d("person", (0.10, 0.10, 0.60, 0.95), 0.69)
    b = d("person", (0.45, 0.15, 0.95, 0.95), 0.64)   # big overlap, similar size
    assert len(fe.suppress_nested_duplicates([a, b])) == 2


def test_different_classes_may_legitimately_nest():
    person = d("person", (0.10, 0.05, 0.90, 0.99), 0.69)
    phone = d("cell phone", (0.40, 0.55, 0.52, 0.70), 0.61)
    assert len(fe.suppress_nested_duplicates([person, phone])) == 2


def test_a_nested_box_just_below_the_containment_threshold_is_kept():
    big = d("person", (0.0, 0.0, 1.0, 1.0), 0.69)
    # half of the small box hangs outside the large one
    edge = d("person", (0.90, 0.40, 1.10, 0.50), 0.60)
    assert len(fe.suppress_nested_duplicates([big, edge])) == 2


def test_a_nested_box_too_large_relative_to_the_parent_is_kept():
    big = d("person", (0.0, 0.0, 1.0, 1.0), 0.69)
    half = d("person", (0.2, 0.2, 0.8, 0.8), 0.60)   # 36% of the parent
    assert len(fe.suppress_nested_duplicates([half, big])) == 2


def test_suppression_is_order_independent():
    big = d("person", (0.10, 0.10, 0.90, 0.95), 0.69)
    tiny = d("person", (0.40, 0.20, 0.50, 0.35), 0.58)
    assert len(fe.suppress_nested_duplicates([tiny, big])) == 1
    assert len(fe.suppress_nested_duplicates([big, tiny])) == 1


def test_empty_and_single_detection_lists_are_unchanged():
    assert fe.suppress_nested_duplicates([]) == []
    assert len(fe.suppress_nested_duplicates([d("person", (0, 0, 1, 1))])) == 1


def test_thresholds_are_configurable_and_the_defaults_are_what_protect_the_real_person():
    main = d("person", (0.125, 0.02, 0.98, 0.99), 0.69)
    background = d("person", (0.102, 0.882, 0.183, 0.999), 0.59)
    # Reckless settings WOULD merge the verified real person, which is exactly
    # why the defaults sit well above the 0.65 containment it exhibits.
    reckless = fe.suppress_nested_duplicates([main, background],
                                             containment_min=0.60, area_ratio_max=0.50)
    assert len(reckless) == 1
    assert len(fe.suppress_nested_duplicates([main, background])) == 2


def test_iou_and_containment_differ_on_nested_boxes():
    """The reason IoU-based NMS misses these."""
    big = (0.0, 0.0, 1.0, 1.0)
    small = (0.45, 0.45, 0.55, 0.55)
    assert fe.box_iou(big, small) < 0.02
    assert fe.box_containment(big, small) == pytest.approx(1.0)


# --- ROI attribute inspection ------------------------------------------
#
# The detector has 80 COCO classes. `cap`, `label`, `glasses`, `goggles` and
# `damage` are not among them, so the VLM must judge those - but at 1280x720
# scaled down for the vision encoder, a bottle cap is a few pixels. ROI crops the
# detector-grounded parent object and enlarges it.

def _frame(dets, ts=0.0):
    return _Rec(ts=ts, detections=dets)


def test_roi_picks_the_smallest_required_object():
    """For 'the person must hold a phone', zooming the phone is what helps."""
    frames = [_frame([d("person", (0.1, 0.0, 0.9, 1.0)),
                      d("cell phone", (0.45, 0.55, 0.55, 0.65))], ts=i * 0.2)
              for i in range(10)]
    label, area = fe.pick_roi_class(frames, ["person", "cell phone"])
    assert label == "cell phone"
    assert area < 0.02


def test_roi_picks_the_bottle_when_it_is_the_only_required_object():
    frames = [_frame([d("bottle", (0.4, 0.3, 0.5, 0.8))], ts=i * 0.2) for i in range(10)]
    assert fe.pick_roi_class(frames, ["bottle"])[0] == "bottle"


def test_roi_ignores_a_required_class_the_detector_never_saw():
    frames = [_frame([d("person", (0.1, 0.0, 0.9, 1.0))], ts=i * 0.2) for i in range(10)]
    assert fe.pick_roi_class(frames, ["person", "cell phone"])[0] == "person"


def test_roi_returns_nothing_when_no_required_object_is_present():
    frames = [_frame([], ts=i * 0.2) for i in range(5)]
    assert fe.pick_roi_class(frames, ["bottle"])[0] is None


def test_stable_bbox_is_the_median_not_a_single_frame():
    frames = [_frame([d("bottle", (0.40, 0.30, 0.50, 0.80), 0.7)], ts=0.0),
              _frame([d("bottle", (0.41, 0.31, 0.51, 0.81), 0.7)], ts=0.2),
              _frame([d("bottle", (0.90, 0.90, 0.99, 0.99), 0.7)], ts=0.4)]  # one bad frame
    box = fe.stable_bbox(frames, "bottle")
    assert box[0] == pytest.approx(0.41, abs=0.01), "an outlier frame must not move the box"


def test_stable_bbox_prefers_the_highest_confidence_box_in_each_frame():
    frames = [_frame([d("bottle", (0.1, 0.1, 0.2, 0.2), 0.55),
                      d("bottle", (0.6, 0.6, 0.7, 0.7), 0.68)], ts=0.0)]
    assert fe.stable_bbox(frames, "bottle")[0] == pytest.approx(0.6)


def test_stable_bbox_is_none_when_the_class_is_absent():
    assert fe.stable_bbox([_frame([d("person", (0, 0, 1, 1))])], "bottle") is None


def test_missing_reason_falls_back_to_the_evidence_list():
    reply = ('{"meets_requirement":true,"verdict":"PASS","per_frame":[true,true,true],'
             '"evidence":["a person is visible in every frame"]}')
    j = fe.parse_window_judgement(reply, 3)
    assert j["reason"] == "A person is visible in every frame"


def test_missing_reason_and_evidence_says_so_plainly():
    j = fe.parse_window_judgement('{"verdict":"PASS","per_frame":[true,true,true]}', 3)
    assert j["reason"] == "No reason given."
