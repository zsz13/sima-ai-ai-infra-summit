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


# --- speech language control -------------------------------------------
#
# Thresholds and the dual-decode rule were calibrated on the DevKit against
# whisper-small-a16w8; the numbers used below are measured, not invented.

def test_script_detection():
    assert fe.transcript_script("This person must be holding a phone.") == "latin"
    assert fe.transcript_script("человек должен держать телефон") == "cyrillic"
    assert fe.transcript_script("...") == "none"


def test_script_agreement_rejects_a_decode_in_the_wrong_alphabet():
    # Measured: forced-ru on English audio returned the English sentence verbatim.
    assert not fe.script_agrees("ru", "There must be no phone in view.")
    assert fe.script_agrees("ru", "человек должен держать телефон")
    assert fe.script_agrees("en", "There must be no phone in view.")
    assert not fe.script_agrees("en", "человек должен держать телефон")


def test_auto_picks_english_for_english_audio():
    # e1.wav, measured on the DevKit.
    best = fe.choose_transcript([
        {"language": "en", "text": "This person must be holding a phone.",
         "avg_logprob": -0.061},
        {"language": "ru", "text": "Это человек должен быть держит телефон.",
         "avg_logprob": -1.230},
    ])
    assert best["language"] == "en"


def test_auto_picks_russian_for_russian_audio():
    # r1.wav, measured on the DevKit.
    best = fe.choose_transcript([
        {"language": "en", "text": "A person should hold the phone.",
         "avg_logprob": -0.512},
        {"language": "ru", "text": "человек должен держать телефон.",
         "avg_logprob": -0.056},
    ])
    assert best["language"] == "ru"


def test_script_disagreement_beats_likelihood():
    """A forced-ru decode that emits Latin text is evidence against Russian.

    Measured case e3.wav: asked for Russian, Whisper returned the English
    sentence. Without the script rule a higher-likelihood wrong-language decode
    could win.
    """
    best = fe.choose_transcript([
        {"language": "en", "text": "There must be no phone in view.",
         "avg_logprob": -0.900},
        {"language": "ru", "text": "There must be no phone in view.",
         "avg_logprob": -0.238},
    ])
    assert best["language"] == "en"


def test_choose_transcript_ignores_empty_decodes():
    best = fe.choose_transcript([
        {"language": "en", "text": "", "avg_logprob": -0.01},
        {"language": "ru", "text": "человек должен держать телефон", "avg_logprob": -0.4},
    ])
    assert best["language"] == "ru"


def test_good_speech_is_accepted():
    ok, why = fe.assess_transcript("This person must be holding a phone.", 0.0025, -0.061)
    assert ok and why == ""


def test_silence_is_rejected_despite_a_confident_hallucination():
    """Whisper invents text for silence at a healthy likelihood.

    Measured: pink noise produced "СПОКОЙНАЯ МУЗЫКА" at avg_logprob -0.529, which
    a likelihood threshold alone would accept. no_speech_prob was 0.905.
    """
    ok, why = fe.assess_transcript("СПОКОЙНАЯ МУЗЫКА", 0.905, -0.529)
    assert not ok
    assert "speech" in why


def test_incoherent_decode_is_rejected():
    ok, why = fe.assess_transcript("Это человек должен быть держит телефон.", 0.004, -1.230)
    assert not ok


def test_empty_and_trivial_transcripts_are_rejected():
    assert not fe.assess_transcript("", 0.001, -0.05)[0]
    assert not fe.assess_transcript("you", 0.001, -0.05)[0]
    assert not fe.assess_transcript("...", 0.001, -0.05)[0]


def test_choose_transcript_handles_an_empty_candidate_list():
    assert fe.choose_transcript([]) is None


def test_choose_transcript_falls_back_when_every_decode_is_empty():
    cands = [{"language": "en", "text": "", "avg_logprob": -0.1},
             {"language": "ru", "text": "   ", "avg_logprob": -0.2}]
    assert fe.choose_transcript(cands) is cands[0]


def test_choose_transcript_falls_back_when_no_script_agrees():
    """Both decodes contradict their forced language: likelihood then decides
    rather than the caller getting nothing back."""
    best = fe.choose_transcript([
        {"language": "en", "text": "человек", "avg_logprob": -0.90},
        {"language": "ru", "text": "hello there", "avg_logprob": -0.20},
    ])
    assert best is not None
    assert best["language"] == "ru"


def test_assess_transcript_thresholds_are_inclusive():
    """The documented boundaries are rejections, not acceptances."""
    assert not fe.assess_transcript("a real sentence", fe.NO_SPEECH_MAX, -0.05)[0]
    assert fe.assess_transcript("a real sentence", fe.NO_SPEECH_MAX - 0.001, -0.05)[0]
    assert not fe.assess_transcript("a real sentence", 0.01, fe.LOGPROB_MIN)[0]
    assert fe.assess_transcript("a real sentence", 0.01, fe.LOGPROB_MIN + 0.001)[0]


# --- GenAI.transcribe: the mode logic, with the network stubbed out -----

class _StubGenAI(fe.GenAI):
    """Exercises transcribe()'s language logic without a DevKit.

    Only the single network call is replaced; the mode handling, early break and
    candidate selection under test are the real implementation.
    """

    def __init__(self, replies):
        super().__init__("http://stub", "vlm", "asr")
        self.replies = replies
        self.calls = []

    def transcribe_forced(self, audio, filename, language):
        self.calls.append(language)
        return dict(self.replies[language], language=language, inference_ms=240.0)


_EN = {"text": "This person must be holding a phone.", "avg_logprob": -0.061,
       "no_speech_prob": 0.0025}
_RU = {"text": "человек должен держать телефон.", "avg_logprob": -0.056,
       "no_speech_prob": 0.0046}


def test_forced_mode_makes_exactly_one_call():
    g = _StubGenAI({"en": _EN, "ru": _RU})
    out = g.transcribe(b"audio", "s.wav", "ru")
    assert g.calls == ["ru"]
    assert out["language"] == "ru"
    assert out["mode"] == "ru"
    assert out["metrics"]["asr_calls"] == 1


def test_auto_mode_decodes_both_languages():
    g = _StubGenAI({"en": dict(_EN, avg_logprob=-1.20), "ru": _RU})
    out = g.transcribe(b"audio", "s.wav", "auto")
    assert g.calls == ["en", "ru"]
    assert out["language"] == "ru", "the more likely reading must win"
    assert out["metrics"]["asr_calls"] == 2
    assert len(out["candidates"]) == 2


def test_auto_mode_stops_early_when_there_is_no_speech():
    """no_speech_prob comes from the audio, not the decode, so a second pass
    cannot change it. Spending it would double the latency for nothing."""
    silent = {"text": "you", "avg_logprob": -0.48, "no_speech_prob": 0.944}
    g = _StubGenAI({"en": silent, "ru": silent})
    out = g.transcribe(b"audio", "s.wav", "auto")
    assert g.calls == ["en"], "must not decode a second time for silent audio"
    assert out["accepted"] is False
    assert out["metrics"]["asr_calls"] == 1


def test_an_unknown_mode_falls_back_to_auto():
    g = _StubGenAI({"en": _EN, "ru": _RU})
    out = g.transcribe(b"audio", "s.wav", "bg")
    assert out["mode"] == "auto"
    assert out["language"] in ("en", "ru"), "a third language can never be returned"


def test_forced_mode_reports_the_language_the_text_is_actually_in():
    """Asked for Russian, handed English: say English.

    Labelling it "ru" would send Latin text to the Russian lexicon on the Mac,
    which matches nothing and silently removes detector grounding.
    """
    english_back = {"text": "There must be no phone in view.", "avg_logprob": -0.238,
                    "no_speech_prob": 0.003}
    g = _StubGenAI({"en": _EN, "ru": english_back})
    out = g.transcribe(b"audio", "s.wav", "ru")
    assert g.calls == ["ru"]
    assert out["language"] == "en", "the label must follow the alphabet, not the request"


def test_language_from_script():
    assert fe.language_from_script("There must be no phone.", "ru") == "en"
    assert fe.language_from_script("человек должен держать телефон", "en") == "ru"
    assert fe.language_from_script("123", "ru") == "ru"
