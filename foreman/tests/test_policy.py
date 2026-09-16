"""Regression tests for temporal grounding.

Case 1-7 below are the scenarios that motivated this work. Case 1 and case 7 are
the ones that matter most: they are the hallucination that was observed on real
hardware, where the VLM asserted a smartphone that the detector never saw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host.policy import (  # noqa: E402
    FAIL,
    PASS,
    UNCLEAR,
    Decision,
    GroundingConfig,
    ObjectEvidence,
    VlmJudgement,
    build_evidence,
    decide,
)
from host.standard_parser import parse_standard  # noqa: E402

WINDOW = 45  # ~3 s at 15 fps


def ev(label, present, total=WINDOW, conf=0.65):
    return ObjectEvidence(label, present, total, tuple([conf] * present))


def vlm(verdict, reason="", per_frame=(True, True, True), **kw):
    return VlmJudgement(verdict=verdict, reason=reason, per_frame=per_frame, **kw)


def run(standard, evidence, judgement, config=None) -> Decision:
    return decide(parse_standard(standard), evidence, judgement, config)


# =====================================================================
# Case 1 - phone absent across the whole window must NEVER pass
# =====================================================================

def test_case1_absent_phone_never_passes_even_when_vlm_says_pass():
    """The exact observed failure: VLM hallucinates the phone, detector saw none."""
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 0)},
            vlm(PASS, "The person is holding a smartphone, as indicated by the "
                      "visible screen and the way the person is holding the device."))
    assert d.verdict == FAIL
    assert d.decided_by == "detector-absent"
    assert "cell phone" in d.reason
    assert "never detected" in d.reason


def test_case1_notes_record_that_the_vlm_was_overridden():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 0)},
            vlm(PASS, "It is clearly in their hand."))
    assert any("Detector evidence wins" in n for n in d.notes)


@pytest.mark.parametrize("vlm_verdict", [PASS, FAIL, UNCLEAR])
def test_case1_absent_required_object_fails_whatever_the_vlm_says(vlm_verdict):
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 0)},
            vlm(vlm_verdict, "whatever"))
    assert d.verdict == FAIL


# =====================================================================
# Case 2 - phone reliably present and visibly held -> PASS
# =====================================================================

def test_case2_reliable_phone_and_supporting_vlm_passes():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 42, conf=0.66)},
            vlm(PASS, "The phone is in the person's right hand in every frame."))
    assert d.verdict == PASS
    assert d.decided_by == "vlm"
    assert "right hand" in d.reason


def test_case2_reports_the_presence_ratio_it_relied_on():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 42)},
            vlm(PASS, "Held."))
    phone = next(o for o in d.required if o["label"] == "cell phone")
    assert phone["frames_present"] == 42
    assert phone["frames_total"] == 45
    assert phone["presence_ratio"] == pytest.approx(42 / 45, abs=1e-3)


# =====================================================================
# Case 3 - phone present but lying on a table -> FAIL on the relationship
# =====================================================================

def test_case3_phone_present_but_not_held_fails_via_the_vlm():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 44)},
            vlm(FAIL, "The phone is lying flat on the table, not in anyone's hand.",
                per_frame=(False, False, False)))
    assert d.verdict == FAIL
    assert d.decided_by == "vlm"
    assert "table" in d.reason


def test_case3_detector_grounding_is_recorded_as_satisfied():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 44)},
            vlm(FAIL, "On the table.", per_frame=(False, False, False)))
    assert any("Detector confirmed every required object" in n for n in d.notes)


# =====================================================================
# Case 4 - one ambiguous frame -> not a confident PASS
# =====================================================================

def test_case4_single_ambiguous_sighting_does_not_pass():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 1)},
            vlm(PASS, "I think I can see a phone."))
    assert d.verdict != PASS
    assert d.verdict == FAIL          # 1/45 = 2% <= absent_ratio_max
    assert d.decided_by == "detector-absent"
    assert "only 1 of 45 frames" in d.reason


def test_case4_a_few_more_sightings_become_unclear_not_pass():
    # 9/45 = 20%: above "absent", below "reliably present"
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 9)},
            vlm(PASS, "Looks held."))
    assert d.verdict == UNCLEAR
    assert d.decided_by == "intermittent-required"
    assert "inconsistently" in d.reason


# =====================================================================
# Case 5 - temporary occlusion, clearly present otherwise
# =====================================================================

def test_case5_occluded_for_part_of_the_window_still_decides():
    # present in 34/45 = 76%: above present_ratio_min, so the VLM decides
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 34)},
            vlm(PASS, "The phone is held throughout; a hand briefly covers it."))
    assert d.verdict == PASS
    assert d.decided_by == "vlm"


def test_case5_threshold_boundary_is_inclusive_at_present_ratio_min():
    exactly = int(round(0.60 * WINDOW))   # 27/45 = 60%
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", exactly)},
            vlm(PASS, "Held."))
    assert d.verdict == PASS


def test_case5_just_below_the_boundary_is_unclear():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 26)},
            vlm(PASS, "Held."))
    assert d.verdict == UNCLEAR


# =====================================================================
# Case 6 - the VLM contradicts itself across frames
# =====================================================================

def test_case6_contradictory_per_frame_evidence_is_unclear():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 44)},
            vlm(PASS, "Mostly held.", per_frame=(True, False, True)))
    assert d.verdict == UNCLEAR
    assert d.decided_by == "vlm-contradictory"
    assert "2 support, 1 contradict" in d.reason


def test_case6_unanimous_frames_are_not_contradictory():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 44)},
            vlm(PASS, "Held in all three.", per_frame=(True, True, True)))
    assert d.verdict == PASS


def test_case6_unknown_frames_alone_do_not_count_as_contradiction():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 44)},
            vlm(PASS, "Two clear, one obscured.", per_frame=(True, None, True)))
    assert d.verdict == PASS


# =====================================================================
# Case 7 - the VLM may not override detector grounding
# =====================================================================

def test_case7_vlm_cannot_invent_an_object_the_detector_never_saw():
    d = run("There must be a dog in the picture",
            {"dog": ev("dog", 0)},
            vlm(PASS, "A dog is clearly visible on the left.",
                evidence=("a dog on the left",)))
    assert d.verdict == FAIL
    assert d.decided_by == "detector-absent"


def test_case7_prohibited_object_reliably_seen_fails_despite_vlm_pass():
    d = run("The person must not be holding a phone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 40)},
            vlm(PASS, "No phone anywhere."))
    assert d.verdict == FAIL
    assert d.decided_by == "detector-prohibited"
    assert "must not be present" in d.reason


def test_prohibited_object_absent_defers_to_the_vlm():
    d = run("The person must not be holding a phone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 0)},
            vlm(PASS, "Hands are empty."))
    assert d.verdict == PASS


def test_prohibited_object_intermittent_is_unclear():
    d = run("The person must not be holding a phone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 15)},
            vlm(PASS, "Hands look empty."))
    assert d.verdict == UNCLEAR
    assert d.decided_by == "intermittent-prohibited"


# =====================================================================
# Window sufficiency and ungrounded standards
# =====================================================================

def test_short_window_is_unclear_regardless_of_the_vlm():
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 5, total=5), "cell phone": ev("cell phone", 5, total=5)},
            vlm(PASS, "Held."))
    assert d.verdict == UNCLEAR
    assert d.decided_by == "insufficient-window"
    assert "5 frames" in d.reason


def test_standard_with_no_detector_objects_defers_to_the_vlm_and_says_so():
    d = run("every box must have a label facing up and the lid closed",
            {}, vlm(PASS, "Label up, lid shut."))
    assert d.verdict == PASS
    assert d.decided_by == "vlm"
    assert any("No detector-supported objects" in n for n in d.notes)


def test_ungrounded_standard_still_reports_a_vlm_fail():
    d = run("the lid must be closed", {}, vlm(FAIL, "The lid is open."))
    assert d.verdict == FAIL


# =====================================================================
# build_evidence: per-frame detections -> temporal evidence
# =====================================================================

def det(label, conf):
    return {"label": label, "confidence": conf, "bbox": [0, 0, 1, 1]}


def test_build_evidence_counts_frames_not_detections():
    frames = [[det("person", 0.7), det("person", 0.6)], [det("person", 0.7)], []]
    e = build_evidence(frames, ["person"])["person"]
    assert e.frames_present == 2
    assert e.frames_total == 3


def test_build_evidence_takes_the_best_confidence_in_each_frame():
    frames = [[det("person", 0.58), det("person", 0.69)]]
    assert build_evidence(frames, ["person"])["person"].conf_max == pytest.approx(0.69)


def test_build_evidence_ignores_detections_below_min_confidence():
    frames = [[det("cell phone", 0.51)], [det("cell phone", 0.52)]]
    e = build_evidence(frames, ["cell phone"], GroundingConfig(min_confidence=0.55))["cell phone"]
    assert e.frames_present == 0
    assert e.presence_ratio == 0.0


def test_build_evidence_reports_zero_for_a_class_never_seen():
    e = build_evidence([[det("person", 0.7)]] * 10, ["cell phone"])["cell phone"]
    assert e.frames_present == 0
    assert e.conf_median == 0.0


def test_build_evidence_median_confidence():
    frames = [[det("person", c)] for c in (0.60, 0.70, 0.80)]
    assert build_evidence(frames, ["person"])["person"].conf_median == pytest.approx(0.70)


# =====================================================================
# Config validation
# =====================================================================

@pytest.mark.parametrize("kwargs", [
    {"absent_ratio_max": 0.7, "present_ratio_min": 0.6},   # inverted
    {"absent_ratio_max": -0.1},
    {"present_ratio_min": 1.5},
    {"min_confidence": 2.0},
    {"min_window_frames": 0},
])
def test_invalid_grounding_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GroundingConfig(**kwargs)


def test_thresholds_are_configurable():
    lenient = GroundingConfig(absent_ratio_max=0.01, present_ratio_min=0.05)
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 45), "cell phone": ev("cell phone", 5)},
            vlm(PASS, "Held."), lenient)
    assert d.verdict == PASS


# =====================================================================
# Window readiness: the first item after connecting must not be refused
# =====================================================================

def test_window_readiness_guard_is_documented_by_the_short_window_rule():
    """The policy refuses a short window; the orchestrator's job is to not ask.

    Regression note: before the readiness guard, the gate could settle ~0.4 s
    after connecting, so the very first inspection was always judged on a
    part-filled window and came back 'insufficient-window'. The orchestrator now
    re-arms the gate instead of consuming the item.
    """
    d = run("The person must be holding a smartphone",
            {"person": ev("person", 9, total=9), "cell phone": ev("cell phone", 9, total=9)},
            vlm(PASS, "Held."))
    assert d.verdict == UNCLEAR
    assert d.decided_by == "insufficient-window"
