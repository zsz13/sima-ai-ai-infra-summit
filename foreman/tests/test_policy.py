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
    assert d.decided_by == "relationship-met"
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
    assert d.decided_by == "relationship-violated"
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
    assert d.decided_by == "relationship-met"


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
    """An ABSENCE standard: the object itself is banned, so seeing it is a FAIL.

    Phrased as "no phone in view" rather than "must not be holding a phone".
    Those are different rules - the second one allows a phone on the table - and
    conflating them was the negation bug.
    """
    d = run("There must be no phone in view.",
            {"cell phone": ev("cell phone", 40)},
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
    d = run("There must be no phone in view.",
            {"cell phone": ev("cell phone", 15)},
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


def test_policy_names_objects_the_detector_cannot_check():
    """A partly-grounded rule must say which part went unchecked."""
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must be holding a pen.")
    evidence = {"person": ObjectEvidence("person", 45, 45, tuple([0.9] * 45))}
    d = decide(parsed, evidence, VlmJudgement(verdict="pass", reason="A pen is in the hand."))
    joined = " ".join(d.notes)
    assert "no class for pen" in joined
    assert "carries no detector evidence" in joined
    assert d.verdict == "pass", "the model still decides the part the detector cannot"


# --- the model may not overturn a measurement ---------------------------
#
# This happened for real: a phone detected in 42 of 45 frames at 85% median
# confidence, and the model replied "No phone is visible in any frame." Presence
# is the detector's question; the model was asked about the relationship.

def _phone_case(reason: str, verdict: str = "fail", standard: str | None = None):
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard(standard or "The person must be holding a phone.")
    evidence = {
        "person": ObjectEvidence("person", 45, 45, tuple([0.92] * 45)),
        "cell phone": ObjectEvidence("cell phone", 42, 45, tuple([0.85] * 42)),
    }
    return decide(parsed, evidence, VlmJudgement(verdict=verdict, reason=reason))


def test_A_reason_stays_scoped_to_the_standard():
    """"A person must be visible." must not be answered with a claim about a phone."""
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("A person must be visible.")
    evidence = {"person": ObjectEvidence("person", 45, 45, tuple([0.93] * 45))}
    d = decide(parsed, evidence, VlmJudgement(verdict="fail",
                                              reason="No phone is visible."))
    assert "phone" not in d.reason.lower(), (
        "a reason about a phone reached the operator for a standard about a person")


def test_B_denying_a_detector_confirmed_object_never_reaches_the_user():
    d = _phone_case("No phone is visible in any frame.")
    assert d.verdict == "unclear"
    assert d.decided_by == "vlm-contradicts-detector"
    assert "no phone is visible" not in d.reason.lower()
    assert "confirmed by the detector" in d.reason
    assert any("discarded" in n for n in d.notes)


def test_C_a_real_relationship_failure_is_still_a_fail():
    """The phone being present but not held is a legitimate FAIL, not a
    contradiction - the guard must not swallow it."""
    d = _phone_case("The phone is visible but lying on the table.")
    assert d.verdict == "fail"
    assert d.decided_by == "relationship-violated"
    # The headline states the rule decision; the model's own detail is kept.
    assert "lying on the table" in d.reason


def test_D_a_clear_holding_judgement_passes():
    d = _phone_case("The person is clearly holding the phone.", verdict="pass")
    assert d.verdict == "pass"
    assert d.decided_by == "relationship-met"


def test_E_an_ambiguous_relationship_is_unclear():
    d = _phone_case("It cannot be determined whether the phone is held.",
                    verdict="unclear")
    assert d.verdict == "unclear"
    assert d.decided_by == "relationship-unclear"


@pytest.mark.parametrize("reason", [
    "No phone is visible in any frame.",
    "The phone is not visible.",
    "There is no phone in the frame.",
    "No cell phone is visible.",
    "the cell phone is absent",
])
def test_absence_claims_are_caught_however_they_are_phrased(reason):
    assert _phone_case(reason).decided_by == "vlm-contradicts-detector"


def test_russian_absence_claims_are_caught_too():
    d = _phone_case("Телефон отсутствует в кадре.",
                    standard="Человек должен держать телефон.")
    assert d.decided_by == "vlm-contradicts-detector"


def test_the_guard_only_fires_for_reliably_present_objects():
    """An intermittently seen object is not "confirmed", so denying it is not a
    contradiction - and the intermittent rule handles it first anyway."""
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must be holding a phone.")
    evidence = {
        "person": ObjectEvidence("person", 45, 45, tuple([0.92] * 45)),
        "cell phone": ObjectEvidence("cell phone", 9, 45, tuple([0.6] * 9)),
    }
    d = decide(parsed, evidence, VlmJudgement(verdict="fail",
                                              reason="No phone is visible."))
    assert d.decided_by == "intermittent-required"


def test_an_honest_reason_about_another_object_is_not_a_contradiction():
    """The guard is scoped per object: a sentence about a bottle must not trip
    the phone check."""
    d = _phone_case("The person is holding the phone next to an empty desk.",
                    verdict="pass")
    assert d.verdict == "pass"


# --- temporal coverage --------------------------------------------------
#
# A frame ratio cannot tell a small object the detector flickers on from one
# that was genuinely present for only part of the window. This is the real case
# that motivated it: a phone in hand, detected in 25 of 45 frames at 71% median
# confidence, visible in all six evidence images - and ruled UNCLEAR.

def _window(per_bucket, n=45, buckets=6, conf=0.71):
    """Frames where the phone appears `per_bucket[i]` times in segment i."""
    marks = set()
    for b in range(buckets):
        lo, hi = b * n // buckets, (b + 1) * n // buckets
        for k, i in enumerate(range(lo, hi)):
            if k < per_bucket[b]:
                marks.add(i)
    frames = []
    for i in range(n):
        dets = [{"label": "person", "confidence": 0.93, "bbox": [0, 0, 1, 1]}]
        if i in marks:
            dets.append({"label": "cell phone", "confidence": conf, "bbox": [.4, .4, .5, .5]})
        frames.append(dets)
    return frames


def _judge(per_bucket, vlm_verdict="pass", reason="The person is holding the phone."):
    from host.policy import VlmJudgement, build_evidence, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must be holding a phone.")
    ev = build_evidence(_window(per_bucket), list(parsed.required))
    return decide(parsed, ev, VlmJudgement(verdict=vlm_verdict, reason=reason)), ev["cell phone"]


def test_flicker_across_the_whole_window_is_treated_as_present():
    """The reported case: 25/45 frames but every segment covered."""
    d, ev = _judge([4, 4, 4, 4, 4, 5])
    assert ev.frames_present == 25 and ev.frames_total == 45
    assert ev.presence_ratio < 0.60, "this is below the frame-level threshold"
    assert ev.spans_window, "every segment should be supported"
    assert d.verdict == "pass"
    assert d.decided_by == "relationship-met", \
        "grounding must hold so the model judges the relationship"
    assert any("detector flicker" in n for n in d.notes)


def test_detections_only_at_the_start_stay_intermittent():
    d, ev = _judge([7, 7, 7, 0, 0, 0])
    assert ev.buckets_supported == 3
    assert d.verdict == "unclear"
    assert d.decided_by == "intermittent-required"
    assert "3 of 6 time segments" in d.reason


def test_detections_only_at_the_end_stay_intermittent():
    d, ev = _judge([0, 0, 0, 7, 7, 7])
    assert ev.buckets_supported == 3
    assert d.decided_by == "intermittent-required"


def test_detections_only_in_the_middle_stay_intermittent():
    d, ev = _judge([0, 0, 7, 7, 0, 0])
    assert ev.buckets_supported == 2
    assert d.decided_by == "intermittent-required"


def test_one_isolated_detection_is_not_reliably_present():
    d, ev = _judge([0, 0, 1, 0, 0, 0])
    assert ev.buckets_supported == 1
    assert not ev.spans_window
    assert d.verdict == "fail"
    assert d.decided_by == "detector-absent"


def test_zero_detections_is_absent():
    d, ev = _judge([0, 0, 0, 0, 0, 0])
    assert ev.frames_present == 0
    assert d.verdict == "fail"
    assert d.decided_by == "detector-absent"


def test_continuous_detections_are_present():
    d, ev = _judge([7, 7, 7, 7, 7, 8])
    assert ev.presence_ratio >= 0.60
    assert d.verdict == "pass"
    assert d.decided_by == "relationship-met"


def test_one_missing_segment_is_enough_to_stay_intermittent():
    """Every segment, not most: the rule is deliberately strict."""
    d, ev = _judge([5, 5, 0, 5, 5, 5])
    assert ev.buckets_supported == 5
    assert not ev.spans_window
    assert d.decided_by == "intermittent-required"


def test_coverage_is_computed_over_every_frame_not_the_selected_images():
    from host.policy import build_evidence
    ev = build_evidence(_window([4, 4, 4, 4, 4, 5]), ["cell phone"])["cell phone"]
    assert ev.frames_total == 45, "coverage must be measured over the whole window"
    assert ev.bucket_count == 6
    assert ev.longest_gap_frames > 0, "the flicker gaps should be reported"


def test_a_prohibited_object_spanning_the_window_fails_despite_flicker():
    """Presence is presence: a banned object seen in every segment is present."""
    from host.policy import VlmJudgement, build_evidence, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("There must be no phone in view.")
    ev = build_evidence(_window([4, 4, 4, 4, 4, 5]), list(parsed.prohibited))
    d = decide(parsed, ev, VlmJudgement(verdict="pass", reason="No phone."))
    assert d.verdict == "fail"
    assert d.decided_by == "detector-prohibited"
    assert "every one of the 6 time segments" in d.reason


# --- the guard must not fire on the field that exists to name absences -------
#
# Reproduced from audit-local record 9ec96a60492a: the model judged correctly
# ("The phone is visible in the person's right hand in all frames") and filled
# "missing_evidence" with the thing it would have needed to see to fail -
# "no phone in the hand". Scanning that field for absence claims turns a correct
# PASS into UNCLEAR, because naming an absence is the field's whole purpose.

def test_missing_evidence_wording_is_not_a_presence_contradiction():
    from host.policy import ObjectEvidence, VlmJudgement, find_presence_contradiction
    vlm = VlmJudgement(
        verdict="pass",
        reason="The phone is visible in the person's right hand in all frames.",
        evidence=("phone is in the right hand of the person in all frames",),
        missing_evidence=("no phone in the hand",),
    )
    confirmed = [ObjectEvidence("cell phone", 43, 45, tuple([0.67] * 43))]
    assert find_presence_contradiction(vlm, confirmed) is None, (
        "the model was right and said so; only missing_evidence named an absence")


def test_a_correct_pass_survives_its_own_missing_evidence_field():
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must be holding a phone.")
    evidence = {
        "person": ObjectEvidence("person", 45, 45, tuple([0.93] * 45)),
        "cell phone": ObjectEvidence("cell phone", 43, 45, tuple([0.67] * 43)),
    }
    d = decide(parsed, evidence, VlmJudgement(
        verdict="pass",
        reason="The phone is visible in the person's right hand in all frames.",
        evidence=("phone is in the right hand of the person in all frames",),
        missing_evidence=("no phone in the hand",)))
    assert d.verdict == "pass", f"correct PASS was overturned: {d.decided_by}"


def test_a_real_denial_in_the_reason_is_still_caught():
    """The guard keeps its job: an absence claim in the reason still wins nothing."""
    from host.policy import ObjectEvidence, VlmJudgement, find_presence_contradiction
    vlm = VlmJudgement(verdict="fail", reason="No phone is visible in any frame.")
    confirmed = [ObjectEvidence("cell phone", 42, 45, tuple([0.85] * 42))]
    assert find_presence_contradiction(vlm, confirmed) == "cell phone"


def test_a_real_denial_in_the_evidence_list_is_still_caught():
    from host.policy import ObjectEvidence, VlmJudgement, find_presence_contradiction
    vlm = VlmJudgement(verdict="fail", reason="The requirement is not met.",
                       evidence=("there is no phone in the frame",))
    confirmed = [ObjectEvidence("cell phone", 42, 45, tuple([0.85] * 42))]
    assert find_presence_contradiction(vlm, confirmed) == "cell phone"


# --- relationship polarity: the full truth table -----------------------------
#
# The reported bug: "the person must NOT be holding a phone" was reduced to
# "the phone is prohibited", so a phone on the table failed the rule and a phone
# that was not there at all could not pass it.

def _rel_case(standard: str, *, person: int = 45, obj: int = 45,
              observed: str | None = None, obj_label: str = "cell phone",
              vlm_verdict: str = "unclear"):
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard(standard)
    evidence = {}
    for label, n in (("person", person), (obj_label, obj)):
        evidence[label] = ObjectEvidence(label, n, 45, tuple([0.88] * n),
                                         bucket_support=tuple([n > 0] * 6))
    return decide(parsed, evidence,
                  VlmJudgement(verdict=vlm_verdict, reason="",
                               observed_relationship=observed,
                               per_frame=tuple([None] * 6)))


# 1-2: positive form
def test_1_must_hold_and_is_holding_passes():
    d = _rel_case("The person must be holding a phone.", observed="holding")
    assert d.verdict == "pass", d.decided_by


def test_2_must_hold_but_is_not_holding_fails():
    d = _rel_case("The person must be holding a phone.", observed="not_holding")
    assert d.verdict == "fail", d.decided_by
    assert "holding" in d.reason.lower()


# 3-5: negative form
def test_3_must_not_hold_but_is_holding_fails():
    d = _rel_case("The person must NOT be holding a phone.", observed="holding")
    assert d.verdict == "fail", d.decided_by
    assert "holding" in d.reason.lower()


def test_4_must_not_hold_and_is_not_holding_passes():
    """The phone is in shot and that is allowed - only holding it is not."""
    d = _rel_case("The person must NOT be holding a phone.", observed="not_holding")
    assert d.verdict == "pass", d.decided_by


def test_5_must_not_hold_with_no_phone_at_all_passes():
    """With no phone anywhere, the person cannot be holding one."""
    d = _rel_case("The person must NOT be holding a phone.", obj=0, observed=None)
    assert d.verdict == "pass", d.decided_by
    assert "cell phone" in d.reason.lower() or "phone" in d.reason.lower()


def test_an_uncertain_relationship_is_unclear_either_way():
    for std in ("The person must be holding a phone.",
                "The person must NOT be holding a phone."):
        d = _rel_case(std, observed="unclear")
        assert d.verdict == "unclear", f"{std}: {d.decided_by}"


def test_the_subject_must_still_be_present_for_a_negative_rule():
    d = _rel_case("The person must NOT be holding a phone.", person=0, observed="not_holding")
    assert d.verdict == "fail", "no person means the rule cannot be evaluated as met"


# 6-7: object absence is a different rule entirely
def test_6_no_phone_in_view_with_a_phone_present_fails():
    d = _rel_case("There must be no phone in view.", observed=None)
    assert d.verdict == "fail", d.decided_by
    assert d.decided_by == "detector-prohibited"


def test_7_no_phone_in_view_with_no_phone_passes():
    from host.policy import ObjectEvidence, VlmJudgement, decide
    from host.standard_parser import parse_standard
    parsed = parse_standard("There must be no phone in view.")
    evidence = {"cell phone": ObjectEvidence("cell phone", 0, 45, (),
                                             bucket_support=tuple([False] * 6))}
    d = decide(parsed, evidence, VlmJudgement(verdict="pass", reason="No phone in view."))
    assert d.verdict == "pass", d.decided_by


def test_a_phone_on_the_table_passes_the_relationship_rule_but_fails_the_absence_rule():
    """The distinction, in one test: same scene, two standards, two answers."""
    held_nowhere = {"obj": 45, "observed": "not_holding"}
    rel = _rel_case("The person must not be holding a phone.", **held_nowhere)
    ban = _rel_case("There must be no phone in view.", obj=45, observed=None)
    assert rel.verdict == "pass", "a phone on the table does not break a holding rule"
    assert ban.verdict == "fail", "but it does break an absence rule"


# --- the same truth table for the bottle ------------------------------------

def test_bottle_positive_and_negative_forms():
    assert _rel_case("The person must be holding a bottle.",
                     obj_label="bottle", observed="holding").verdict == "pass"
    assert _rel_case("The person must be holding a bottle.",
                     obj_label="bottle", observed="not_holding").verdict == "fail"
    assert _rel_case("The person must NOT be holding a bottle.",
                     obj_label="bottle", observed="holding").verdict == "fail"
    assert _rel_case("The person must NOT be holding a bottle.",
                     obj_label="bottle", observed="not_holding").verdict == "pass"
    assert _rel_case("The person must NOT be holding a bottle.",
                     obj_label="bottle", obj=0).verdict == "pass"


# --- and in Russian ----------------------------------------------------------

def test_russian_relationship_truth_table():
    assert _rel_case("Человек должен держать телефон.", observed="holding").verdict == "pass"
    assert _rel_case("Человек должен держать телефон.", observed="not_holding").verdict == "fail"
    assert _rel_case("Человек не должен держать телефон.", observed="holding").verdict == "fail"
    assert _rel_case("Человек не должен держать телефон.", observed="not_holding").verdict == "pass"
    assert _rel_case("Человек не должен держать телефон.", obj=0).verdict == "pass"


def test_russian_bottle_relationship_truth_table():
    assert _rel_case("Человек не должен держать бутылку.",
                     obj_label="bottle", observed="holding").verdict == "fail"
    assert _rel_case("Человек не должен держать бутылку.",
                     obj_label="bottle", observed="not_holding").verdict == "pass"


# --- operator-facing wording -------------------------------------------------

def test_the_reasons_read_like_a_person_wrote_them():
    passed = _rel_case("The person must not be holding a phone.", observed="not_holding")
    failed = _rel_case("The person must not be holding a phone.", observed="holding")
    for d in (passed, failed):
        assert "relation_expected" not in d.reason
        assert "observed_relationship" not in d.reason
        assert "holding" in d.reason.lower()


# --- the guard must not treat "X is not <doing something>" as "X is absent" ---
#
# The pattern "{o} is not" matched "the person is not holding a phone", which is
# the natural way to report a satisfied negative standard. The guard then decided
# the model had denied the person and overturned a correct verdict.

@pytest.mark.parametrize("sentence", [
    "The person is not holding a phone.",
    "The person is not holding the bottle.",
    "The person is not wearing gloves.",
    "The person is not touching the phone.",
])
def test_a_person_not_doing_something_is_not_a_claim_they_are_absent(sentence):
    from host.policy import contradicts_presence
    assert contradicts_presence(sentence, "person") is False, sentence


@pytest.mark.parametrize("sentence", [
    "No person is visible.",
    "The person is not visible.",
    "The person is not present.",
    "There is no person in the frame.",
    "the person is absent",
])
def test_a_genuine_denial_of_the_person_is_still_caught(sentence):
    from host.policy import contradicts_presence
    assert contradicts_presence(sentence, "person") is True, sentence


def test_a_satisfied_negative_standard_is_not_overturned_by_the_guard():
    """End to end: the natural PASS wording must survive."""
    d = _rel_case("The person must not be holding a phone.",
                  observed="not_holding", vlm_verdict="pass")
    assert d.verdict == "pass", d.decided_by
    assert d.decided_by != "vlm-contradicts-detector"


# --- the relation object is named by the standard ----------------------------
#
# "The person must not be holding a phone" mentions a phone, but the phone is
# neither required nor prohibited - it is the object of the relationship. The
# off-topic guard only knew about required+prohibited, so it decided a reply
# about the phone was about something the standard never mentioned, and a clean
# PASS became UNCLEAR via vlm-offtopic.

def test_a_reply_about_the_relation_object_is_on_topic():
    from host.policy import VlmJudgement, find_offtopic_absence
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must not be holding a phone.")
    vlm = VlmJudgement(verdict="pass", reason="No phone is visible in any frame.",
                       observed_relationship="not_holding")
    assert find_offtopic_absence(vlm, parsed) is None


def test_a_reply_about_an_unrelated_object_is_still_off_topic():
    from host.policy import VlmJudgement, find_offtopic_absence
    from host.standard_parser import parse_standard
    parsed = parse_standard("The person must not be holding a phone.")
    vlm = VlmJudgement(verdict="pass", reason="No bottle is visible.")
    assert find_offtopic_absence(vlm, parsed) == "bottle"


def test_no_phone_anywhere_passes_a_must_not_hold_rule_end_to_end():
    """The scene that exposed it: no phone at all, so nothing can be held."""
    d = _rel_case("The person must NOT be holding a phone.", obj=0,
                  observed="not_holding", vlm_verdict="pass")
    assert d.verdict == "pass", d.decided_by
    assert d.decided_by == "relationship-object-absent"


def test_the_observed_relationship_is_recorded_in_the_audit():
    """It decided the verdict, so it has to be auditable."""
    from host.policy import VlmJudgement
    pub = VlmJudgement(verdict="pass", observed_relationship="not_holding").public()
    assert pub["observed_relationship"] == "not_holding"


def test_the_normalised_rule_is_recorded_in_the_notes():
    d = _rel_case("The person must not be holding a phone.", observed="not_holding")
    assert any("person HOLDING cell phone" in n for n in d.notes), d.notes
    assert any("required: NO" in n for n in d.notes), d.notes


# --- aggregating two independent rule verdicts -------------------------------
#
# The combined verdict is arithmetic, not judgement: the model never sees both
# rules together and never decides the overall result.

def test_aggregate_truth_table():
    from host.policy import aggregate_verdicts
    assert aggregate_verdicts(["pass", "pass"]) == "pass"
    assert aggregate_verdicts(["pass", "fail"]) == "fail"
    assert aggregate_verdicts(["fail", "pass"]) == "fail"
    assert aggregate_verdicts(["fail", "unclear"]) == "fail"
    assert aggregate_verdicts(["unclear", "fail"]) == "fail"
    assert aggregate_verdicts(["pass", "unclear"]) == "unclear"
    assert aggregate_verdicts(["unclear", "pass"]) == "unclear"
    assert aggregate_verdicts(["unclear", "unclear"]) == "unclear"
    assert aggregate_verdicts(["fail", "fail"]) == "fail"


def test_aggregate_of_one_rule_is_that_rule():
    from host.policy import aggregate_verdicts
    for v in ("pass", "fail", "unclear"):
        assert aggregate_verdicts([v]) == v


def test_aggregate_of_nothing_is_unclear():
    """No rules means nothing was established, which is not a pass."""
    from host.policy import aggregate_verdicts
    assert aggregate_verdicts([]) == "unclear"


def test_a_failure_anywhere_dominates_regardless_of_order():
    from host.policy import aggregate_verdicts
    assert aggregate_verdicts(["unclear", "pass", "fail"]) == "fail"


def test_an_unknown_verdict_string_is_treated_as_unclear():
    """A verdict that is not one of the three must never become a pass."""
    from host.policy import aggregate_verdicts
    assert aggregate_verdicts(["pass", "weird"]) == "unclear"
