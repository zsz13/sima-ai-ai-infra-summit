"""Grounding policy: combine temporal detector evidence with the VLM judgement.

The rule this module exists to enforce: **a vision-language model may not
hallucinate an object into existence that the detector never saw.**

The failure that motivated it was real. Standard: "The person must be holding a
smartphone." No smartphone was in the scene. The VLM returned PASS with
"the person is holding a smartphone, as indicated by the visible screen". The
detector supports a `cell phone` class and had reported zero detections; that
signal existed and was being thrown away.

Pure functions, no I/O, so the whole decision table is unit-testable without
hardware.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .standard_parser import ParsedStandard, known_labels, surface_names

PASS, FAIL, UNCLEAR = "pass", "fail", "unclear"


@dataclass(frozen=True)
class ObjectEvidence:
    """How often the detector saw one class across the evidence window."""

    label: str
    frames_present: int
    frames_total: int
    #: detector confidence in each frame where the class was present
    confidences: tuple[float, ...] = ()
    #: One flag per equal time segment of the window: did the detector see this
    #: class confidently at least once in that segment? A frame ratio alone
    #: cannot tell a small object the detector flickers on from one that was
    #: genuinely there for only part of the window. Segment coverage can: the
    #: first is seen in every segment, the second is not.
    bucket_support: tuple[bool, ...] = ()
    #: Longest run of consecutive frames with no confident detection. Reported
    #: for the operator; no rule depends on it.
    longest_gap_frames: int = 0

    @property
    def presence_ratio(self) -> float:
        return self.frames_present / self.frames_total if self.frames_total else 0.0

    @property
    def conf_median(self) -> float:
        return statistics.median(self.confidences) if self.confidences else 0.0

    @property
    def conf_max(self) -> float:
        return max(self.confidences) if self.confidences else 0.0

    @property
    def buckets_supported(self) -> int:
        return sum(1 for b in self.bucket_support if b)

    @property
    def bucket_count(self) -> int:
        return len(self.bucket_support)

    @property
    def spans_window(self) -> bool:
        """Seen confidently in EVERY time segment of the window.

        Requiring every segment is the conservative reading and it is not an
        invented number: it is the definition of "was there throughout". An
        object that appears only at the start, only at the end, or once in the
        middle fails it, which is exactly the distinction the frame ratio misses.
        """
        return self.bucket_count > 0 and self.buckets_supported == self.bucket_count

    def public(self) -> dict:
        return {
            "label": self.label,
            "frames_present": self.frames_present,
            "frames_total": self.frames_total,
            "presence_ratio": round(self.presence_ratio, 4),
            "conf_median": round(self.conf_median, 4),
            "conf_max": round(self.conf_max, 4),
            "buckets_supported": self.buckets_supported,
            "bucket_count": self.bucket_count,
            "spans_window": self.spans_window,
            "longest_gap_frames": self.longest_gap_frames,
        }


@dataclass(frozen=True)
class VlmJudgement:
    """What the vision-language model said about the selected frames."""

    verdict: str = UNCLEAR
    reason: str = ""
    evidence: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    #: per selected frame: True supports the requirement, False contradicts it,
    #: None means the model could not tell from that frame
    per_frame: tuple[bool | None, ...] = ()
    #: what the model OBSERVED about the relationship the standard is about:
    #: "holding", "not_holding" or "unclear". Deliberately an observation and not
    #: a verdict - whether holding means pass or fail is policy, and policy is
    #: decided here rather than by the model. None when the model did not report
    #: one, in which case the verdict is used as a fallback.
    observed_relationship: str | None = None

    @property
    def frames_judged(self) -> int:
        return len(self.per_frame)

    @property
    def frames_supporting(self) -> int:
        return sum(1 for v in self.per_frame if v is True)

    @property
    def frames_contradicting(self) -> int:
        return sum(1 for v in self.per_frame if v is False)

    @property
    def is_contradictory(self) -> bool:
        """The model both supported and contradicted the requirement across frames."""
        return self.frames_supporting > 0 and self.frames_contradicting > 0

    def public(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "missing_evidence": list(self.missing_evidence),
            "frames_judged": self.frames_judged,
            "frames_supporting": self.frames_supporting,
            "frames_contradicting": self.frames_contradicting,
            # The observation the polarity decision was made from. Recorded
            # because a reviewer has to be able to see what the model actually
            # reported, not just which way the rule went.
            "observed_relationship": self.observed_relationship,
        }


@dataclass(frozen=True)
class GroundingConfig:
    """Thresholds. Defaults are conservative and were set from measurements on
    real hardware - see docs/BENCHMARKS.md, "Temporal grounding thresholds"."""

    #: presence ratio at or below which a class is treated as ABSENT
    absent_ratio_max: float = 0.10
    #: presence ratio at or above which a class is treated as RELIABLY PRESENT
    present_ratio_min: float = 0.60
    #: a detection below this confidence does not count as a sighting at all
    min_confidence: float = 0.55
    #: fewer frames than this and the window is too short to judge anything
    min_window_frames: int = 10
    #: How many equal time segments the window is divided into for coverage.
    #: Matches the number of evidence frames, so the segments the policy reasons
    #: about are the same ones the operator sees images from.
    coverage_buckets: int = 6

    def __post_init__(self) -> None:
        if not 0.0 <= self.absent_ratio_max < self.present_ratio_min <= 1.0:
            raise ValueError("require 0 <= absent_ratio_max < present_ratio_min <= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0,1]")
        if self.min_window_frames < 1:
            raise ValueError("min_window_frames must be >= 1")


@dataclass
class Decision:
    verdict: str
    reason: str
    #: which rule decided, so the audit trail explains itself
    decided_by: str
    required: list[dict] = field(default_factory=list)
    prohibited: list[dict] = field(default_factory=list)
    vlm: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "required_objects": self.required,
            "prohibited_objects": self.prohibited,
            "vlm": self.vlm,
            "notes": self.notes,
        }


def _pct(ratio: float) -> str:
    return f"{round(ratio * 100)}%"


def _describe(ev: ObjectEvidence) -> str:
    return (f"{ev.label} in {ev.frames_present}/{ev.frames_total} frames "
            f"({_pct(ev.presence_ratio)})")


#: Phrasings that assert an object is NOT there. Matched against the model's own
#: reason and evidence lists, per object, so the check is scoped to the object the
#: detector actually confirmed rather than firing on any negative sentence.
_ABSENCE_PATTERNS = (
    # "{o} is not" used to be here and was far too broad: it matched "the person
    # is not holding a phone", which is the natural way to report that a negative
    # standard was SATISFIED. The guard then read it as a denial that the person
    # existed and overturned the verdict. The absence claim has to be spelled
    # out, so only the completions that really mean absence are listed.
    "no {o}", "not {o}", "{o} is absent", "{o} is missing",
    "no {o} is visible", "{o} is not visible", "{o} is not present",
    "{o} is not in view", "{o} is not in the frame", "{o} is not there",
    "cannot see {o}", "can't see {o}",
    "without {o}", "there is no {o}", "{o} not present", "no visible {o}",
    "нет {o}", "{o} отсутств", "не видно {o}", "{o} не виден", "{o} не видна",
    "{o} не видно", "без {o}",
)


def contradicts_presence(text: str, label: str, language: str | None = None) -> bool:
    """True when `text` claims `label` is absent.

    Deliberately literal and narrow: it only looks for a claim about this
    specific object, so an honest sentence about something else is untouched.
    Surface names come from the parser's own synonym tables, so a Russian reply
    ("телефон отсутствует") is checked as readily as an English one.
    """
    low = (text or "").lower()
    if not low:
        return False
    for name in surface_names(label, language):
        for pattern in _ABSENCE_PATTERNS:
            if pattern.format(o=name) in low:
                return True
    return False


def aggregate_verdicts(verdicts: list[str]) -> str:
    """Combine independent per-rule verdicts into one overall verdict.

    Deliberately arithmetic rather than judgement, and deliberately not the
    model's job: each rule is judged on its own against the same evidence, and
    the combination is decided here.

        any FAIL     -> FAIL
        any UNCLEAR  -> UNCLEAR
        otherwise    -> PASS

    A verdict string that is none of the three counts as UNCLEAR, so an
    unrecognised value can never be promoted into a pass. No rules at all is
    UNCLEAR too: nothing was established, which is not the same as passing.
    """
    if not verdicts:
        return UNCLEAR
    seen = [v if v in (PASS, FAIL, UNCLEAR) else UNCLEAR for v in verdicts]
    if FAIL in seen:
        return FAIL
    if UNCLEAR in seen:
        return UNCLEAR
    return PASS


def _observed_relationship(vlm: VlmJudgement, want_held: bool | None) -> str | None:
    """What the model observed: "holding", "not_holding", "unclear" or None.

    Prefers the model's explicit `observed_relationship`. When it did not report
    one - an older build, or a model that ignored the field - the overall verdict
    is read back through the polarity the model was told to apply: it was shown
    the expected relationship, so PASS means it saw what was expected and FAIL
    means it saw the opposite. That keeps both backends working while only the
    explicit field is trusted when present.
    """
    raw = (vlm.observed_relationship or "").strip().lower().replace(" ", "_")
    if raw in {"holding", "held", "holds"}:
        return "holding"
    if raw in {"not_holding", "not_held", "none", "no"}:
        return "not_holding"
    if raw == "unclear":
        return "unclear"
    if raw:
        return "unclear"
    if want_held is None or vlm.verdict not in (PASS, FAIL):
        return None
    met = vlm.verdict == PASS
    return "holding" if met == want_held else "not_holding"


def _claim_text(vlm: VlmJudgement) -> str:
    """The parts of the reply that assert something about the scene.

    `missing_evidence` is deliberately excluded. That field exists to name what
    the model would have had to see, so absence wording in it is the field being
    filled in correctly, not a claim that the object was not there. Scanning it
    produced exactly the false positive it looks like: a reply whose reason read
    "The phone is visible in the person's right hand in all frames" was still
    overturned, because its `missing_evidence` said "no phone in the hand".
    `reason` and `evidence` are the fields that make assertions, and they are
    also the only ones an operator is shown.
    """
    return " ".join([vlm.reason, *vlm.evidence])


def find_presence_contradiction(vlm: VlmJudgement, confirmed: list[ObjectEvidence],
                                language: str | None = None) -> str | None:
    """The label the model called absent although the detector confirmed it.

    This exists because it actually happened: a phone detected in 42 of 45 frames
    at 85% median confidence, and the model still answered "No phone is visible
    in any frame." Prompting alone is not a guarantee, so the contradiction is
    caught here, deterministically, before a reason reaches anyone.
    """
    haystack = _claim_text(vlm)
    for ev in confirmed:
        if contradicts_presence(haystack, ev.label, language):
            return ev.label
    return None


def find_offtopic_absence(vlm: VlmJudgement, parsed: ParsedStandard) -> str | None:
    """An object the model says is missing that the standard never asked about.

    "A person must be visible" answered with "No phone is visible" is the model
    describing the scene rather than judging the rule. The wording is about the
    wrong object, so it must not be handed to the operator as the reason for a
    verdict about a person.
    """
    # The subject and object of a relationship are named by the standard just as
    # much as a required or prohibited class is. For "must not be holding a
    # phone" the phone is neither required nor prohibited, and leaving it out
    # here made a perfectly on-topic reply about the phone look like a scene
    # caption, turning a clean PASS into UNCLEAR.
    named = set(parsed.required) | set(parsed.prohibited)
    named.update(x for x in (parsed.relation_subject, parsed.relation_object) if x)
    haystack = _claim_text(vlm)
    for label in known_labels() - named:
        if contradicts_presence(haystack, label, parsed.language):
            return label
    return None


def decide(
    parsed: ParsedStandard,
    evidence: dict[str, ObjectEvidence],
    vlm: VlmJudgement,
    config: GroundingConfig | None = None,
) -> Decision:
    """Apply the grounding policy. Rules are evaluated in this order:

    0. Window too short                      -> UNCLEAR
    1. Required object effectively absent    -> FAIL   (VLM cannot override)
    2. Prohibited object reliably present    -> FAIL   (VLM cannot override)
    3. Required object only intermittent     -> UNCLEAR
    4. Prohibited object only intermittent   -> UNCLEAR
    5. Detector grounding satisfied          -> the model's reply is checked for
                                                self-contradiction, for denying a
                                                measured object, and for being
                                                off-topic
    5b. The standard names a relationship     -> compare the OBSERVED relationship
                                                against the polarity the standard
                                                asks for. "must be holding" and
                                                "must not be holding" differ only
                                                here, which is why the object is
                                                never simply prohibited.
    6. Otherwise                              -> take the model's verdict
    """
    cfg = config or GroundingConfig()
    required = [evidence[c] for c in parsed.required if c in evidence]
    prohibited = [evidence[c] for c in parsed.prohibited if c in evidence]
    req_pub = [e.public() for e in required]
    proh_pub = [e.public() for e in prohibited]
    notes: list[str] = []

    if not parsed.is_grounded:
        notes.append(
            "No detector-supported objects in this standard, so the verdict rests "
            "on the vision-language model alone.")
    if parsed.unsupported:
        named = ", ".join(parsed.unsupported)
        notes.append(
            f"The detector has no class for {named}, so that part of the standard "
            f"was judged by the vision-language model alone and carries no detector "
            f"evidence.")

    total = max((e.frames_total for e in [*required, *prohibited]), default=0)
    if total and total < cfg.min_window_frames:
        return Decision(
            UNCLEAR,
            f"Only {total} frames of evidence were collected, fewer than the "
            f"{cfg.min_window_frames} needed for a reliable judgement.",
            "insufficient-window", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 1: a required object the detector essentially never saw ---
    for ev in required:
        if ev.presence_ratio <= cfg.absent_ratio_max:
            seen = ("never detected" if ev.frames_present == 0
                    else f"detected in only {ev.frames_present} of {ev.frames_total} frames")
            reason = (f"No {ev.label} was found during the inspection window: "
                      f"{seen}. The requirement cannot be met.")
            if vlm.verdict == PASS:
                notes.append(
                    f"The vision-language model reported PASS, but the detector "
                    f"{seen} across the window. Detector evidence wins.")
            return Decision(FAIL, reason, "detector-absent",
                            req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 2: a prohibited object the detector reliably saw ---
    for ev in prohibited:
        if ev.presence_ratio >= cfg.present_ratio_min:
            return Decision(
                FAIL,
                f"A {ev.label} must not be present, but one was {_describe(ev)}.",
                "detector-prohibited", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 2b: a prohibited object present throughout, despite flicker ---
    for ev in prohibited:
        if ev.presence_ratio < cfg.present_ratio_min and ev.spans_window:
            return Decision(
                FAIL,
                f"A {ev.label} must not be present. It was detected in every one of "
                f"the {ev.bucket_count} time segments of the window "
                f"({_describe(ev)}), so it was there throughout.",
                "detector-prohibited", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 3: a required object seen only intermittently ---
    #
    # Frame ratio alone is too brittle here. A small object the detector flickers
    # on - a phone in a hand at 56% of frames - is not the same as an object that
    # was genuinely there for only part of the window, but the ratio scores them
    # identically. Segment coverage separates them: if the object was seen
    # confidently in EVERY segment, it was present throughout and the misses are
    # detector flicker, so grounding holds and the model judges the relationship.
    # Anything less than every segment is still intermittent.
    for ev in required:
        if ev.presence_ratio >= cfg.present_ratio_min:
            continue
        if ev.spans_window:
            notes.append(
                f"The {ev.label} was detected in only {_pct(ev.presence_ratio)} of "
                f"frames, but in all {ev.bucket_count} time segments of the window "
                f"(median confidence {_pct(ev.conf_median)}, longest gap "
                f"{ev.longest_gap_frames} frames). Treated as present throughout: "
                f"the misses are detector flicker, not absence.")
            continue
        return Decision(
            UNCLEAR,
            f"The {ev.label} was visible inconsistently - {_describe(ev)}, covering "
            f"{ev.buckets_supported} of {ev.bucket_count} time segments - so there "
            f"is not enough evidence for a reliable judgement.",
            "intermittent-required", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 4: a prohibited object seen intermittently ---
    for ev in prohibited:
        if ev.presence_ratio > cfg.absent_ratio_max:
            return Decision(
                UNCLEAR,
                f"Something that looks like a {ev.label} appeared intermittently - "
                f"{_describe(ev)}, covering {ev.buckets_supported} of "
                f"{ev.bucket_count} time segments - so the prohibition cannot be "
                f"judged reliably.",
                "intermittent-prohibited", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 5: grounding is satisfied; the VLM judges the semantics ---
    if required:
        notes.append("Detector confirmed every required object across the window; "
                     "the relationship was judged by the vision-language model.")

    if vlm.is_contradictory:
        return Decision(
            UNCLEAR,
            f"The model read the evidence differently across frames "
            f"({vlm.frames_supporting} support, {vlm.frames_contradicting} contradict), "
            f"so the result is not reliable.",
            "vlm-contradictory", req_pub, proh_pub, vlm.public(), notes)

    # The model may not overturn a measurement. Presence is the detector's
    # question and it has already answered it; the model was asked about the
    # relationship. A reply that denies a confirmed object is answering the wrong
    # question, so its verdict AND its wording are both discarded rather than
    # shown - "No phone is visible" must never be what an operator reads when the
    # phone was detected in 42 of 45 frames.
    denied = find_presence_contradiction(vlm, required, parsed.language)
    if denied is not None:
        ev = next(e for e in required if e.label == denied)
        notes.append(
            f"The model's reply claimed the {denied} was not visible, but the "
            f"detector measured it {_describe(ev)}. Presence is settled by "
            f"measurement, so the model's reading was discarded.")
        return Decision(
            UNCLEAR,
            f"{denied.capitalize()} presence was confirmed by the detector "
            f"({_describe(ev)}), but the visual relationship could not be "
            f"determined reliably.",
            "vlm-contradicts-detector", req_pub, proh_pub, vlm.public(), notes)

    # The reason must be about the rule that was asked. A reply that denies an
    # object the standard never mentions is a scene caption, not a judgement, and
    # showing it would answer a question the operator did not ask.
    offtopic = find_offtopic_absence(vlm, parsed)
    if offtopic is not None:
        notes.append(
            f"The model's reply was about a {offtopic}, which this standard does "
            f"not mention. It was discarded as off-topic.")
        return Decision(
            UNCLEAR,
            "The model's answer did not address this standard, so the result is "
            "not reliable.",
            "vlm-offtopic", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 5b: the standard is about a relationship, not just presence ---
    #
    # "The person must not be holding a phone" is not "the phone is prohibited":
    # a phone on the table breaks the second rule and not the first. So the
    # object is measured but never forbidden, and the decision is made by
    # comparing what the model OBSERVED against the polarity the standard asks
    # for. The model reports an observation; whether that observation passes is
    # decided here.
    #
    # This runs AFTER the contradiction and off-topic guards on purpose: a model
    # that denies a detector-confirmed object is answering the wrong question,
    # and its relationship reading is no more trustworthy than its presence one.
    if parsed.relation and parsed.relation_object:
        obj = evidence.get(parsed.relation_object)
        label = parsed.relation_object
        want_held = parsed.relation_expected

        # With the object nowhere in the window, the relationship is settled by
        # measurement: you cannot hold what is not there. That satisfies a
        # negative rule outright. For a positive rule Rule 1 has already failed
        # it, because there the object is genuinely required.
        if want_held is False and obj is not None and \
                obj.presence_ratio <= cfg.absent_ratio_max and not obj.spans_window:
            seen = ("never detected" if obj.frames_present == 0
                    else f"detected in only {obj.frames_present} of {obj.frames_total} frames")
            return Decision(
                PASS,
                f"No {label} was visible during the inspection window ({seen}), "
                f"so the person cannot have been holding one.",
                "relationship-object-absent", req_pub, proh_pub, vlm.public(), notes)

        observed = _observed_relationship(vlm, want_held)
        notes.append(
            f"Normalised rule: {parsed.describe_rule()}. "
            f"Observed: {observed or 'not reported'}.")

        if observed in (None, "unclear"):
            return Decision(
                UNCLEAR,
                f"Whether the person is holding the {label} could not be "
                f"determined from the evidence frames.",
                "relationship-unclear", req_pub, proh_pub, vlm.public(), notes)

        held = observed == "holding"
        met = held == want_held
        if met:
            headline = (f"The person is holding the {label}, as the standard requires."
                        if held else
                        f"The {label} is visible, but the person is not holding it.")
        else:
            headline = (f"The person is holding the {label}, which violates the standard."
                        if held else
                        f"The person is not holding the {label}, which the standard requires.")
        # The headline states the polarity decision plainly, because that is the
        # part an operator has to act on. The model's own sentence is kept after
        # it when it adds detail the headline cannot carry - "lying on the table"
        # is worth more than a second restatement of the rule.
        reason = headline
        extra = (vlm.reason or "").strip()
        if extra and extra.lower() not in headline.lower():
            reason = f"{headline} {extra}"
        return Decision(PASS if met else FAIL, reason,
                        "relationship-met" if met else "relationship-violated",
                        req_pub, proh_pub, vlm.public(), notes)

    if vlm.verdict == PASS:
        return Decision(PASS, vlm.reason or "The requirement is met.",
                        "vlm", req_pub, proh_pub, vlm.public(), notes)
    if vlm.verdict == FAIL:
        return Decision(FAIL, vlm.reason or "The requirement is not met.",
                        "vlm", req_pub, proh_pub, vlm.public(), notes)

    return Decision(UNCLEAR,
                    vlm.reason or "The evidence was not sufficient to decide.",
                    "vlm", req_pub, proh_pub, vlm.public(), notes)


def build_evidence(
    frames: list[list[dict]],
    labels: list[str],
    config: GroundingConfig | None = None,
) -> dict[str, ObjectEvidence]:
    """Aggregate per-frame detections into per-class temporal evidence.

    `frames` is one list of detection dicts per frame in the window. A class
    counts as present in a frame if any detection of that class clears
    `min_confidence`; the highest such confidence in the frame is recorded.

    The window is also divided into equal time segments and each is marked
    supported if the class was seen confidently at least once inside it. That is
    computed over EVERY frame, not over the handful of images sent to the model,
    so coverage reflects what the detector actually measured.
    """
    cfg = config or GroundingConfig()
    total = len(frames)
    n_buckets = max(1, cfg.coverage_buckets) if total else 0
    out: dict[str, ObjectEvidence] = {}
    for label in labels:
        present = 0
        confs: list[float] = []
        seen: list[bool] = []
        for dets in frames:
            best = max((float(d.get("confidence", 0.0)) for d in dets
                        if d.get("label") == label), default=0.0)
            hit = best >= cfg.min_confidence
            seen.append(hit)
            if hit:
                present += 1
                confs.append(best)

        support: list[bool] = []
        for b in range(n_buckets):
            lo = b * total // n_buckets
            hi = (b + 1) * total // n_buckets
            support.append(any(seen[lo:hi]))

        gap = longest = 0
        for hit in seen:
            gap = 0 if hit else gap + 1
            longest = max(longest, gap)

        out[label] = ObjectEvidence(label, present, total, tuple(confs),
                                    tuple(support), longest)
    return out
