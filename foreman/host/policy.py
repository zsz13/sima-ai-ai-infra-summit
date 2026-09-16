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

from .standard_parser import ParsedStandard

PASS, FAIL, UNCLEAR = "pass", "fail", "unclear"


@dataclass(frozen=True)
class ObjectEvidence:
    """How often the detector saw one class across the evidence window."""

    label: str
    frames_present: int
    frames_total: int
    #: detector confidence in each frame where the class was present
    confidences: tuple[float, ...] = ()

    @property
    def presence_ratio(self) -> float:
        return self.frames_present / self.frames_total if self.frames_total else 0.0

    @property
    def conf_median(self) -> float:
        return statistics.median(self.confidences) if self.confidences else 0.0

    @property
    def conf_max(self) -> float:
        return max(self.confidences) if self.confidences else 0.0

    def public(self) -> dict:
        return {
            "label": self.label,
            "frames_present": self.frames_present,
            "frames_total": self.frames_total,
            "presence_ratio": round(self.presence_ratio, 4),
            "conf_median": round(self.conf_median, 4),
            "conf_max": round(self.conf_max, 4),
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
    5. Detector grounding satisfied          -> defer to the VLM, which alone can
                                                judge relationships and attributes
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

    # --- Rule 3: a required object seen only intermittently ---
    for ev in required:
        if ev.presence_ratio < cfg.present_ratio_min:
            return Decision(
                UNCLEAR,
                f"The {ev.label} was visible inconsistently - {_describe(ev)} - so "
                f"there is not enough evidence for a reliable judgement.",
                "intermittent-required", req_pub, proh_pub, vlm.public(), notes)

    # --- Rule 4: a prohibited object seen intermittently ---
    for ev in prohibited:
        if ev.presence_ratio > cfg.absent_ratio_max:
            return Decision(
                UNCLEAR,
                f"Something that looks like a {ev.label} appeared intermittently - "
                f"{_describe(ev)} - so the prohibition cannot be judged reliably.",
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
    """
    cfg = config or GroundingConfig()
    total = len(frames)
    out: dict[str, ObjectEvidence] = {}
    for label in labels:
        present = 0
        confs: list[float] = []
        for dets in frames:
            best = max((float(d.get("confidence", 0.0)) for d in dets
                        if d.get("label") == label), default=0.0)
            if best >= cfg.min_confidence:
                present += 1
                confs.append(best)
        out[label] = ObjectEvidence(label, present, total, tuple(confs))
    return out
