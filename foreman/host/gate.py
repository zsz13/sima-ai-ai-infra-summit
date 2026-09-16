"""Presence-and-stability gate.

Decides *when* to spend a VLM inference. The detector runs continuously and
cheaply on the MLA; the VLM costs seconds. This gate turns a stream of
detections into at most one inspection per physical item.

Pure logic, no I/O, so it is unit-testable without a DevKit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # normalised to [0,1] as x1, y1, x2, y2
    bbox: tuple[float, float, float, float]
    track_id: int | None = None


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two boxes. 0.0 when they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class GateState(Enum):
    WAITING = "waiting"      # nothing in the frame worth inspecting
    SETTLING = "settling"    # something is there, waiting for it to hold still
    INSPECTED = "inspected"  # already judged; waiting for it to leave


@dataclass(frozen=True)
class GateConfig:
    #: labels that count as an inspectable item; None means "any label"
    labels: frozenset[str] | None = None
    min_confidence: float = 0.50
    #: how much a box must overlap its previous position to count as "held still"
    stable_iou: float = 0.90
    #: consecutive stable frames required before firing
    stable_frames: int = 5
    #: consecutive frames with no qualifying detection before re-arming
    absent_frames: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0,1]")
        if not 0.0 <= self.stable_iou <= 1.0:
            raise ValueError("stable_iou must be in [0,1]")
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if self.absent_frames < 1:
            raise ValueError("absent_frames must be >= 1")


@dataclass
class Gate:
    config: GateConfig = field(default_factory=GateConfig)
    state: GateState = GateState.WAITING
    _anchor: tuple[float, float, float, float] | None = None
    _stable_count: int = 0
    _absent_count: int = 0
    #: the detection that caused the most recent fire, for evidence/telemetry
    last_trigger: Detection | None = None

    def qualifies(self, det: Detection) -> bool:
        if det.confidence < self.config.min_confidence:
            return False
        return self.config.labels is None or det.label in self.config.labels

    def _best(self, detections: list[Detection]) -> Detection | None:
        """The most confident qualifying detection, or None."""
        candidates = [d for d in detections if self.qualifies(d)]
        return max(candidates, key=lambda d: d.confidence, default=None)

    def update(self, detections: list[Detection]) -> bool:
        """Feed one frame's detections. Returns True exactly once per item,
        on the frame where it has settled and should be inspected."""
        best = self._best(detections)

        if best is None:
            self._absent_count += 1
            if self._absent_count >= self.config.absent_frames:
                self._rearm()
            return False

        self._absent_count = 0

        if self.state is GateState.INSPECTED:
            # Already judged this item; it has to leave before we look again.
            return False

        if self._anchor is None or iou(self._anchor, best.bbox) < self.config.stable_iou:
            # New item, or the item moved: restart the stability count.
            self._anchor = best.bbox
            self._stable_count = 1
            self.state = GateState.SETTLING
            return False

        self._stable_count += 1
        if self._stable_count >= self.config.stable_frames:
            self.state = GateState.INSPECTED
            self.last_trigger = best
            return True
        return False

    def _rearm(self) -> None:
        self.state = GateState.WAITING
        self._anchor = None
        self._stable_count = 0
        self._absent_count = 0

    def reset(self) -> None:
        """Forget all state, e.g. when the operator states a new standard."""
        self._rearm()
        self.last_trigger = None

    @property
    def progress(self) -> float:
        """0.0-1.0 settling progress, for the UI."""
        if self.state is not GateState.SETTLING:
            return 1.0 if self.state is GateState.INSPECTED else 0.0
        return min(1.0, self._stable_count / self.config.stable_frames)
