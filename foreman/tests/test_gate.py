"""Unit tests for the presence/stability gate.

These run on the Mac with no DevKit and no models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host.gate import Detection, Gate, GateConfig, GateState, iou  # noqa: E402

BOX = (0.30, 0.30, 0.60, 0.60)


def det(bbox=BOX, label="box", conf=0.9, track_id=1) -> Detection:
    return Detection(label=label, confidence=conf, bbox=bbox, track_id=track_id)


def feed(gate: Gate, frames: list[list[Detection]]) -> list[bool]:
    return [gate.update(f) for f in frames]


# --- iou ---------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    assert iou(BOX, BOX) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.9, 0.9)) == 0.0


def test_iou_touching_edges_is_zero_not_negative():
    assert iou((0, 0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5)) == 0.0


def test_iou_half_overlap():
    # two unit-ish boxes overlapping on exactly half their area
    a, b = (0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 0.3, 0.2)
    assert iou(a, b) == pytest.approx(1 / 3)


def test_iou_is_symmetric():
    a, b = (0.0, 0.0, 0.4, 0.4), (0.2, 0.1, 0.5, 0.6)
    assert iou(a, b) == pytest.approx(iou(b, a))


# --- firing behaviour --------------------------------------------------

def test_fires_once_after_stable_frames():
    g = Gate(GateConfig(stable_frames=3))
    fired = feed(g, [[det()]] * 6)
    assert fired == [False, False, True, False, False, False], fired
    assert g.state is GateState.INSPECTED


def test_does_not_fire_before_settling():
    g = Gate(GateConfig(stable_frames=5))
    assert feed(g, [[det()]] * 4) == [False] * 4
    assert g.state is GateState.SETTLING


def test_moving_item_never_fires():
    g = Gate(GateConfig(stable_frames=3, stable_iou=0.9))
    frames = [[det(bbox=(0.1 * i, 0.1 * i, 0.1 * i + 0.2, 0.1 * i + 0.2))] for i in range(8)]
    assert not any(feed(g, frames))


def test_item_that_settles_after_moving_fires():
    g = Gate(GateConfig(stable_frames=3))
    moving = [[det(bbox=(0.1, 0.1, 0.3, 0.3))], [det(bbox=(0.5, 0.5, 0.7, 0.7))]]
    assert not any(feed(g, moving))
    assert feed(g, [[det()]] * 3) == [False, False, True]


def test_low_confidence_is_ignored():
    g = Gate(GateConfig(stable_frames=2, min_confidence=0.8))
    assert not any(feed(g, [[det(conf=0.5)]] * 10))
    assert g.state is GateState.WAITING


def test_label_filter_excludes_other_labels():
    g = Gate(GateConfig(stable_frames=2, labels=frozenset({"bottle"})))
    assert not any(feed(g, [[det(label="person")]] * 10))


def test_label_filter_admits_listed_label():
    g = Gate(GateConfig(stable_frames=2, labels=frozenset({"bottle"})))
    assert feed(g, [[det(label="bottle")]] * 2) == [False, True]


def test_picks_most_confident_qualifying_detection():
    g = Gate(GateConfig(stable_frames=2))
    frame = [det(bbox=(0.0, 0.0, 0.1, 0.1), conf=0.6), det(bbox=BOX, conf=0.95)]
    feed(g, [frame, frame])
    assert g.last_trigger is not None
    assert g.last_trigger.confidence == pytest.approx(0.95)


# --- re-arming ---------------------------------------------------------

def test_rearms_only_after_item_leaves_then_fires_again():
    cfg = GateConfig(stable_frames=2, absent_frames=3)
    g = Gate(cfg)
    assert feed(g, [[det()]] * 2) == [False, True]
    # still present: must not fire again
    assert not any(feed(g, [[det()]] * 5))
    # leaves
    assert not any(feed(g, [[]] * 3))
    assert g.state is GateState.WAITING
    # next item fires
    assert feed(g, [[det()]] * 2) == [False, True]


def test_brief_dropout_does_not_rearm():
    cfg = GateConfig(stable_frames=2, absent_frames=4)
    g = Gate(cfg)
    feed(g, [[det()]] * 2)              # fires
    feed(g, [[]] * 3)                   # 3 < absent_frames, still INSPECTED
    assert g.state is GateState.INSPECTED
    assert not any(feed(g, [[det()]] * 5))


def test_absent_counter_resets_on_reappearance():
    cfg = GateConfig(stable_frames=2, absent_frames=3)
    g = Gate(cfg)
    feed(g, [[det()]] * 2)
    feed(g, [[]] * 2)        # 2 absent
    feed(g, [[det()]])       # resets the absent counter
    feed(g, [[]] * 2)        # 2 absent again, still short of 3
    assert g.state is GateState.INSPECTED


def test_reset_clears_everything():
    g = Gate(GateConfig(stable_frames=2))
    feed(g, [[det()]] * 2)
    g.reset()
    assert g.state is GateState.WAITING
    assert g.last_trigger is None
    assert feed(g, [[det()]] * 2) == [False, True]


# --- progress ----------------------------------------------------------

def test_progress_reports_settling_fraction():
    g = Gate(GateConfig(stable_frames=4))
    g.update([det()])
    assert g.progress == pytest.approx(0.25)
    g.update([det()])
    assert g.progress == pytest.approx(0.50)


def test_progress_is_zero_when_waiting_and_one_when_inspected():
    g = Gate(GateConfig(stable_frames=2))
    assert g.progress == 0.0
    feed(g, [[det()]] * 2)
    assert g.progress == 1.0


# --- config validation -------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"min_confidence": 1.5},
    {"stable_iou": -0.1},
    {"stable_frames": 0},
    {"absent_frames": 0},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        GateConfig(**kwargs)
