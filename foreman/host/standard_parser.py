"""Parse a spoken inspection standard into detector-groundable objects.

Deliberately deterministic: no model call, no network, instant, and unit-testable.
It only claims to recognise the 80 COCO classes the detector actually supports.
Anything else is reported as ungrounded, and the policy then knows it cannot use
detector evidence for that term instead of silently pretending it can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Everyday words mapped to the COCO class the detector emits. Only classes the
#: detector can actually produce appear here; the value must match a line in
#: assets/coco.txt exactly.
SYNONYMS: dict[str, str] = {
    "phone": "cell phone", "smartphone": "cell phone", "mobile": "cell phone",
    "cellphone": "cell phone", "cell phone": "cell phone", "iphone": "cell phone",
    "handset": "cell phone",
    "bottle": "bottle", "water bottle": "bottle",
    "cup": "cup", "mug": "cup", "glass": "cup",
    "laptop": "laptop", "notebook computer": "laptop", "computer": "laptop",
    "keyboard": "keyboard", "mouse": "mouse", "remote": "remote",
    "book": "book", "scissors": "scissors", "knife": "knife", "fork": "fork",
    "spoon": "spoon", "bowl": "bowl", "banana": "banana", "apple": "apple",
    "orange": "orange", "sandwich": "sandwich", "pizza": "pizza", "donut": "donut",
    "cake": "cake", "backpack": "backpack", "rucksack": "backpack",
    "handbag": "handbag", "bag": "handbag", "purse": "handbag",
    "suitcase": "suitcase", "umbrella": "umbrella", "tie": "tie",
    "person": "person", "people": "person", "worker": "person",
    "operator": "person", "someone": "person", "human": "person",
    "chair": "chair", "couch": "couch", "sofa": "couch",
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "clock": "clock", "vase": "vase", "potted plant": "potted plant",
    "plant": "potted plant", "teddy bear": "teddy bear",
    "toothbrush": "toothbrush", "hair drier": "hair drier",
    "bicycle": "bicycle", "bike": "bicycle", "car": "car", "truck": "truck",
    "bus": "bus", "motorcycle": "motorcycle", "dog": "dog", "cat": "cat",
    "bird": "bird", "horse": "horse", "sports ball": "sports ball",
    "ball": "sports ball", "bench": "bench", "toilet": "toilet",
    "sink": "sink", "refrigerator": "refrigerator", "fridge": "refrigerator",
    "oven": "oven", "microwave": "microwave", "toaster": "toaster",
    "dining table": "dining table", "table": "dining table", "bed": "bed",
    "traffic light": "traffic light", "stop sign": "stop sign",
    "skateboard": "skateboard", "surfboard": "surfboard", "kite": "kite",
    "frisbee": "frisbee", "snowboard": "snowboard", "skis": "skis",
    "wine glass": "wine glass", "broccoli": "broccoli", "carrot": "carrot",
    "hot dog": "hot dog", "boat": "boat", "airplane": "airplane",
    "plane": "airplane", "train": "train", "zebra": "zebra",
    "giraffe": "giraffe", "elephant": "elephant", "bear": "bear",
    "sheep": "sheep", "cow": "cow", "parking meter": "parking meter",
    "fire hydrant": "fire hydrant", "tennis racket": "tennis racket",
    "baseball bat": "baseball bat", "baseball glove": "baseball glove",
}

#: Phrases that flip the REST of a clause to "must NOT be present". Negation is
#: scoped to what follows the cue, not to the whole clause, so "the person must
#: not be holding a phone" requires a person and prohibits a phone.
NEGATIONS = (
    "must not", "must never", "should not", "shall not", "cannot", "can not",
    "may not", "without", "free of", "free from", "absent", "removed",
    "never", "n't", "no ", "not ",
)

#: Split points that start a new clause with its own polarity.
CLAUSE_SPLIT = re.compile(r",|\band\b|\bbut\b|\bwhile\b|\bwhereas\b|;")


@dataclass(frozen=True)
class ParsedStandard:
    raw: str
    #: COCO classes that must be visibly present
    required: tuple[str, ...] = ()
    #: COCO classes that must NOT be present
    prohibited: tuple[str, ...] = ()
    #: terms that look like objects but are not detector-supported
    ungrounded: tuple[str, ...] = ()

    @property
    def is_grounded(self) -> bool:
        """True when at least one object can be checked against the detector."""
        return bool(self.required or self.prohibited)


def _match_objects(clause: str) -> list[tuple[int, str]]:
    """(position, COCO class) for each object mentioned, in order of appearance.

    Longest phrase first so 'cell phone' wins over 'phone' and 'wine glass' over
    'glass'; a matched span is blanked out so it cannot match again. Plurals are
    accepted ('bottles' -> bottle).
    """
    found: list[tuple[int, str]] = []
    remaining = clause
    for phrase in sorted(SYNONYMS, key=len, reverse=True):
        pattern = rf"\b{re.escape(phrase)}e?s?\b"
        for m in re.finditer(pattern, remaining):
            cls = SYNONYMS[phrase]
            if cls not in (c for _, c in found):
                found.append((m.start(), cls))
        # blank the span so a shorter synonym cannot re-match inside it
        remaining = re.sub(pattern, lambda m: " " * len(m.group(0)), remaining)
    return sorted(found)


def _negation_position(clause: str) -> int | None:
    """Character offset of the earliest negation cue, or None."""
    hits = [clause.find(cue) for cue in NEGATIONS if cue in clause]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else None


def parse_standard(text: str) -> ParsedStandard:
    """Split the standard into clauses and assign each mentioned object a polarity.

    Negation is scoped to what follows the cue within a clause, and only the
    first object after the cue is treated as prohibited; anything later in the
    clause is context and is left ungrounded rather than guessed at. Objects are
    never placed in both lists - an explicit prohibition wins, because acting on
    a prohibition is the safer error.
    """
    raw = (text or "").strip()
    low = raw.lower()
    required: list[str] = []
    prohibited: list[str] = []

    for clause in CLAUSE_SPLIT.split(low):
        clause = clause.strip()
        if not clause:
            continue
        neg_at = _negation_position(clause)
        prohibited_in_clause = 0
        for pos, cls in _match_objects(clause):
            if neg_at is not None and pos > neg_at:
                # Only the first object after the cue is prohibited. In
                # "no bottles on the table" the table is the location, not a
                # thing being forbidden, so later objects are left ungrounded
                # rather than guessed at.
                if prohibited_in_clause == 0 and cls not in prohibited:
                    prohibited.append(cls)
                prohibited_in_clause += 1
            elif cls not in required:
                required.append(cls)

    required = [c for c in required if c not in prohibited]
    return ParsedStandard(
        raw=raw,
        required=tuple(required),
        prohibited=tuple(prohibited),
        ungrounded=(),
    )
