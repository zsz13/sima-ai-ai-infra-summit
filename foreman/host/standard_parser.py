"""Parse a spoken inspection standard into detector-groundable objects.

Deliberately deterministic: no model call, no network, instant, and unit-testable.
It only claims to recognise the 80 COCO classes the detector actually supports.
A term it does not recognise simply does not appear in `required` or
`prohibited`, and `is_grounded` then tells the caller that the detector cannot
back this standard up - host/policy.py says so in its notes and the console shows
a "no detector grounding" chip, rather than implying a check that is not happening.

Known limits, shared by both languages and confirmed by adversarial review
rather than assumed. Each fails toward a wrong *polarity*, so they are listed
here rather than hidden:

  - Negation is positional, not syntactic. "Телефон не должен лежать на столе"
    / "The phone must not be on the table" puts the object before the cue, so
    the phone reads as required and the table as prohibited.
  - A coordinated prohibition splits at "и" / "and" into a clause with no cue of
    its own: "не должно быть телефона и бутылки" prohibits the phone and
    *requires* the bottle.
  - A mixed-script standard is classified by letter count, so Latin brand names
    in a Russian sentence ("не должно быть iPhone и MacBook") can select the
    English lexicon, which cannot see the Cyrillic "не".

Fixing these properly needs a syntactic parser, not more keywords. Until then the
console shows the parsed reading back to the operator, so a misread rule is
visible before it is acted on.

English and Russian are both parsed here, and both produce the *same* internal
representation. That matters for safety rather than convenience: if a Russian
standard parsed to zero objects, host/policy.py would find nothing to ground and
would fall back to trusting the vision-language model alone - disabling the exact
protection that temporal grounding exists to provide. A rule must not become
weaker because of the language it was spoken in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: English. Everyday words mapped to the COCO class the detector emits. Only
#: classes the detector can actually produce appear here; the value must match a
#: line in assets/coco.txt exactly.
EN_SYNONYMS: dict[str, str] = {
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

#: Russian. Keys are stems, matched with a bounded set of noun endings (see
#: RU_ENDING), so "телефон" also matches "телефона"/"телефону"/"телефоном" without
#: needing a morphological analyser. Irregular forms are listed outright.
RU_SYNONYMS: dict[str, str] = {
    "телефон": "cell phone", "смартфон": "cell phone", "мобильник": "cell phone",
    "мобильный телефон": "cell phone", "трубк": "cell phone",
    "бутылк": "bottle", "бутылок": "bottle", "бутыль": "bottle",
    "чашк": "cup", "кружк": "cup", "стакан": "cup",
    "ноутбук": "laptop", "лэптоп": "laptop", "компьютер": "laptop",
    "клавиатур": "keyboard", "мышк": "mouse", "мышь": "mouse",
    "пульт": "remote", "книг": "book", "ножниц": "scissors",
    "нож": "knife", "вилк": "fork", "ложк": "spoon",
    "миск": "bowl", "тарелк": "bowl",
    "банан": "banana", "яблок": "apple", "апельсин": "orange",
    "бутерброд": "sandwich", "сэндвич": "sandwich", "пицц": "pizza",
    "пончик": "donut", "торт": "cake",
    "рюкзак": "backpack", "сумк": "handbag", "чемодан": "suitcase",
    "зонт": "umbrella", "галстук": "tie",
    "человек": "person", "люди": "person", "людей": "person",
    "людям": "person", "людьми": "person", "работник": "person",
    "рабочий": "person", "рабочего": "person", "оператор": "person",
    "сотрудник": "person",
    "стул": "chair", "кресл": "chair", "диван": "couch",
    "телевизор": "tv", "монитор": "tv", "экран": "tv",
    # "часов" is dropped on purpose: it is also the genitive plural of "час"
    # (hour), so "проверка длится шесть часов" used to require a clock.
    "часы": "clock", "ваз": "vase",
    "растени": "potted plant", "цветок": "potted plant",
    # fleeting vowel: цветок -> цветка/цветку/цветком, so the stem loses the "о"
    "цветк": "potted plant",
    "мишк": "teddy bear", "зубная щётка": "toothbrush", "зубная щетка": "toothbrush",
    "фен": "hair drier",
    "велосипед": "bicycle", "машин": "car", "автомобил": "car",
    "грузовик": "truck", "автобус": "bus", "мотоцикл": "motorcycle",
    "собак": "dog", "кошк": "cat", "птиц": "bird", "лошад": "horse",
    "мяч": "sports ball", "скамейк": "bench", "унитаз": "toilet",
    "раковин": "sink", "холодильник": "refrigerator", "духовк": "oven",
    "микроволновк": "microwave", "тостер": "toaster",
    "стол": "dining table", "кроват": "bed",
    "светофор": "traffic light", "знак стоп": "stop sign",
    "скейтборд": "skateboard", "сноуборд": "snowboard", "лыж": "skis",
    "бокал": "wine glass", "брокколи": "broccoli", "морков": "carrot",
    "хот-дог": "hot dog", "лодк": "boat", "самолёт": "airplane",
    "самолет": "airplane", "поезд": "train", "слон": "elephant",
    "медвед": "bear", "овц": "sheep", "коров": "cow",
}

#: Russian noun endings. Deliberately a closed set rather than "any few letters":
#: a wildcard suffix makes "стол" match "столько" and "кот" match "который".
#: "о" is needed for neuter nominatives - without it "кресло" and "яблоко" did not
#: match at all, so a Russian standard grounded where its English twin did not.
RU_ENDING = r"(?:ами|ах|ам|ов|ой|ом|ем|ей|ья|ьи|ью|и|ы|а|у|е|ю|я|о|й|ь)?"

#: Phrases that flip the REST of a clause to "must NOT be present". Negation is
#: scoped to what follows the cue, not to the whole clause, so "the person must
#: not be holding a phone" requires a person and prohibits a phone.
#:
#: Matched on word boundaries, like the Russian cues. Plain substring matching
#: found "no " inside "piano ", "casino " and "domino ", which silently flipped
#: the polarity of everything after it: "the piano must have a bench" prohibited
#: the bench instead of requiring it. "n't" is deliberately a suffix - it only
#: ever appears attached to the end of a word ("doesn't"), so it takes a trailing
#: boundary and no leading one.
EN_NEGATIONS = (
    r"\bmust not\b", r"\bmust never\b", r"\bshould not\b", r"\bshall not\b",
    r"\bcannot\b", r"\bcan not\b", r"\bmay not\b", r"\bwithout\b",
    r"\bfree of\b", r"\bfree from\b", r"\babsent\b", r"\bremoved\b",
    r"\bnever\b", r"n't\b", r"\bno\b", r"\bnot\b",
)

#: Russian negation cues, matched on word boundaries. Substring matching would be
#: wrong here: the particle "не" also occurs inside "телефоне" and "кресле".
RU_NEGATIONS = (
    # "не менее / не более / не реже / не чаще двух бутылок" is a quantifier, not
    # a prohibition - it used to forbid the very object the standard demands.
    r"\bне\b(?!\s+(?:менее|более|реже|чаще|ниже|выше))",
    r"\bнет\b", r"\bбез\b", r"\bнельзя\b", r"\bни\b",
    r"запрещ", r"отсутств",
    # "убрать" and the imperative "уберите"/"убери" share no single stem
    r"убра", r"убер",
)

#: Split points that start a new clause with its own polarity.
EN_CLAUSE_SPLIT = re.compile(r",|\band\b|\bbut\b|\bwhile\b|\bwhereas\b|;")
RU_CLAUSE_SPLIT = re.compile(r",|\bи\b|\bно\b|\bа\b|\bтакже\b|;")

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")

#: The two languages Foreman accepts a spoken standard in.
LANGUAGES = ("en", "ru")


@dataclass(frozen=True)
class Lexicon:
    synonyms: dict[str, str]
    negations: tuple[str, ...]
    clause_split: re.Pattern[str]
    #: True when negation cues are regular expressions rather than literals.
    #: Both lexicons use regexes; the flag stays so a future literal lexicon
    #: does not silently get compiled as a pattern.
    negations_are_regex: bool
    #: optional inflection suffix appended to every synonym pattern
    ending: str


LEXICONS: dict[str, Lexicon] = {
    "en": Lexicon(EN_SYNONYMS, EN_NEGATIONS, EN_CLAUSE_SPLIT, True, "e?s?"),
    "ru": Lexicon(RU_SYNONYMS, RU_NEGATIONS, RU_CLAUSE_SPLIT, True, RU_ENDING),
}


def detect_language(text: str) -> str:
    """Which of the two supported languages a standard is written in.

    Script is the signal, because that is what actually distinguishes them and it
    cannot be fooled by vocabulary overlap. Anything not predominantly Cyrillic is
    treated as English, which is the safe default: the English lexicon is larger.
    """
    cyr = len(_CYRILLIC.findall(text or ""))
    lat = len(_LATIN.findall(text or ""))
    return "ru" if cyr > lat else "en"


def resolve_language(text: str, preferred: str | None = None) -> str:
    """Which lexicon should parse this text.

    The script wins over a caller's label, because the lexicon has to match the
    alphabet the words are actually written in: parsing Latin text with the
    Russian lexicon matches nothing, and a standard that matches nothing grounds
    nothing. `preferred` is only consulted when the text has no letters at all to
    judge by.
    """
    if not (_CYRILLIC.search(text or "") or _LATIN.search(text or "")):
        return preferred if preferred in LANGUAGES else "en"
    return detect_language(text)


@dataclass(frozen=True)
class ParsedStandard:
    raw: str
    #: COCO classes that must be visibly present
    required: tuple[str, ...] = ()
    #: COCO classes that must NOT be present
    prohibited: tuple[str, ...] = ()
    #: the language the standard was parsed as
    language: str = "en"

    @property
    def is_grounded(self) -> bool:
        """True when at least one object can be checked against the detector."""
        return bool(self.required or self.prohibited)

    def public(self) -> dict:
        return {
            "raw": self.raw,
            "language": self.language,
            "required": list(self.required),
            "prohibited": list(self.prohibited),
            "grounded": self.is_grounded,
        }


def _match_objects(clause: str, lex: Lexicon) -> list[tuple[int, str]]:
    """(position, COCO class) for each object mentioned, in order of appearance.

    Longest phrase first so 'cell phone' wins over 'phone' and 'ножниц' over
    'нож'; a matched span is blanked out so it cannot match again. Inflection is
    handled by the lexicon's ending pattern - plurals in English, noun case
    endings in Russian.
    """
    found: list[tuple[int, str]] = []
    remaining = clause
    for phrase in sorted(lex.synonyms, key=len, reverse=True):
        pattern = rf"\b{re.escape(phrase)}{lex.ending}\b"
        for m in re.finditer(pattern, remaining):
            cls = lex.synonyms[phrase]
            if cls not in (c for _, c in found):
                found.append((m.start(), cls))
        # blank the span so a shorter synonym cannot re-match inside it
        remaining = re.sub(pattern, lambda m: " " * len(m.group(0)), remaining)
    return sorted(found)


def _negation_position(clause: str, lex: Lexicon) -> int | None:
    """Character offset of the earliest negation cue, or None."""
    hits: list[int] = []
    for cue in lex.negations:
        if lex.negations_are_regex:
            m = re.search(cue, clause)
            if m:
                hits.append(m.start())
        else:
            at = clause.find(cue)
            if at >= 0:
                hits.append(at)
    return min(hits) if hits else None


def parse_standard(text: str, language: str | None = None) -> ParsedStandard:
    """Split the standard into clauses and assign each mentioned object a polarity.

    Negation is scoped to what follows the cue within a clause, and only the
    first object after the cue is treated as prohibited; anything later in the
    clause is context and is left ungrounded rather than guessed at. Objects are
    never placed in both lists - an explicit prohibition wins, because acting on
    a prohibition is the safer error.

    `language` is "en" or "ru"; when omitted it is detected from the script, so
    an English and a Russian phrasing of the same rule produce identical
    `required` and `prohibited` sets.
    """
    raw = (text or "").strip()
    lang = language if language in LEXICONS else detect_language(raw)
    lex = LEXICONS[lang]
    low = raw.lower()
    required: list[str] = []
    prohibited: list[str] = []

    for clause in lex.clause_split.split(low):
        clause = clause.strip()
        if not clause:
            continue
        neg_at = _negation_position(clause, lex)
        prohibited_in_clause = 0
        for pos, cls in _match_objects(clause, lex):
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
        language=lang,
    )
