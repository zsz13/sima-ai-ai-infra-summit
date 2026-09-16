"""Bilingual parsing: an English and a Russian rule must mean the same thing.

The safety property under test is not translation quality. It is that a Russian
standard still produces detector-groundable objects, because a standard that
parses to nothing silently disables grounding in host/policy.py and hands the
verdict to the vision-language model alone - the exact failure temporal
grounding exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from host.standard_parser import (  # noqa: E402
    detect_language,
    parse_standard,
    resolve_language,
)

#: (English phrasing, Russian phrasing, required, prohibited)
EQUIVALENT = [
    ("This person must be holding a phone.",
     "Человек должен держать телефон.",
     ("person", "cell phone"), ()),
    ("The person must be holding a bottle.",
     "Человек должен держать бутылку.",
     ("person", "bottle"), ()),
    ("There must be no phone in view.",
     "В кадре не должно быть телефона.",
     (), ("cell phone",)),
]


@pytest.mark.parametrize(("en", "ru", "required", "prohibited"), EQUIVALENT)
def test_english_and_russian_resolve_to_the_same_semantics(en, ru, required, prohibited):
    pe, pr = parse_standard(en), parse_standard(ru)
    assert set(pe.required) == set(required)
    assert set(pr.required) == set(required)
    assert set(pe.prohibited) == set(prohibited)
    assert set(pr.prohibited) == set(prohibited)


@pytest.mark.parametrize(("en", "ru", "required", "prohibited"), EQUIVALENT)
def test_both_languages_are_detector_grounded(en, ru, required, prohibited):
    """Neither phrasing may fall through to the model-only path."""
    assert parse_standard(en).is_grounded
    assert parse_standard(ru).is_grounded


def test_language_is_detected_from_script():
    assert detect_language("The person must be holding a phone.") == "en"
    assert detect_language("Человек должен держать телефон.") == "ru"
    # An empty or symbol-only standard defaults to English rather than failing.
    assert detect_language("") == "en"


def test_reported_language_travels_with_the_parse():
    assert parse_standard("Человек должен держать телефон.").language == "ru"
    assert parse_standard("The person must be holding a phone.").language == "en"


def test_explicit_language_overrides_detection():
    """What Whisper decoded wins over what the characters look like."""
    parsed = parse_standard("Человек должен держать телефон.", language="ru")
    assert parsed.required == ("person", "cell phone")


# --- Russian morphology: the same noun in different cases ---------------

@pytest.mark.parametrize("text", [
    "Человек должен держать телефон.",
    "У человека должен быть телефон.",
    "Человек с телефоном.",
    "На столе должен быть телефон.",
])
def test_russian_noun_cases_all_reach_cell_phone(text):
    assert "cell phone" in parse_standard(text).required


def test_russian_negation_is_word_bounded():
    """The particle "не" also occurs inside "телефоне"; substring matching would
    turn a requirement into a prohibition."""
    parsed = parse_standard("Человек должен держать телефон.")
    assert parsed.prohibited == ()
    assert "cell phone" in parsed.required


def test_russian_negation_scopes_to_what_follows():
    parsed = parse_standard("Человек не должен держать телефон.")
    assert "person" in parsed.required
    assert "cell phone" in parsed.prohibited


def test_russian_bez_prohibits():
    parsed = parse_standard("Человек без телефона.")
    assert "person" in parsed.required
    assert "cell phone" in parsed.prohibited


def test_russian_endings_do_not_overmatch():
    """A closed ending set keeps "стол" from matching "столько"."""
    assert parse_standard("Столько людей.").required == ("person",)


def test_longest_russian_phrase_wins():
    """"ножниц" must beat "нож"."""
    assert "scissors" in parse_standard("На столе должны быть ножницы.").required
    assert "knife" not in parse_standard("На столе должны быть ножницы.").required


def test_terms_with_no_detector_class_are_not_invented_in_either_language():
    """A hard hat is not a COCO class. The person is grounded; the hat is not."""
    assert parse_standard("The worker must wear a hard hat.").required == ("person",)
    assert parse_standard("Человек должен надеть каску.").required == ("person",)


def test_public_payload_is_serialisable():
    payload = parse_standard("Человек должен держать телефон.").public()
    assert payload["language"] == "ru"
    assert payload["grounded"] is True
    assert payload["required"] == ["person", "cell phone"]


def test_russian_fleeting_vowel_nouns_still_match():
    """цветок loses its "о" when declined (цветком), so the bare stem misses it.

    Under-matching is the dangerous direction: an object the parser fails to find
    is one host/policy.py cannot ground, which silently hands the verdict to the
    vision-language model alone.
    """
    for text in ("Цветок на столе.", "Ваза с цветком.", "Цветка нет в кадре."):
        assert "potted plant" in (parse_standard(text).required
                                  + parse_standard(text).prohibited), text


# --- negation cues must not match inside a word ------------------------

@pytest.mark.parametrize(("text", "required", "prohibited"), [
    # "no " occurs inside "piano ", "casino " and "domino ". Matched as a
    # substring it flipped the polarity of everything after it.
    ("The piano must have a bench.", ("bench",), ()),
    ("The casino table must be clear.", ("dining table",), ()),
    ("A domino and a bottle on the table.", ("bottle", "dining table"), ()),
    # and the real cues must still fire
    ("There must be no phone in view.", (), ("cell phone",)),
    ("The person must not be holding a phone.", ("person",), ("cell phone",)),
])
def test_english_negation_is_word_bounded(text, required, prohibited):
    parsed = parse_standard(text)
    assert set(parsed.required) == set(required), text
    assert set(parsed.prohibited) == set(prohibited), text


def test_apostrophe_contraction_still_negates():
    """"n't" is a suffix, so it takes a trailing boundary and no leading one."""
    assert "cell phone" in parse_standard("The person doesn't have a phone.").prohibited


def test_a_prohibition_in_one_clause_wins_over_a_requirement_in_another():
    """Acting on a prohibition is the safer error, so it wins the conflict."""
    parsed = parse_standard("The person must hold a phone, but no phone on the table.")
    assert "cell phone" in parsed.prohibited
    assert "cell phone" not in parsed.required


def test_unrecognised_terms_leave_the_standard_ungrounded():
    """With no detector-supported object, is_grounded must be False so that
    host/policy.py knows the verdict rests on the model alone."""
    parsed = parse_standard("every label must be legible")
    assert parsed.required == ()
    assert parsed.prohibited == ()
    assert parsed.is_grounded is False
    assert parsed.public()["grounded"] is False


# --- a mislabelled transcript must not silently lose grounding ---------

def test_the_script_wins_over_a_wrong_language_label():
    """Whisper does not always honour a forced language.

    Pinning Russian and speaking English returns the English sentence verbatim.
    If that were parsed with the Russian lexicon it would match nothing, and a
    standard that matches nothing grounds nothing - handing the verdict to the
    vision-language model alone, which is the failure this layer exists to stop.
    """
    text = "The person must be holding a phone."
    assert resolve_language(text, "ru") == "en"
    assert parse_standard(text, resolve_language(text, "ru")).required == \
        ("person", "cell phone")

    ru = "Человек должен держать телефон."
    assert resolve_language(ru, "en") == "ru"
    assert parse_standard(ru, resolve_language(ru, "en")).required == \
        ("person", "cell phone")


def test_a_letterless_standard_falls_back_to_the_label():
    assert resolve_language("123 456", "ru") == "ru"
    assert resolve_language("123 456", None) == "en"
