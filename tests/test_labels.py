"""Pinned English labels: models are trained on these exact targets, so a change here changes what a
checkpoint expects at inference. Update only together with a new label-rules version."""

import pytest

from echora_kara.text import label_rules, normalization

CASES = [
    ("feminine", False, ["fe", "ma", "nan"]),
    ("feminine", True, ["fe", "mi", "nin"]),
    ("intuition", False, ["intuishan"]),
    ("intuition", True, ["in", "tu", "i", "shon"]),
    ("because", False, ["bi", "koz"]),
    ("the", False, ["dha"]),
    ("the", True, ["dhe"]),
]


@pytest.mark.parametrize("word,sung,expected", CASES)
def test_english_pronunciation_units(word, sung, expected):
    """v3 English pronunciation items match the pinned syllable labels, with and without sung vowels."""
    items, _ = label_rules.pronunciation_items_v3(word, "en", None, sung=sung)
    assert [item["pron"] for item in items] == expected


def test_normalizer_version_is_recorded():
    """The v3-sung normalizer version string is the pinned one."""
    assert normalization.normalizer_version("v3-sung") == (
        "ko_pronunciation_v2+scripts_v1+label_rules_v3_r2+sung_vowels_r1:v3-sung"
    )
