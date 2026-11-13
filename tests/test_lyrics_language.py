"""lyrics_language votes per line; fastText is stubbed so the tests need no model file."""

import pytest

from echora_kara.text import label_rules

ALLOWED = frozenset({"en", "de", "ko"})


@pytest.fixture
def lid(monkeypatch):
    """Stub fastText: each test maps line text to a (language, probability) answer."""
    answers = {}
    monkeypatch.setattr(label_rules, "line_lid", lambda text: answers[text])
    return answers


def test_confident_majority_in_allowed_is_returned(lid):
    lid.update({"one two three": ("en", 0.9), "four five six": ("en", 0.8), "eins zwei": ("de", 0.9)})
    assert label_rules.lyrics_language(["One two three", "four, five six!", "eins zwei"], ALLOWED) == "en"


def test_single_words_do_not_vote(lid):
    lid.update({"one two": ("de", 0.9)})
    # "Hello" alone is skipped (one ASCII word), so only the German vote counts.
    assert label_rules.lyrics_language(["Hello", "one two"], ALLOWED) == "de"


def test_language_outside_allowed_is_not_returned(lid):
    lid.update({"un deux trois": ("fr", 0.95), "quatre cinq": ("fr", 0.95)})
    assert label_rules.lyrics_language(["un deux trois", "quatre cinq"], ALLOWED) is None


def test_unsure_lines_keep_the_generic_path(lid):
    # Confident lines must hold 60 % of all voting letters.
    lid.update({"aaaa bbbb": ("en", 0.9), "cccc dddd": ("en", 0.2), "eeee ffff": ("de", 0.3)})
    assert label_rules.lyrics_language(["aaaa bbbb", "cccc dddd", "eeee ffff"], ALLOWED) is None


def test_split_vote_returns_none(lid):
    lid.update({"aaaa bbbb": ("en", 0.9), "cccc dddd": ("de", 0.9)})
    assert label_rules.lyrics_language(["aaaa bbbb", "cccc dddd"], ALLOWED) is None


def test_unspaced_script_votes_with_four_letters(lid):
    lid.update({"가나다라": ("ko", 0.9)})
    assert label_rules.lyrics_language(["가나다라", "가나"], ALLOWED) == "ko"
