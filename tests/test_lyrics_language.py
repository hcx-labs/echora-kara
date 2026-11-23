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


# --- line_language: every line decides its own language ---

ENGLISH = set("so i i'm my the you to me be don't crazy sweet baby when chasing main hum bas".split())


@pytest.fixture
def english(monkeypatch):
    """Stub the pronunciation dictionary, so the tests need no resources."""
    monkeypatch.setattr(label_rules, "_english_known", lambda word: word in ENGLISH)
    label_rules.line_language.cache_clear()
    yield
    label_rules.line_language.cache_clear()


def test_romanised_line_of_an_english_song_is_read_as_written(lid, english):
    # fastText has no class for romanised Urdu: it guesses, unsure. The share of known words decides.
    lid.update({"mera bhi usse batana bhool na": ("en", 0.61), "chal diya chal diya": ("it", 0.3)})
    assert label_rules.line_language("Mera bhi usse batana bhool na", "en") == "auto"
    assert label_rules.line_language("Chal diya, chal diya", "en") == "auto"


def test_english_with_unknown_words_stays_english(lid, english):
    lid.update(
        {
            "i'm so undebatable unavoidable": ("en", 0.7),
            "selfie selfie selfie selfie": ("it", 0.4),
            "when i'm chasing you": ("en", 0.53),
        }
    )
    # Half known and leaning English (coinages); one unknown word repeated; everything known.
    assert label_rules.line_language("I'm so undebatable, unavoidable", "en") == "en"
    assert label_rules.line_language("Selfie, selfie, selfie, selfie", "en") == "en"
    assert label_rules.line_language("When I'm chasing you", "en") == "en"


def test_english_line_of_any_other_song_is_english(lid, english):
    lid.update({"don't be crazy sweet baby": ("en", 0.95)})
    for language in ("ur", "es", "ja", "auto", "de"):
        assert label_rules.line_language("Don't be crazy, sweet baby", language) == "en", language
    # The song's own script on the line keeps the song's hint, which also selects how that script is read.
    assert label_rules.line_language("Don't be crazy, sweet baby پیار", "ur") == "ur"
