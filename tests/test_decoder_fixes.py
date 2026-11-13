"""CPU-only checks of the decoder fixes: capped in-line gaps, inline backing, voice-onset pull, line-end extension."""

import math
import string

import numpy as np
import pytest
import torch
from torchaudio.functional import TokenSpan

from echora_kara.engine import aligner as al
from echora_kara.engine.ctc_segmentation import align_with_minimum_durations

VOCAB = {letter: i for i, letter in enumerate(string.ascii_lowercase)}
BLANK = 29
HIGH, LOW = math.log(0.9), math.log(1e-4)


def lattice(frames, active, size=30, letter=0.9):
    """Log-probs: blank everywhere except ``active`` {(start, end): letter}, where that letter has ``letter``."""
    emission = torch.full((frames, size), LOW)
    emission[:, BLANK] = HIGH
    for (start, end), name in active.items():
        emission[start:end, BLANK] = math.log(1 - letter) if letter < 0.9 else LOW
        emission[start:end, VOCAB[name]] = math.log(letter)
    return emission


def starts(spans):
    return [group[0].start for group in spans]


def test_gap_cap_off_is_the_plain_decoder():
    torch.manual_seed(0)
    groups = [[3, 1, 20], [4, 15], [7, 9], [2, 5]]
    emission = torch.randn(60, 30).log_softmax(-1)
    options = dict(blank=BLANK, minimum_frames=[1] * 9, line_ranges=[(0, 2), (2, 4)], in_line_gap_cost=0.2)
    plain = align_with_minimum_durations(emission, groups, **options)
    # A cap at or below the per-frame cost is no cap.
    assert align_with_minimum_durations(emission, groups, in_line_gap_cap=0.2, **options) == plain
    assert align_with_minimum_durations(emission, groups, in_line_gap_cap=0.0, **options) == plain


def test_capped_gap_keeps_a_real_pause_inside_a_line():
    # One line "da ... da" with a 120-frame pause between the two units, heard only moderately well.
    emission = lattice(160, {(5, 10): "d", (10, 15): "a", (135, 140): "d", (140, 145): "a"}, letter=0.6)
    groups = [[VOCAB["d"], VOCAB["a"]], [VOCAB["d"], VOCAB["a"]]]
    options = dict(blank=BLANK, minimum_frames=[1] * 4, line_ranges=[(0, 2)], in_line_gap_cost=0.5)
    linear = align_with_minimum_durations(emission, groups, **options)
    capped = align_with_minimum_durations(emission, groups, in_line_gap_cap=1.0, in_line_gap_free_cost=0.0, **options)
    # Linear: 120 frames x 0.5 nats is dearer than misplacing a unit, so the gap is closed.
    assert starts(linear)[1] - linear[0][-1].end < 10
    # Capped: the pause costs 1 nat at most, so the second unit stays where it is sung.
    assert starts(capped) == [5, 135]


def test_capped_gap_lets_a_letter_follow_directly():
    # Adjacent units in a line need no blank frame between them, with the cap as without it.
    emission = lattice(20, {(2, 5): "b", (5, 8): "a", (8, 11): "k", (11, 14): "i"})
    groups = [[VOCAB["b"], VOCAB["a"]], [VOCAB["k"], VOCAB["i"]]]
    spans = align_with_minimum_durations(
        emission,
        groups,
        blank=BLANK,
        minimum_frames=[1] * 4,
        line_ranges=[(0, 2)],
        in_line_gap_cost=0.5,
        in_line_gap_cap=1.0,
    )
    assert starts(spans) == [2, 8]


def test_span_result_never_starts_before_the_song():
    spans = [TokenSpan(1, 0, 3, 0.5)]
    result = al._span_result("a", spans, 0.02, offset=-0.1)
    assert result["original_start"] == 0.0
    assert result["original_end"] >= result["original_start"]


def test_second_stream_holds_skip_over_the_other_stream():
    spans = [
        [TokenSpan(1, 0, 2, 1.0)],  # lead
        [TokenSpan(2, 3, 4, 1.0)],  # second stream, inside the lead's gap
        [TokenSpan(3, 8, 9, 1.0)],  # lead
    ]
    held = al._assign_ctc_blank_holds(spans, [(0, 3)], 0.02, 1.0, second_stream={1})
    # The first lead word is held up to the next lead word, not to the echo.
    assert held[0][-1].end == 8
    assert held[1][-1].end == 4


def test_pull_to_voice_onset_moves_a_late_start_after_silence():
    frame = 0.02
    rms = torch.zeros(200)
    rms[100:160] = 1.0  # voice from frame 100 after 2 s of silence
    spans = [[TokenSpan(1, 10, 20, 1.0)], [TokenSpan(2, 150, 160, 1.0)]]
    assert al.pull_to_voice_onset(spans, rms, frame_shift=frame) == [1]
    assert spans[1][0].start == 100


def test_pull_to_voice_onset_leaves_short_lead_ins():
    rms = torch.zeros(200)
    rms[140:160] = 1.0  # 0.2 s before the word: a breath, not a missed word
    spans = [[TokenSpan(2, 150, 160, 1.0)]]
    assert al.pull_to_voice_onset(spans, rms, frame_shift=0.02) == []
    assert spans[0][0].start == 150


def test_frame_rms_has_one_value_per_frame():
    waveform = torch.ones(1, 16000)
    rms = al.frame_rms(waveform, 16000, 60, 0.02)
    assert rms.shape == (60,)
    assert rms[:50].allclose(torch.ones(50)) and rms[50:].eq(0).all()


class FakeTokenizer:
    def get_vocab(self):
        return dict(VOCAB)


class FakeAligner:
    """Just enough of Wav2Vec2ForcedAligner for align_audio_with_timeline."""

    blank = BLANK
    backbone = "wav2vec2"
    text_aligner = None
    last_edge_logits = None
    last_frame_times = None
    last_inference_passes = 1

    def __init__(self, emission, background):
        self.emission = emission
        self.last_background_log_probs = background
        self.tokenizer = FakeTokenizer()

    def tokenize(self, batch):
        return [[VOCAB[c] for c in word] for word in batch]

    def infer(self, waveform, sample_rate):
        return self.emission, None, 0.02


@pytest.fixture
def song(monkeypatch):
    # Line 1: "ba di (no)" with the bracketed echo sung over "di" on the background head; line 2: "ka".
    emission = lattice(
        150,
        {
            (10, 14): "b",
            (14, 20): "a",
            (30, 34): "d",
            (34, 40): "i",
            (44, 48): "n",
            (48, 52): "o",
            (100, 104): "k",
            (104, 110): "a",
        },
    )
    background = lattice(150, {(28, 33): "n", (33, 38): "o"})
    fake = FakeAligner(emission, background)
    monkeypatch.setattr(al, "_wav2vec_aligner", lambda *args, **kwargs: fake)
    monkeypatch.delenv("ECHORA_KARA_IN_LINE_GAP_COST", raising=False)
    return np.zeros(150 * 320, dtype=np.float32)


def test_inline_backing_is_placed_on_the_background_head(song):
    results = al.align_audio_with_timeline(
        song,
        [["ba", "di", "no"], ["ka"]],
        sr=16000,
        use_gpu=False,
        hf_model_id="fake",
        blank_free_tokens=True,
        inline_backing_tokens=[False, False, True, False],
        inline_backing_anchor_words=1,
    )
    assert len(results) == 4
    assert [r.get("stream") for r in results] == [None, None, 2, None]
    echo, lead = results[2], results[1]
    # The echo overlaps the lead word it answers instead of following it.
    assert echo["original_start"] < lead["original_end"]
    diagnostics = results[0]["source_diagnostics"]
    assert diagnostics["inline_backing"] is True
    assert diagnostics["inline_backing_moved_groups"] == 1


def test_inline_backing_needs_one_flag_per_token(song):
    with pytest.raises(ValueError):
        al.align_audio_with_timeline(
            song, [["ba", "di", "no"], ["ka"]], sr=16000, hf_model_id="fake", inline_backing_tokens=[True]
        )


def test_line_end_extension_can_be_switched_off(song, monkeypatch):
    lines = [["ba", "di", "no"], ["ka"]]
    extended = al.align_audio_with_timeline(song, lines, sr=16000, hf_model_id="fake", blank_free_tokens=True)
    monkeypatch.setenv("ECHORA_KARA_LINE_END_EXTEND_S", "0")
    plain = al.align_audio_with_timeline(song, lines, sr=16000, hf_model_id="fake", blank_free_tokens=True)
    assert "line_end_extend" in extended[0]["source_diagnostics"]
    assert "line_end_extend" not in plain[0]["source_diagnostics"]
    # A line's last word never ends after the next lead word starts.
    assert extended[2]["original_end"] <= extended[3]["original_start"]
    assert extended[2]["original_end"] >= plain[2]["original_end"]


def test_vocable_vowels_follow_the_vowel_the_model_hears(monkeypatch):
    # "uu" is sung (and heard) as o at frames 60-80.
    emission = lattice(120, {(10, 14): "b", (14, 20): "a", (60, 80): "o"})
    fake = FakeAligner(emission, None)
    monkeypatch.setattr(al, "_wav2vec_aligner", lambda *args, **kwargs: fake)
    audio = np.zeros(120 * 320, dtype=np.float32)
    on = al.align_audio_with_timeline(audio, [["ba", "uu"]], sr=16000, hf_model_id="fake")
    assert on[0]["source_diagnostics"]["vocable_vowels"] is True
    assert 1.15 <= on[1]["original_start"] <= 1.25
    monkeypatch.setenv("ECHORA_KARA_VOCABLE_VOWELS", "0")
    off = al.align_audio_with_timeline(audio, [["ba", "uu"]], sr=16000, hf_model_id="fake")
    # Without the merged column the unit has no evidence anywhere.
    assert on[1]["score"] > 0.5 > 0.01 > off[1]["score"]


def test_cli_keeps_opening_backing_on_the_lead_and_marks_inline_backing(tmp_path, monkeypatch):
    import json

    import soundfile as sf

    from echora_kara.inference import cli

    emission = lattice(
        200,
        {
            (2, 4): "l",
            (4, 8): "a",
            (10, 12): "l",
            (12, 16): "a",
            (30, 34): "b",
            (34, 40): "a",
            (50, 54): "d",
            (54, 60): "i",
            (64, 68): "n",
            (68, 72): "o",
            (120, 124): "k",
            (124, 130): "a",
        },
    )
    background = lattice(200, {(48, 53): "n", (53, 58): "o"})
    fake = FakeAligner(emission, background)
    monkeypatch.setattr(al, "_wav2vec_aligner", lambda *args, **kwargs: fake)
    (tmp_path / "i.txt").write_text("(la la)\nba di (no)\nka\n", encoding="utf-8")
    sf.write(tmp_path / "i.wav", np.zeros(200 * 320, dtype=np.float32), 16000)
    cli.main(
        ["--path_io", str(tmp_path), "--input_audio", "i.wav", "--hf_model_path", "fake", "--no-gpu"]
        + ["--blank_free_tokens", "--background_decoding", "--inline_backing", "--inline_backing_anchor", "1"]
    )
    document = json.loads((tmp_path / "o.alignment.json").read_text(encoding="utf-8"))
    lines = document["alignment"]["lines"]
    assert lines[0]["voice"] == "background"  # an opening backing line, aligned on the lead path
    assert lines[0]["start_ms"] < lines[1]["start_ms"]
    assert [token.get("stream") for token in lines[1]["tokens"]] == [None, None, 2]
    assert document["diagnostics"]["opening_backing_lines_on_lead"] == 1
    assert document["diagnostics"]["inline_backing_moved_groups"] == 1
