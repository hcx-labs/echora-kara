"""CPU-only: the text head's inference windows never exceed its training length and lose no frame."""

import pytest

from echora_kara.model.text_aligner import TRAINING_WINDOW_FRAMES, WINDOW_OVERLAP_FRAMES, frame_windows


@pytest.mark.parametrize("frames", [1, 249, 250, 251, 1000, 1250, 1251, 2249, 2250, 10500, 10501])
def test_windows_cover_every_frame_once(frames):
    kept = []
    for piece, keep in frame_windows(frames):
        assert piece.stop - piece.start <= TRAINING_WINDOW_FRAMES
        kept += list(range(piece.start, piece.stop))[keep]
    assert kept == list(range(frames))


def test_neighbours_share_the_overlap():
    windows = frame_windows(3 * TRAINING_WINDOW_FRAMES)
    assert len(windows) > 1
    for (before, _), (after, _) in zip(windows, windows[1:]):
        assert before.stop - after.start == WINDOW_OVERLAP_FRAMES


def test_a_short_song_is_one_window():
    assert frame_windows(900) == [(slice(0, 900), slice(0, 900))]
