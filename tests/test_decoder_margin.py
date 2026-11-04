"""CPU-only: the batched decoder in decoder_margin must pick the vendor decoders' own paths."""

from itertools import product

import torch

from echora_kara.engine.ctc_segmentation import align_with_minimum_durations
from echora_kara.train import decoder_margin as dm

BLANK = 29


def spans_of(path, lat_row, groups, frames):
    flat = [(k, i) for k, g in enumerate(groups) for i in range(len(g))]
    out = {}
    for t in range(frames):
        index = lat_row.letter[path[t]]
        if index >= 0:
            out.setdefault(flat[index], [t, t])[1] = t
    return [[(out[(k, i)][0], out[(k, i)][1] + 1) for i in range(len(g))] for k, g in enumerate(groups)]


def vendor(spans):
    return [[(s.start, s.end) for s in g] for g in spans]


def test_plain_pass_matches_the_ordinary_ctc_decoder():
    for seed in range(5):
        torch.manual_seed(seed)
        groups = [[3, 1, 20], [4, 15], [7, 7, 9], [2]]
        emission = torch.randn(45, 30).log_softmax(-1)
        lat_row = dm.lattice(groups, [1] * 9, [0, 0, 1, 1], BLANK, blank_free=False, gap_cost=0.0)
        lat = dm.stack([lat_row], "cpu")
        em = emission[None].gather(2, lat.label.unsqueeze(1).expand(-1, 45, -1))
        _, path = dm.viterbi(em, lat, torch.tensor([45]), backtrace=True)
        # One-frame floors without blank-free tokens are the ordinary CTC topology (the plain pass).
        expected = align_with_minimum_durations(emission, groups, blank=BLANK, minimum_frames=[1] * 9)
        assert spans_of(path[0], lat_row, groups, 45) == vendor(expected)


def test_final_lattice_matches_the_production_decoder_with_floors_and_gap_cost():
    for seed in range(5):
        torch.manual_seed(seed)
        groups = [[3, 1, 20], [4, 15], [7, 7, 9], [2, 5]]
        lines = [0, 0, 1, 1]
        floors = [2, 1, 3, 1, 2, 2, 1, 1, 1, 2]
        emission = torch.randn(60, 30).log_softmax(-1)
        lat_row = dm.lattice(groups, floors, lines, BLANK, blank_free=True, gap_cost=0.2)
        lat = dm.stack([lat_row], "cpu")
        em = emission[None].gather(2, lat.label.unsqueeze(1).expand(-1, 60, -1))
        _, path = dm.viterbi(em, lat, torch.tensor([60]), backtrace=True)
        expected = align_with_minimum_durations(
            emission,
            groups,
            blank=BLANK,
            minimum_frames=floors,
            blank_free_tokens=True,
            line_ranges=[(0, 2), (2, 4)],
            in_line_gap_cost=0.2,
        )
        assert spans_of(path[0], lat_row, groups, 60) == vendor(expected)


def test_gold_restriction_is_the_best_measured_path():
    torch.manual_seed(3)
    regions = [(3, 8, [5, 6]), (11, 15, [8, 9, 4])]
    groups = [ids for *_, ids in regions]
    emission = torch.randn(18, 30).log_softmax(-1)
    lat_row = dm.lattice(groups, [1] * 5, [0, 0], BLANK, blank_free=True, gap_cost=0.0)
    lat = dm.stack([lat_row], "cpu")
    em = emission[None].gather(2, lat.label.unsqueeze(1).expand(-1, 18, -1))
    owner = torch.full((1, 18), -1)
    for k, (first, stop, _) in enumerate(regions):
        owner[0, first:stop] = k
    gold = float(dm.viterbi(em, lat, torch.tensor([18]), gold_owner=owner)[0])
    expected = float(emission[[t for t in range(18) if owner[0, t] < 0], BLANK].sum())
    for first, stop, ids in regions:
        best = float("-inf")
        for cuts in product(range(first + 1, stop), repeat=len(ids) - 1):
            bounds = [first, *cuts, stop]
            if all(a < b for a, b in zip(bounds, bounds[1:])):
                best = max(
                    best, sum(float(emission[a:b, letter].sum()) for a, b, letter in zip(bounds, bounds[1:], ids))
                )
        expected += best
    assert abs(gold - expected) < 1e-4


def batch(regions, frames, lines=None):
    table = torch.zeros((1, len(regions), 3), dtype=torch.long)
    targets = torch.zeros((1, len(regions), max(len(r[2]) for r in regions)), dtype=torch.long)
    for column, (first, stop, ids) in enumerate(regions):
        table[0, column] = torch.tensor([first, stop, len(ids)])
        targets[0, column, : len(ids)] = torch.tensor(ids)
    line_table = torch.tensor([lines or [0] * len(regions)])
    return table, targets, torch.tensor([frames]), line_table


def sharp(regions, frames):
    emission = torch.full((frames, 30), -8.0)
    emission[:, BLANK] = -0.01
    for first, stop, ids in regions:
        bounds = [first + (stop - first) * i // len(ids) for i in range(len(ids) + 1)]
        for a, b, letter in zip(bounds, bounds[1:], ids):
            emission[a:b] = -8.0
            emission[a:b, letter] = -0.01
    return emission


def test_hinge_is_zero_on_gold_audio_and_positive_with_the_right_gradient_when_shifted():
    regions = [(10, 18, [1, 2]), (22, 30, [3, 4])]
    table, targets, frames, lines = batch(regions, 44, [0, 0])
    loss, stats = dm.decoder_margin_loss(
        sharp(regions, 44)[None], table, targets, frames, blank=BLANK, region_lines=lines
    )
    assert float(loss) == 0.0 and float(stats["windows"]) == 1
    shifted = sharp(regions, 44).roll(8, dims=0).requires_grad_(True)
    loss, _ = dm.decoder_margin_loss(shifted[None], table, targets, frames, blank=BLANK, region_lines=lines)
    assert float(loss.detach()) > 0
    loss.backward()
    assert shifted.grad[10, 1] < 0 and shifted.grad[18, 1] > 0  # raise gold's first letter, lower the decoder's


def test_text_bonus_reaches_the_text_head_and_activity_runs():
    regions = [(10, 18, [1, 2]), (22, 30, [3, 4])]
    table, targets, frames, lines = batch(regions, 44, [0, 1])
    columns = torch.tensor([[[0, 0], [1, 1]]])
    tokens = torch.randn(1, 44, 2).log_softmax(-1).requires_grad_(True)
    wave = torch.randn(1, 44 * 320 + 80)
    loss, _ = dm.decoder_margin_loss(
        sharp(regions, 44).roll(8, dims=0)[None],
        table,
        targets,
        frames,
        blank=BLANK,
        region_lines=lines,
        region_columns=columns,
        token_log_probs=tokens,
        waveforms=wave,
        sample_counts=torch.tensor([wave.shape[1]]),
    )
    loss.backward()
    assert float(loss.detach()) > 0 and tokens.grad.abs().sum() > 0


def test_touching_repeated_letter_gives_the_junction_frame_to_blank():
    regions = [(2, 6, [1, 2]), (6, 10, [2, 3])]
    assert dm.gold_bounds([(*r, 0, [-1, -1]) for r in regions]) == [(2, 6), (7, 10)]
    table, targets, frames, lines = batch(regions, 12)
    loss, stats = dm.decoder_margin_loss(
        sharp(regions, 12)[None], table, targets, frames, blank=BLANK, region_lines=lines
    )
    assert float(stats["windows"]) == 1 and float(loss) == 0.0  # a 1-frame shift is within tolerance
    tight = [(2, 4, [1, 2]), (4, 6, [2, 3])]
    assert dm.gold_bounds([(*r, 0, [-1, -1]) for r in tight]) is None
