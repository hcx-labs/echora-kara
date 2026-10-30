"""Dependency-free save completion/pruning and strict resume identity policy."""
import json
from pathlib import Path

MARKER = 'save_complete.json'


def complete_checkpoint(path):
    path = Path(path)
    try:
        sizes = json.loads((path / MARKER).read_text())
        return bool(sizes) and all((path / name).stat().st_size == size and size > 0
                                   for name, size in sizes.items())
    except (OSError, ValueError, TypeError):
        return False


def mark_and_prune(path):
    """Call ONLY after Trainer.on_save: keep two completed optimizer states.

    Unmarked (legacy or interrupted) directories are never pruned. A new
    directory alone is not proof of a successful save.
    """
    path = Path(path)
    required = ['optimizer.pt', 'scheduler.pt', 'trainer_state.json']
    weights = list(path.glob('*.safetensors')) or list(path.glob('pytorch_model*.bin'))
    rng = list(path.glob('rng_state*.pth'))
    if not weights or not rng:
        raise ValueError(f'Incomplete checkpoint: {path}')
    files = [path / name for name in required] + weights + rng
    sizes = {p.name: p.stat().st_size for p in files}
    if not all(sizes.values()):
        raise ValueError(f'Empty checkpoint file: {path}')
    json.loads((path / 'trainer_state.json').read_text())
    temporary = path / (MARKER + '.tmp')
    temporary.write_text(json.dumps(sizes))
    temporary.replace(path / MARKER)
    completed = sorted((p for p in path.parent.glob('checkpoint-*')
                        if p.name.removeprefix('checkpoint-').isdigit() and complete_checkpoint(p)),
                       key=lambda p: int(p.name.split('-')[-1]))
    for old in completed[:-2]:
        (old / 'optimizer.pt').unlink(missing_ok=True)


def validate_resume(identity_path, identity, resume_requested, checkpoint):
    if resume_requested and not checkpoint:
        raise ValueError('--resume requested but no checkpoint exists')
    if not checkpoint:
        return
    path = Path(identity_path)
    if not path.exists():
        raise ValueError('Resume refused: identity is unrecorded')
    old = json.loads(path.read_text())
    if 'training_recipe' not in old:
        raise ValueError('Legacy resume refused: optimizer/schedule recipe was not recorded. '
                         'Do not bypass identity checks. Migrate weights only with --model CHECKPOINT '
                         'in a NEW output directory, without --resume; this starts a new optimizer '
                         'and schedule, not a continuation.')
    if old != identity:
        raise ValueError('Resume refused: dataset/window/training recipe identity changed')
    for name in ('optimizer.pt', 'scheduler.pt', 'trainer_state.json'):
        if not (Path(checkpoint) / name).is_file():
            raise ValueError(f'Resume refused: missing {name}')
    if not complete_checkpoint(checkpoint):
        raise ValueError('Resume refused: checkpoint completion validation failed')
