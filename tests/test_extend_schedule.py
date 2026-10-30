import pytest

from echora_kara.train.train import extend_identity, windows_seen


def identity(batch=4, accumulation=2, max_steps=70000, decay=20000, **extra):
    return {
        "manifests": {"train.jsonl": "abc"},
        "seed": 1,
        "batch_size": batch,
        "gradient_accumulation": accumulation,
        "world_size": 1,
        "training_recipe": {"max_steps": max_steps, "decay_steps": decay, "learning_rate": 8e-6},
        **extra,
    }


def test_windows_seen_counts_each_segment_at_its_own_rate():
    assert windows_seen([(0, 8)], 34000) == 272000
    assert windows_seen([(0, 8), (34000, 16)], 34000) == 272000
    assert windows_seen([(0, 8), (34000, 16)], 40000) == 272000 + 6000 * 16


def test_extension_records_the_new_schedule_from_the_resume_step():
    extended = extend_identity(identity(), identity(batch=16, accumulation=1, max_steps=234000, decay=100000), 34000)
    assert extended["schedule_history"] == [
        {"from_step": 0, "windows_per_step": 8, "batch_size": 4, "gradient_accumulation": 2, "max_steps": 70000,
         "decay_steps": 20000},
        {"from_step": 34000, "windows_per_step": 16, "batch_size": 16, "gradient_accumulation": 1,
         "max_steps": 234000, "decay_steps": 100000},
    ]
    # Resuming the extended run again changes nothing further.
    again = extend_identity(extended, identity(batch=16, accumulation=1, max_steps=234000, decay=100000), 60000)
    assert again["schedule_history"] == extended["schedule_history"]


def test_extension_refuses_any_other_change():
    with pytest.raises(ValueError, match="learning|training_recipe"):
        new = identity(max_steps=234000)
        new["training_recipe"]["learning_rate"] = 1e-5
        extend_identity(identity(), new, 34000)
    with pytest.raises(ValueError, match="manifests"):
        extend_identity(identity(), identity(manifests={"train.jsonl": "other"}), 34000)
