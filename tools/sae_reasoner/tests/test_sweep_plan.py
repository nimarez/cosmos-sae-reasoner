import argparse

from tools.sae_reasoner.scripts.plan_training_sweep import build_configs, render_script


def args(**overrides):
    data = {
        "activation_dir": "s3://bucket/acts",
        "output_root": __import__("pathlib").Path("outputs/sweep"),
        "run_prefix": "test",
        "stage": "lr_batch",
        "steps": 2000,
        "log_every": 50,
        "analysis_batch_size": 4096,
        "python": ".venv/bin/python",
        "best_batch_size": 1024,
        "best_lr": 3e-4,
        "best_expansion_factor": 8,
        "best_top_k": 32,
        "wandb_project": "proj",
        "wandb_tags": "sae,sweep",
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_lr_batch_stage_uses_8x_baseline_and_9_runs():
    configs = list(build_configs(args(stage="lr_batch")))

    assert len(configs) == 9
    assert {config.expansion_factor for config in configs} == {8}
    assert {config.top_k for config in configs} == {32}
    assert {config.batch_size for config in configs} == {512, 1024, 2048}
    assert {config.lr for config in configs} == {1e-4, 3e-4, 1e-3}


def test_capacity_stage_uses_dataset_sized_expansion_grid():
    configs = list(build_configs(args(stage="capacity", best_batch_size=2048, best_lr=1e-4)))

    assert [config.expansion_factor for config in configs] == [4, 8, 16]
    assert all(config.top_k == 32 for config in configs)
    assert all(config.batch_size == 2048 for config in configs)
    assert all(config.lr == 1e-4 for config in configs)


def test_rendered_script_runs_analysis_after_each_train():
    configs = list(build_configs(args(stage="topk")))
    script = render_script(configs[:1], args(stage="topk"))

    assert "train-sae" in script
    assert "analyze-sae" in script
    assert ".venv/bin/python -m tools.sae_reasoner train-sae" in script
    assert script.index("train-sae") < script.index("analyze-sae")
