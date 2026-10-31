import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ugr16ad.synthetic import SANITY_GRID

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep.py"
_spec = importlib.util.spec_from_file_location("sweep", SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep"] = sweep  # dataclasses looks its module up here
_spec.loader.exec_module(sweep)

LSTM = sweep.SPECS["lstm"]


def write_config(root, config, scores, n_parameters, model="lstm"):
    spec = sweep.SPECS[model]
    for seed, score in enumerate(scores):
        meta = {
            **{k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
            "seed": seed,
            "n_parameters": n_parameters,
            "best_epoch": 3,
            "best_tune_loss": 0.7,
            # The fake selection_score below reads the score back from here.
            "tune_sanity": [
                {"magnitude": m, "n_features": f, "auc": 0.5, "ap": score, "recall": 0.1}
                for m, f in SANITY_GRID
            ],
        }
        path = root / model / sweep.tag(config, spec) / spec.folder / f"seed{seed}" / "meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(meta))


@pytest.fixture
def fake_score(monkeypatch):
    monkeypatch.setattr(sweep, "selection_score", lambda meta: meta["tune_sanity"][0]["ap"])


def test_tie_goes_to_the_smaller_model(tmp_path, fake_score):
    big = {**LSTM.base, "hidden_dim": 128}
    small = {**LSTM.base, "hidden_dim": 32}
    far = {**LSTM.base, "hidden_dim": 64}
    write_config(tmp_path, big, [0.50, 0.52, 0.54], 200_000)
    write_config(tmp_path, small, [0.49, 0.51, 0.53], 30_000)  # within seed noise
    write_config(tmp_path, far, [0.20, 0.21, 0.22], 80_000)
    summary = sweep.summarise(sweep.load_runs("lstm", root=tmp_path), "lstm")
    assert summary.index[0] == sweep.tag(big, LSTM)  # ranked by mean
    assert summary.loc[sweep.tag(small, LSTM), "pick"]
    assert summary.pick.sum() == 1
    assert not summary.loc[sweep.tag(far, LSTM), "tied"]


def test_unfinished_configs_are_not_ranked(tmp_path, fake_score):
    write_config(tmp_path, LSTM.base, [0.5, 0.5, 0.5], 1)
    write_config(tmp_path, {**LSTM.base, "latent_dim": 4}, [0.9], 1)
    summary = sweep.summarise(sweep.load_runs("lstm", root=tmp_path), "lstm")
    assert list(summary.index) == [sweep.tag(LSTM.base, LSTM)]


def test_lstm_tags_are_unchanged():
    assert sweep.tag({**LSTM.base, "hidden_dim": 32, "latent_dim": 32, "lr": 3e-3}, LSTM) == (
        "h32_z32_l1_d0.0_lr0.003"
    )
    assert sweep.tag(LSTM.base, LSTM) == "h64_z16_l1_d0.0_lr0.001"


@pytest.mark.parametrize("model", sorted(sweep.SPECS))
def test_stage_one_contains_the_baseline(model):
    configs = sweep.stage_configs(1, None, model)
    assert len(configs) == 12
    assert sweep.SPECS[model].base in configs


def test_later_stages_build_on_the_pick(tmp_path, fake_score):
    write_config(tmp_path, LSTM.base, [0.5, 0.5, 0.5], 80_000)
    write_config(tmp_path, {**LSTM.base, "latent_dim": 8}, [0.3, 0.3, 0.3], 70_000)
    summary = sweep.summarise(sweep.load_runs("lstm", root=tmp_path), "lstm")
    stage2 = sweep.stage_configs(2, summary, "lstm")
    assert [(c["num_layers"], c["dropout"], c["latent_dim"]) for c in stage2] == [
        (2, 0.0, 16),
        (2, 0.2, 16),
    ]
    stage3 = sweep.stage_configs(3, summary, "lstm")
    assert {(c["latent_dim"], c["lr"]) for c in stage3} == {
        (16, 3e-4),
        (16, 3e-3),
        (8, 3e-4),
        (8, 3e-3),
    }
    with pytest.raises(ValueError):
        sweep.stage_configs(2, None, "lstm")


def _summary_for(tmp_path, model, config):
    write_config(tmp_path, config, [0.5, 0.5, 0.5], 80_000, model=model)
    return sweep.summarise(sweep.load_runs(model, root=tmp_path), model)


def test_cnn_has_no_stage_two(tmp_path, fake_score):
    summary = _summary_for(tmp_path, "cnn", sweep.SPECS["cnn"].base)
    with pytest.raises(ValueError, match="no stage 2"):
        sweep.stage_configs(2, summary, "cnn")


def test_windowed_mlp_stage_two_varies_dropout_and_noise(tmp_path, fake_score):
    base = {**sweep.SPECS["windowed_mlp"].base, "hidden_dim": 32, "latent_dim": 8}
    summary = _summary_for(tmp_path, "windowed_mlp", base)
    stage2 = sweep.stage_configs(2, summary, "windowed_mlp")
    assert [(c["dropout"], c["noise_std"]) for c in stage2] == [(0.0, 0.0), (0.0, 0.1), (0.2, 0.0)]
    assert {(c["hidden_dim"], c["latent_dim"]) for c in stage2} == {(32, 8)}


def test_mlp_stage_two_is_shallower_and_deeper(tmp_path, fake_score):
    base = {**sweep.SPECS["mlp"].base, "hidden_dims": (128, 64)}
    summary = _summary_for(tmp_path, "mlp", base)
    stage2 = sweep.stage_configs(2, summary, "mlp")
    assert [c["hidden_dims"] for c in stage2] == [(128,), (256, 128, 64)]


def test_mlp_tag_joins_the_tuple_with_dashes():
    spec = sweep.SPECS["mlp"]
    assert sweep.tag(spec.base, spec) == "h64-32_z16_lr0.001"


def test_load_runs_reads_hidden_dims_back_as_a_tuple(tmp_path, fake_score):
    summary = _summary_for(tmp_path, "mlp", sweep.SPECS["mlp"].base)
    assert summary.hidden_dims.iloc[0] == (64, 32)
    runs = sweep.load_runs("mlp", root=tmp_path)
    assert (runs.hidden_dims == [(64, 32)] * 3).all()


def test_stage_three_has_no_duplicate_configs(tmp_path, fake_score):
    # Top 2 differ only in lr, so naive expansion repeats each lr value twice.
    write_config(tmp_path, {**LSTM.base, "lr": 3e-3}, [0.6, 0.6, 0.6], 80_000)
    write_config(tmp_path, {**LSTM.base, "lr": 3e-4}, [0.5, 0.5, 0.5], 80_000)
    summary = sweep.summarise(sweep.load_runs("lstm", root=tmp_path), "lstm")
    stage3 = sweep.stage_configs(3, summary, "lstm")
    tags = [sweep.tag(c, LSTM) for c in stage3]
    assert len(tags) == len(set(tags)) == 2


def test_launch_args_expand_hidden_dims(tmp_path):
    args = sweep.build_args(sweep.SPECS["mlp"].base, "mlp", 2, tmp_path)
    assert args[args.index("--hidden-dims") : args.index("--hidden-dims") + 3] == [
        "--hidden-dims",
        "64",
        "32",
    ]
    assert args[1] == "-u"  # unbuffered, so per-config logs fill while running
    assert "--blind" in args and args[args.index("--seed") + 1 : args.index("--seed") + 4] == [
        "0",
        "1",
        "2",
    ]


def test_summary_only_with_no_complete_config_prints_message(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, LSTM.base, [0.5], 1)
    monkeypatch.setattr(sweep, "SWEEP_DIR", tmp_path)
    real = sweep.load_runs
    monkeypatch.setattr(sweep, "load_runs", lambda model: real(model, root=tmp_path))
    monkeypatch.setattr(sys, "argv", ["sweep.py", "--model", "lstm", "--summary-only"])
    sweep.main()
    assert "no config has all 3 seeds yet" in capsys.readouterr().out
    assert not (tmp_path / "lstm" / "summary.csv").exists()
