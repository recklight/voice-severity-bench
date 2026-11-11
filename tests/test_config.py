"""Configuration: defaults, YAML round trips, shipped files and validation messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_severity_bench import Config, ConfigError, FrontendConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"


def test_defaults_describe_the_39_dimensional_front_end():
    config = Config()
    assert config.frontend.n_static == 13 and config.frontend.n_dims == 39
    assert config.data.task == "regression" and config.model.name == "ridge"
    assert config.evaluation.n_splits == 5 and config.output.figure_format == "png"


def test_default_yaml_equals_the_dataclass_defaults():
    assert Config.from_yaml(CONFIGS / "default.yaml") == Config()


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_configuration_loads(path):
    assert isinstance(Config.from_yaml(path), Config)


def test_quick_demo_points_at_the_synthetic_dataset():
    config = Config.from_yaml(CONFIGS / "quick_demo.yaml")
    assert config.data.dataset == "data/synthetic/dataset.csv"


def test_yaml_round_trip_keeps_every_value(tmp_path):
    config = Config().override(
        {
            "frontend.delta_order": 1,
            "pooling.statistics": ["mean", "percentile"],
            "ablation.statistics": [["mean"], ["mean", "std"]],
            "ablation.frame_hop_ms": [[25, 10], [30, 15]],
            "model.hidden_units": [32, 16],
        }
    )
    path = config.to_yaml(tmp_path / "config.yaml")
    assert Config.from_yaml(path) == config
    assert b"\r\n" not in path.read_bytes()
    assert config.ablation.frame_hop_ms == ((25.0, 10.0), (30.0, 15.0))


def test_unknown_keys_are_rejected_with_the_valid_ones():
    with pytest.raises(ConfigError, match="valid keys are") as caught:
        Config.from_dict({"frontend": {"n_cepz": 12}})
    assert "n_ceps" in str(caught.value)
    with pytest.raises(ConfigError, match="valid top-level keys"):
        Config.from_dict({"frontnd": {}})
    with pytest.raises(ConfigError, match="unknown configuration key"):
        Config().override({"frontend.n_cepz": 3})


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("data.task", "clustering", "data.task"),
        ("frontend.n_ceps", 23, "frontend.n_ceps"),
        ("frontend.window", "kaiser", "frontend.window"),
        ("frontend.delta_order", 3, "frontend.delta_order"),
        ("frontend.delta_widths", [4, 5], "frontend.delta_widths"),
        ("frontend.preemphasis", 1.0, "frontend.preemphasis"),
        ("pooling.statistics", ["mean", "median"], "pooling.statistics"),
        ("pooling.statistics", ["mean", "mean"], "pooling.statistics"),
        ("pooling.normalization", "cmvn", "pooling.normalization"),
        ("analysis.p_enter", 0.2, "analysis.p_enter"),
        ("analysis.correction", "bonferroni", "analysis.correction"),
        ("model.name", "logistic", "data.task"),
        ("model.reducer", "lda", "lda"),
        ("model.hidden_units", [0], "model.hidden_units"),
        ("model.tune", "yes", "model.tune"),
        ("model.impute", "mean", "model.impute"),
        ("model.block_selection", True, "model.tune"),
        ("model.ridge_alphas", [1.0, 1.0], "model.ridge_alphas"),
        ("model.ridge_alphas", [0.0, 1.0], "model.ridge_alphas"),
        ("audio.rms_dbfs", 0.0, "audio.rms_dbfs"),
        ("evaluation.n_repeats", 0, "evaluation.n_repeats"),
        ("analysis.n_resamples", 50, "analysis.n_resamples"),
        ("ablation.frame_hop_ms", [[25]], "ablation.frame_hop_ms"),
        ("evaluation.n_splits", 1, "evaluation.n_splits"),
        ("synthetic.f0_range_hz", [200, 100], "synthetic.f0_range_hz"),
        ("output.figure_format", "bmp", "output.figure_format"),
        ("output.dpi", True, "output.dpi"),
        ("output.dpi", 150.5, "output.dpi"),
        ("seed", -1, "seed"),
    ],
)
def test_invalid_values_name_their_dotted_key(key, value, fragment):
    with pytest.raises(ConfigError, match=fragment):
        Config().override({key: value})


def test_an_ablation_grid_for_another_setup_does_not_block_other_commands():
    # The default grid compares up to 18 cepstra with ridge and SVR regressors.
    fewer_filters = Config().override({"frontend.n_mels": 16})
    classification = Config().override({"data.task": "classification", "model.name": "logistic"})
    assert fewer_filters.frontend.n_mels == 16
    assert classification.model.name == "logistic"
    assert "ablation.n_ceps" in fewer_filters.ablation_problems()[0]
    assert any("ablation.models" in problem for problem in classification.ablation_problems())
    assert Config().ablation_problems() == []


def test_torch_model_requires_no_external_reducer():
    with pytest.raises(ConfigError, match="torch_mlp"):
        Config().override({"model.name": "torch_mlp", "model.reducer": "pca"})


def test_a_single_value_is_accepted_for_a_list_setting():
    config = Config().override({"ablation.models": "ridge", "pooling.statistics": "mean"})
    assert config.ablation.models == ("ridge",) and config.pooling.statistics == ("mean",)


def test_frontend_dimension_properties():
    assert FrontendConfig(energy_term="none", delta_order=0).n_dims == 12
    assert FrontendConfig(n_ceps=19, n_mels=26, delta_order=1).n_dims == 40


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("data: [", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Config.from_yaml(broken)
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        Config.from_yaml(listing)
