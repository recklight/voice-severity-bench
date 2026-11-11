"""Feature files compared on shared folds: alignment, block choice, contrasts and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from voice_severity_bench import (
    Config,
    ConfigError,
    DatasetError,
    align_feature_sets,
    classification_metrics,
    compare_feature_sets,
    feature_set_from_table,
    pairwise_contrasts,
    repeat_averaged_predictions,
    save_features,
)


@pytest.fixture
def table(rng):
    n = 60
    frame = pd.DataFrame(
        {
            "path": [f"rec{i:02d}.wav" for i in range(n)],
            "group": np.repeat([f"s{i}" for i in range(30)], 2),
            "severity": rng.uniform(0, 100, n),
        }
    )
    frame["good"] = frame["severity"] / 10 + rng.normal(0, 1, n)
    frame["poor"] = frame["severity"] / 10 + rng.normal(0, 4, n)
    for layer in range(3):
        frame[f"l{layer}_a"] = rng.normal(0, 1, n)
        frame[f"l{layer}_b"] = rng.normal(0, 1, n)
    frame["l1_a"] = frame["severity"] / 10 + rng.normal(0, 0.5, n)
    return frame


@pytest.fixture
def config():
    return Config().override(
        {
            "model.tune": True,
            "model.ridge_alphas": [0.1, 1.0, 10.0],
            "evaluation.n_repeats": 3,
            "analysis.n_resamples": 300,
        }
    )


def _single(table, column):
    return feature_set_from_table(table, target="severity", feature_columns=[column])


def test_every_file_is_scored_on_the_same_folds(table, config):
    sets = {"poor": _single(table, "poor"), "good": _single(table, "good")}
    result = compare_feature_sets(sets, config)
    per_file = {
        name: part.sort_values(["repeat", "item"])[["repeat", "fold", "item"]].to_numpy()
        for name, part in result.predictions.groupby("feature_set")
    }
    assert np.array_equal(per_file["poor"], per_file["good"])
    assert result.summary["feature_set"].tolist() == ["good", "poor"]
    assert result.prediction_matrix("good").shape == (60, 3)
    # No speaker on both sides of a fold.
    items = result.items.set_index("item")["group"]
    for (_, _), part in result.predictions.groupby(["feature_set", "repeat"]):
        assert part.groupby(items.loc[part["item"]].to_numpy())["fold"].nunique().max() == 1
    good = result.summary.set_index("feature_set").loc["good"]
    assert good["rmse_low"] < good["rmse"] < good["rmse_high"]


def test_the_pooled_estimate_is_the_mean_over_repetitions(table, config):
    result = compare_feature_sets({"good": _single(table, "good")}, config)
    matrix = result.prediction_matrix("good")
    expected = np.mean([np.sqrt(np.mean((column - result.y_true) ** 2)) for column in matrix.T])
    assert result.summary.loc[0, "rmse"] == pytest.approx(expected)
    averaged = repeat_averaged_predictions(result)
    assert np.allclose(averaged["good"], matrix.mean(axis=1))


def test_the_layer_is_chosen_inside_the_training_folds(table, config):
    blocks = {f"layer{i}": [f"l{i}_a", f"l{i}_b"] for i in range(3)}
    layered = feature_set_from_table(table, target="severity", blocks=blocks, kind="ssl")
    result = compare_feature_sets(
        {"layers": layered}, config.override({"model.block_selection": True})
    )
    chosen = {json.loads(text)["select__block"] for text in result.folds["params"]}
    assert chosen == {"layer1"}
    assert set(result.folds["n_features"]) == {2}


def test_contrasts_are_one_family_with_paired_intervals(table, config):
    sets = {name: _single(table, name) for name in ("poor", "good")}
    sets["copy"] = _single(table, "good")
    result = compare_feature_sets(sets, config)
    contrasts = pairwise_contrasts(
        result, [("good", "poor"), ("copy", "good")], n_resamples=500, equivalence_margin=0.5
    )
    good, copy = contrasts.iloc[0], contrasts.iloc[1]
    assert good["better"] and good["ci_high"] < 0 and not good["equivalent"]
    assert copy["difference"] == 0.0 and copy["equivalent"]
    assert good["p_corrected_t_adjusted"] >= good["p_corrected_t"]
    assert good["n_pairs"] == 15


def test_files_are_aligned_by_path_and_interval(table):
    a = _single(table, "good")
    b = _single(table.iloc[::-1].reset_index(drop=True), "poor")
    aligned = align_feature_sets({"a": a, "b": b})
    assert aligned["b"].paths.tolist() == a.paths.tolist()
    with pytest.raises(DatasetError, match="do not hold the same items"):
        align_feature_sets({"a": a, "b": b.take(np.arange(59))})
    shifted = table.assign(severity=table["severity"] + 1.0)
    with pytest.raises(DatasetError, match="different targets"):
        align_feature_sets({"a": a, "b": _single(shifted, "poor")})
    twice = _single(pd.concat([table, table.iloc[:1]], ignore_index=True), "poor")
    with pytest.raises(DatasetError, match=r"lists items twice: rec00\.wav"):
        align_feature_sets({"a": a, "b": twice})
    parts = table.assign(start_s=0.0, end_s=1.0)
    with pytest.raises(DatasetError, match="do not hold the same items"):
        align_feature_sets({"a": a, "b": _single(parts, "poor")})


def test_unknown_names_are_refused(table, config):
    sets = {"good": _single(table, "good")}
    result = compare_feature_sets(sets, config)
    with pytest.raises(ConfigError, match="unknown feature file"):
        pairwise_contrasts(result, [("good", "missing")])


def test_compare_command_writes_its_tables(tmp_path, table):
    from typer.testing import CliRunner

    from voice_severity_bench.cli import app

    paths = []
    for name in ("poor", "good"):
        paths.append(f"{name}={save_features(tmp_path / f'{name}.npz', _single(table, name))}")
    result = CliRunner().invoke(
        app,
        [
            "compare",
            *paths,
            "--set",
            "evaluation.n_repeats=2",
            "--set",
            "analysis.n_resamples=200",
            "--margin",
            "1.0",
            "--out",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    for name in ("folds", "items", "predictions", "summary", "contrasts"):
        assert (tmp_path / "run" / f"comparison_{name}.csv").is_file(), name
    contrasts = pd.read_csv(tmp_path / "run" / "comparison_contrasts.csv")
    assert contrasts[["first", "second"]].values.tolist() == [["good", "poor"]]
    assert "equivalent" in contrasts.columns
    assert (tmp_path / "run" / "comparison_metric_by_feature_file.png").is_file()
    bad = CliRunner().invoke(app, ["compare", "no-equals-sign.npz", "--out", str(tmp_path / "x")])
    assert bad.exit_code == 2 and "NAME=PATH" in bad.output


def test_regression_tables_keep_their_columns(table, config):
    result = compare_feature_sets(
        {"poor": _single(table, "poor"), "good": _single(table, "good")}, config
    )
    assert result.predictions.columns.tolist() == [
        "feature_set",
        "repeat",
        "fold",
        "item",
        "y_pred",
    ]
    assert "rmse_n_undefined" not in result.summary.columns
    assert "n_undefined" not in pairwise_contrasts(result, n_resamples=100).columns
    assert set(result.as_dict()) == {
        "metric", "n_items", "n_repeats", "feature_sets", "best", "best_value",
    }  # fmt: skip


def test_regression_refuses_classification_settings(table, config):
    sets = {"good": _single(table, "good")}
    with pytest.raises(ConfigError, match="not available"):
        compare_feature_sets(sets, config, metric="uar")
    with pytest.raises(ConfigError, match="strata"):
        compare_feature_sets(sets, config, strata=["a"] * 60)
    with pytest.raises(ConfigError, match="class scores"):
        compare_feature_sets(sets, config).score_matrix("good")


@pytest.fixture
def speakers(rng):
    n_speakers = 40
    frame = pd.DataFrame(
        {
            "path": [f"rec{i:03d}.wav" for i in range(2 * n_speakers)],
            "group": np.repeat([f"s{i}" for i in range(n_speakers)], 2),
            "label": np.repeat(np.tile(["dysphonic", "healthy"], n_speakers // 2), 2),
        }
    )
    sick = (frame["label"] == "dysphonic").to_numpy()
    frame["good"] = sick * 2.0 + rng.normal(0, 1, 2 * n_speakers)
    frame["poor"] = sick * 0.3 + rng.normal(0, 1, 2 * n_speakers)
    for layer in range(3):
        frame[f"l{layer}_a"] = rng.normal(0, 1, 2 * n_speakers)
    frame["l1_a"] = sick * 3.0 + rng.normal(0, 0.5, 2 * n_speakers)
    return frame


@pytest.fixture
def classifier():
    return Config().override(
        {
            "data.task": "classification",
            "data.target": "label",
            "model.name": "logistic",
            "model.tune": True,
            "evaluation.n_repeats": 3,
            "analysis.n_resamples": 300,
        }
    )


def _labelled(frame, column):
    return feature_set_from_table(
        frame, target="label", feature_columns=[column], task="classification"
    )


def test_each_repetition_keeps_labels_and_class_scores(speakers, classifier):
    sets = {"poor": _labelled(speakers, "poor"), "good": _labelled(speakers, "good")}
    result = compare_feature_sets(sets, classifier)
    assert result.task == "classification" and result.metric == "uar"
    assert result.classes == ("dysphonic", "healthy")
    assert result.positive_class == "dysphonic"
    assert result.predictions.columns.tolist() == [
        "feature_set", "repeat", "fold", "item", "y_pred", "score_dysphonic", "score_healthy",
    ]  # fmt: skip
    assert result.predictions.shape[0] == 2 * 3 * 80
    labels = result.prediction_matrix("good")
    scores = result.score_matrix("good")
    assert labels.shape == scores.shape == (80, 3)
    assert set(np.unique(labels)) <= {"dysphonic", "healthy"}
    assert result.score_matrix("good", "healthy") == pytest.approx(1.0 - scores)
    assert set(result.y_true.tolist()) == {"dysphonic", "healthy"}


def test_classification_folds_are_shared_grouped_and_stratified(speakers, classifier):
    sets = {"poor": _labelled(speakers, "poor"), "good": _labelled(speakers, "good")}
    result = compare_feature_sets(sets, classifier)
    per_file = {
        name: part.sort_values(["repeat", "item"])[["repeat", "fold", "item"]].to_numpy()
        for name, part in result.predictions.groupby("feature_set")
    }
    assert np.array_equal(per_file["poor"], per_file["good"])
    items = result.items.set_index("item")
    for (_, _), part in result.predictions.groupby(["feature_set", "repeat"]):
        assert (
            part.groupby(items.loc[part["item"], "group"].to_numpy())["fold"].nunique().max() == 1
        )
        balance = pd.crosstab(part["fold"], items.loc[part["item"], "y_true"].to_numpy())
        assert (balance > 0).all().all()


def test_the_summary_reports_what_the_paper_needs(speakers, classifier):
    sets = {"poor": _labelled(speakers, "poor"), "good": _labelled(speakers, "good")}
    result = compare_feature_sets(sets, classifier)
    summary = result.summary.set_index("feature_set")
    assert result.summary["feature_set"].tolist() == ["good", "poor"]
    expected = ["uar", "accuracy", "f1_macro", "sensitivity", "specificity", "roc_auc"]
    expected += ["uar_low", "uar_high", "uar_n_undefined", "uar_fold_mean", "uar_fold_std"]
    for column in expected:
        assert column in summary.columns, column
    good = summary.loc["good"]
    assert good["uar_low"] < good["uar"] < good["uar_high"]
    assert good["uar"] > summary.loc["poor", "uar"]
    assert good["uar_n_undefined"] == 0


def test_the_pooled_classification_estimate_is_the_mean_over_repetitions(speakers, classifier):
    result = compare_feature_sets({"good": _labelled(speakers, "good")}, classifier)
    labels = result.prediction_matrix("good")
    positive = result.score_matrix("good")
    per_repeat = [
        classification_metrics(
            result.y_true,
            labels[:, repeat],
            np.column_stack([positive[:, repeat], 1.0 - positive[:, repeat]]),
            classes=result.classes,
            positive_class=result.positive_class,
        )
        for repeat in range(labels.shape[1])
    ]
    row = result.summary.iloc[0]
    for key in ("uar", "sensitivity", "specificity", "roc_auc"):
        assert row[key] == pytest.approx(np.mean([scores[key] for scores in per_repeat]))


def test_the_positive_class_comes_from_the_configuration(speakers, classifier):
    sets = {"good": _labelled(speakers, "good")}
    first = compare_feature_sets(sets, classifier).summary.iloc[0]
    swapped = compare_feature_sets(
        sets, classifier.override({"evaluation.positive_class": "healthy"})
    ).summary.iloc[0]
    assert swapped["sensitivity"] == pytest.approx(first["specificity"])
    assert swapped["specificity"] == pytest.approx(first["sensitivity"])
    assert swapped["roc_auc"] == pytest.approx(first["roc_auc"])
    with pytest.raises(ConfigError, match="positive_class"):
        compare_feature_sets(sets, classifier.override({"evaluation.positive_class": "sick"}))


@pytest.mark.parametrize("metric", ["accuracy", "sensitivity", "specificity", "roc_auc"])
def test_any_classification_metric_can_rank_and_be_contrasted(speakers, classifier, metric):
    sets = {"poor": _labelled(speakers, "poor"), "good": _labelled(speakers, "good")}
    result = compare_feature_sets(sets, classifier, metric=metric)
    assert result.metric == metric
    assert f"{metric}_low" in result.summary.columns
    contrast = pairwise_contrasts(result, [("good", "poor")], n_resamples=300).iloc[0]
    assert contrast["difference"] == pytest.approx(
        result.summary.set_index("feature_set").loc["good", metric]
        - result.summary.set_index("feature_set").loc["poor", metric]
    )


def test_contrasts_of_classifiers_favour_the_higher_score(speakers, classifier):
    sets = {"poor": _labelled(speakers, "poor"), "good": _labelled(speakers, "good")}
    result = compare_feature_sets(sets, classifier)
    table = pairwise_contrasts(
        result, [("good", "poor")], n_resamples=500, equivalence_margin=0.01
    ).iloc[0]
    assert table["better"] and table["ci_low"] > 0
    assert table["n_undefined"] == 0 and not table["equivalent"]
    other = pairwise_contrasts(result, [("good", "poor")], metric="roc_auc", n_resamples=200)
    assert other["better"].iloc[0]
    with pytest.raises(ConfigError, match="not available"):
        pairwise_contrasts(result, metric="rmse")


def test_three_classes_have_no_one_against_the_rest_metrics(speakers, classifier):
    frame = speakers.copy()
    frame.loc[frame["group"].isin([f"s{i}" for i in range(0, 40, 3)]), "label"] = "other"
    sets = {"good": _labelled(frame, "good")}
    result = compare_feature_sets(sets, classifier)
    assert result.positive_class is None
    assert "sensitivity" not in result.summary.columns
    assert result.as_dict()["classes"] == ["dysphonic", "healthy", "other"]
    with pytest.raises(ConfigError, match="two classes"):
        compare_feature_sets(sets, classifier, metric="roc_auc")
    with pytest.raises(ConfigError, match="class scores"):
        result.score_matrix("good")
    assert result.score_matrix("good", "other").shape == (80, 3)
    with pytest.raises(ConfigError, match="unknown class"):
        result.score_matrix("good", "missing")


def test_strata_reach_the_folds(speakers, classifier):
    sets = {"good": _labelled(speakers, "good")}
    strata = np.repeat(np.repeat(["a", "b"], 2), 2)
    strata = np.tile(strata, 10)
    result = compare_feature_sets(sets, classifier, strata=strata)
    part = result.predictions[result.predictions["repeat"] == 0]
    crossed = pd.crosstab(part["fold"], [result.y_true[part["item"]], strata[part["item"]]])
    assert crossed.to_numpy().min() >= 1
    with pytest.raises(DatasetError, match="merge strata"):
        lonely = strata.copy()
        lonely[:2] = "c"
        compare_feature_sets(sets, classifier, strata=lonely)


def test_the_layer_is_chosen_inside_the_training_folds_for_classifiers_too(speakers, classifier):
    columns = [f"l{k}_a" for k in range(3)]
    blocks = {f"l{k}": [f"l{k}_a"] for k in range(3)}
    features = feature_set_from_table(
        speakers, target="label", feature_columns=columns, blocks=blocks, task="classification"
    )
    config = classifier.override({"model.block_selection": True})
    result = compare_feature_sets({"layers": features}, config)
    chosen = [json.loads(params)["select__block"] for params in result.folds["params"]]
    assert chosen.count("l1") / len(chosen) > 0.9
    assert result.summary["uar"].iloc[0] > 0.8


def test_balanced_weights_work_through_compare(speakers, classifier):
    keep = np.concatenate(
        [
            np.flatnonzero(speakers["label"] == "healthy"),
            np.flatnonzero(speakers["label"] != "healthy")[:16],
        ]
    )
    frame = speakers.iloc[keep].reset_index(drop=True)
    sets = {"poor": _labelled(frame, "poor")}
    plain = compare_feature_sets(sets, classifier.override({"model.name": "logistic"}))
    weighted = compare_feature_sets(
        sets, classifier.override({"model.name": "logistic", "model.class_weight": "balanced"})
    )
    assert weighted.summary["sensitivity"].iloc[0] > plain.summary["sensitivity"].iloc[0]


def test_averaged_classifier_predictions_are_positive_class_scores(speakers, classifier):
    result = compare_feature_sets({"good": _labelled(speakers, "good")}, classifier)
    averaged = repeat_averaged_predictions(result)
    assert averaged["good"].to_numpy() == pytest.approx(result.score_matrix("good").mean(axis=1))


def test_results_do_not_depend_on_the_number_of_jobs(speakers, classifier):
    sets = {"good": _labelled(speakers, "good")}
    one = compare_feature_sets(sets, classifier, n_jobs=1)
    two = compare_feature_sets(sets, classifier, n_jobs=2)
    pd.testing.assert_frame_equal(one.predictions, two.predictions)
    pd.testing.assert_frame_equal(one.summary, two.summary)


def test_compare_command_handles_classifiers_and_the_metric_option(tmp_path, speakers, table):
    from typer.testing import CliRunner

    from voice_severity_bench.cli import app

    paths = [
        f"{name}={save_features(tmp_path / f'{name}.npz', _labelled(speakers, name))}"
        for name in ("poor", "good")
    ]
    settings = ["--set", "data.task=classification", "--set", "data.target=label",
                "--set", "model.name=logistic", "--set", "evaluation.n_repeats=2",
                "--set", "analysis.n_resamples=200"]  # fmt: skip
    result = CliRunner().invoke(
        app, ["compare", *paths, *settings, "--metric", "roc_auc", "--out", str(tmp_path / "run")]
    )
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(tmp_path / "run" / "comparison_summary.csv")
    assert {"roc_auc_low", "roc_auc_high", "roc_auc_n_undefined"} <= set(summary.columns)
    predictions = pd.read_csv(tmp_path / "run" / "comparison_predictions.csv")
    assert "score_dysphonic" in predictions.columns
    report = json.loads((tmp_path / "run" / "comparison.json").read_text(encoding="utf-8"))
    assert report["metric"] == "roc_auc" and report["positive_class"] == "dysphonic"

    default = CliRunner().invoke(
        app, ["compare", *paths, *settings, "--no-figures", "--out", str(tmp_path / "uar")]
    )
    assert default.exit_code == 0, default.output
    report = json.loads((tmp_path / "uar" / "comparison.json").read_text(encoding="utf-8"))
    assert report["metric"] == "uar"

    wrong = CliRunner().invoke(
        app, ["compare", *paths, *settings, "--metric", "rmse", "--out", str(tmp_path / "bad")]
    )
    assert wrong.exit_code != 0 and "not available" in wrong.output

    regression = f"good={save_features(tmp_path / 'r.npz', _single(table, 'good'))}"
    ranked = CliRunner().invoke(
        app,
        [
            "compare",
            regression,
            f"poor={save_features(tmp_path / 'p.npz', _single(table, 'poor'))}",
            "--set",
            "evaluation.n_repeats=2",
            "--set",
            "analysis.n_resamples=200",
            "--metric",
            "r2",
            "--no-figures",
            "--out",
            str(tmp_path / "r2"),
        ],
    )
    assert ranked.exit_code == 0, ranked.output
    report = json.loads((tmp_path / "r2" / "comparison.json").read_text(encoding="utf-8"))
    assert report["metric"] == "r2"
