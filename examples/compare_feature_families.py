"""MFCCs against other front ends, on the same folds, next to the rater ceiling.

    python examples/compare_feature_families.py --out examples/output/compare_feature_families

1. Write a synthetic corpus and give every recording three simulated ratings;
   the target is their mean, as in a clinical corpus.
2. Build four feature files of the same recordings: MFCC-13 and MFCC-39 from
   ``prepare``, two periodicity measures that are undefined (NaN) for a
   recording without a clear period, and log band energies split into four
   blocks of two bands.
3. Score all four on the same repeated, speaker-grouped folds. Imputation, the
   ridge penalty and, for the band file, the block are chosen inside each
   training fold.
4. Test every file against MFCC-13 (corrected t test, Wilcoxon, Holm, and a
   speaker-level bootstrap of the RMSE difference) and draw the RMSEs next to
   the rating noise floor and the error of a single simulated rater.

The synthetic vowels say nothing about which front end suits real voices; the
script shows the mechanics on data where every step can be checked.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from voice_severity_bench import (
    Config,
    compare_feature_sets,
    comparison_table,
    configure_logging,
    feature_set_from_table,
    icc_oneway,
    leave_one_rater_out,
    load_dataset_index,
    pairwise_contrasts,
    plot_comparison,
    prepare_features,
    rating_noise_floor,
    read_audio,
    regression_metrics,
    save_figure,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_feature_families")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "compare_demo.yaml"
#: Spread of a simulated rater around the synthetic severity (points).
RATER_SD = 8.0
#: Edges of the eight bands of the band-energy file (Hz).
BAND_EDGES = (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 8000)


def periodicity(signal: np.ndarray, rate: int) -> tuple[float, float]:
    """Peak of the normalized autocorrelation in 70-400 Hz, and its lag in ms.

    Both are NaN when the peak is below 0.3, the way a real jitter or HNR
    measurement is undefined for a voice without a clear period.
    """
    frame = signal - signal.mean()
    spectrum = np.fft.rfft(frame, 2 * frame.size)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2)[: frame.size]
    autocorrelation /= autocorrelation[0]
    lags = np.arange(int(rate / 400), int(rate / 70))
    best = lags[np.argmax(autocorrelation[lags])]
    peak = float(autocorrelation[best])
    if peak < 0.3:
        return np.nan, np.nan
    return peak, 1000.0 * best / rate


def band_energies(signal: np.ndarray, rate: int) -> np.ndarray:
    """Mean log power in each band of ``BAND_EDGES`` (dB)."""
    power = np.abs(np.fft.rfft(signal)) ** 2
    frequency = np.fft.rfftfreq(signal.size, 1.0 / rate)
    return np.array(
        [10 * np.log10(power[(frequency >= low) & (frequency < high)].mean() + 1e-12)
         for low, high in itertools.pairwise(BAND_EDGES)]
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_feature_families"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="8 speakers x 3 recordings of 0.5 s, 4 folds x 2 repeats, 300 bootstrap resamples.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(CONFIG).override({"seed": args.seed})
    if args.quick:
        config = config.override(
            {
                "synthetic.n_speakers": 8,
                "synthetic.duration_s": 0.5,
                "evaluation.n_splits": 4,
                "evaluation.n_inner_splits": 3,
                "evaluation.n_repeats": 2,
                "analysis.n_resamples": 300,
            }
        )
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    dataset = write_synthetic_dataset(out / "data", config.synthetic, rng=config.seed)
    index = load_dataset_index(dataset, target="severity")
    rng = np.random.default_rng(config.seed)
    ratings = index["severity"].to_numpy()[:, None] + rng.normal(0, RATER_SD, (len(index), 3))
    target = ratings.mean(axis=1)

    files = {}
    for name, order in (("mfcc13", 0), ("mfcc39", 2)):
        prepared = prepare_features(dataset, config.override({"frontend.delta_order": order}))
        files[name] = dataclasses.replace(prepared, y=target)
    rows = []
    for path in index["path"]:
        signal, rate = read_audio(path, config.audio)
        rows.append([*periodicity(signal, rate), *band_energies(signal, rate)])
    bands = [f"band{i}" for i in range(len(BAND_EDGES) - 1)]
    table = pd.DataFrame(rows, columns=["acf_peak", "period_ms", *bands])
    table.insert(0, "path", index["path"])
    table.insert(1, "group", index["group"])
    table.insert(2, "rating", target)
    files["periodicity"] = feature_set_from_table(
        table, target="rating", feature_columns=["acf_peak", "period_ms"], kind="voice_measures"
    )
    files["bands"] = feature_set_from_table(
        table,
        target="rating",
        blocks={f"bands{i}{i + 1}": bands[i : i + 2] for i in range(0, len(bands), 2)},
        kind="band_energy",
    )
    LOGGER.info("%d recordings without a periodicity peak", int(table["acf_peak"].isna().sum()))

    result = compare_feature_sets(files, config)
    contrasts = pairwise_contrasts(
        result, n_resamples=config.analysis.n_resamples, seed=config.seed, equivalence_margin=1.0
    )
    floor = rating_noise_floor(ratings)
    pairs = leave_one_rater_out(ratings)
    one_rater = regression_metrics(pairs["reference"], pairs["rating"])["rmse"]
    references = {"rating noise floor": floor, "one rater": one_rater}
    for name, frame in (
        ("comparison_folds", result.folds),
        ("comparison_summary", result.summary),
        ("comparison_contrasts", contrasts),
        ("comparison_metric_by_feature_file", comparison_table(result.summary, result.metric)),
    ):
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    save_figure(
        plot_comparison(result.summary, result.metric, references=references),
        out / "comparison_metric_by_feature_file",
    )
    icc = icc_oneway(ratings)
    write_json(
        out / "comparison.json",
        result.as_dict() | {"icc1": icc.icc1, "icck": icc.icck} | references,
    )

    LOGGER.info("ICC(1,1) %.3f, ICC(1,3) %.3f; noise floor %.2f, one rater %.2f",
                icc.icc1, icc.icck, floor, one_rater)  # fmt: skip
    for row in result.summary.itertuples():
        LOGGER.info("#%d %-12s rmse %.2f [%.2f, %.2f]", row.rank, row.feature_set,
                    row.rmse, row.rmse_low, row.rmse_high)  # fmt: skip
    for row in contrasts.itertuples():
        LOGGER.info("%s - %s: %+.2f [%+.2f, %+.2f], Holm p %.3f", row.first, row.second,
                    row.difference, row.ci_low, row.ci_high,
                    row.p_corrected_t_adjusted)  # fmt: skip
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
