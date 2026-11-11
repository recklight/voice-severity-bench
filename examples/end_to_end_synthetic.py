"""End to end on synthetic data: synthesize, prepare, cross-validate, train, predict.

    python examples/end_to_end_synthetic.py --out examples/output/end_to_end_synthetic

1. Write a synthetic dataset of sustained vowels whose jitter, shimmer and
   noise grow with a 0-100 severity rating, in the same on-disk layout as real
   data (WAV files plus ``dataset.csv``).
2. Pool the 39-dimensional MFCC frames of each recording into one vector
   (mean and standard deviation, 78 values).
3. Cross-validate ridge regression with speaker-grouped folds.
4. Train on every recording and save the model.
5. Reload the model and predict two recordings from their WAV files.

The recordings are synthetic, so the scores check the code, not how well the
model would do on real voices.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from voice_severity_bench import (
    Config,
    configure_logging,
    cross_validate,
    load_model,
    predict_inputs,
    prepare_features,
    save_features,
    save_model,
    train_model,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("end_to_end_synthetic")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/end_to_end_synthetic"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use configs/quick_demo.yaml: 8 speakers x 3 recordings of 0.5 s, 4 folds.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")
    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    out: Path = args.out

    # 1. Synthetic recordings and their index.
    index = write_synthetic_dataset(out / "data", config.synthetic, rng=config.seed)
    # 2. One pooled feature vector per recording.
    features = prepare_features(index, config)
    save_features(out / "features.npz", features)
    # 3. Speaker-grouped cross-validation.
    result = cross_validate(features, config)
    write_json(out / "metrics.json", result.metrics)
    result.predictions.to_csv(out / "predictions.csv", index=False, lineterminator="\n")
    summary = result.metrics["summary"]
    for name in ("rmse", "mae", "ccc"):
        LOGGER.info("%s %.3f +/- %.3f", name, summary[name]["mean"], summary[name]["std"])
    # 4. Train on everything and save.
    model = train_model(features, config)
    model_path = save_model(out / "model.joblib", model, config)
    # 5. Reload and predict two recordings.
    table = predict_inputs([features.paths[0], features.paths[1]], load_model(model_path))
    table.to_csv(out / "new_predictions.csv", index=False, lineterminator="\n")
    for row, truth in zip(table.itertuples(), features.y[:2], strict=True):
        LOGGER.info("%s: predicted %.1f, rated %.1f", Path(row.path).name, row.predicted, truth)
    LOGGER.info("written to %s", out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
