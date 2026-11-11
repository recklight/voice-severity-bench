"""A4-A6: frozen speech representations, 13 hidden states pooled per interval (plan section 6).

    .venv/Scripts/python.exe features/extract_ssl.py --model A4 --materials a
    .venv/Scripts/python.exe features/extract_ssl.py --model A6 --materials a --limit 10

Every hidden state (layer 0 is the input of the first Transformer layer) is
reduced to mean and standard deviation over time, 768 x 2 = 1536 columns per
layer; the AVG block averages the 13 layers frame by frame first. Inputs
longer than 30 s are cut into 30 s chunks whose frames are joined before
pooling. Whisper always sees a 30 s input (its encoder takes nothing else),
so its frames are cut back to ceil(samples / 320) of the real signal.

Each item is cached under features/cache/<model>/<material>/<file_id>.npz,
so an interrupted run picks up where it stopped. The models load from
models/hf with HF_HUB_OFFLINE=1.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from voice_severity_bench import feature_set_from_table, save_features  # noqa: E402

MODELS = common.ROOT / "models"
CACHE = common.ROOT / "features" / "cache"
CHUNK_S = 30.0
RATE = 16_000
#: Whisper's encoder hop: 10 ms mel frames, then a stride-2 convolution.
WHISPER_HOP = 320


def model_dir(name: str) -> Path:
    repo = common.plan()["feature_sets"][name]["model"]
    return MODELS / repo.replace("/", "__")


class Encoder:
    def __init__(self, name: str):
        from transformers import AutoFeatureExtractor, AutoModel, WhisperModel

        self.name = name
        self.spec = common.plan()["feature_sets"][name]
        self.path = model_dir(name)
        self.whisper = bool(self.spec.get("encoder_only"))
        self.extractor = AutoFeatureExtractor.from_pretrained(self.path)
        if self.whisper:
            self.model = WhisperModel.from_pretrained(self.path).encoder
        else:
            self.model = AutoModel.from_pretrained(self.path)
        self.model.eval()

    @torch.no_grad()
    def _chunk(self, signal: np.ndarray) -> list[np.ndarray]:
        if self.whisper:
            inputs = self.extractor(signal, sampling_rate=RATE, return_tensors="pt")
            hidden = self.model(inputs.input_features, output_hidden_states=True).hidden_states
            keep = math.ceil(signal.size / WHISPER_HOP)
            return [state[0, :keep].numpy() for state in hidden]
        inputs = self.extractor(signal, sampling_rate=RATE, return_tensors="pt")
        hidden = self.model(inputs.input_values, output_hidden_states=True).hidden_states
        return [state[0].numpy() for state in hidden]

    def layers(self, signal: np.ndarray) -> list[np.ndarray]:
        """Frame matrices of the 13 hidden states for the whole signal."""
        size = int(CHUNK_S * RATE)
        starts = list(range(0, signal.size, size))
        # A last piece under 1 s goes with the chunk before it.
        if len(starts) > 1 and signal.size - starts[-1] < RATE:
            starts.pop()
        pieces = [signal[a : (starts[k + 1] if k + 1 < len(starts) else signal.size)]
                  for k, a in enumerate(starts)]  # fmt: skip
        per_chunk = [self._chunk(piece.astype(np.float32)) for piece in pieces]
        n_layers = len(per_chunk[0])
        expected = int(self.spec["n_hidden_states"])
        if n_layers != expected:
            raise RuntimeError(f"{self.name}: {n_layers} hidden states, the plan says {expected}")
        return [np.concatenate([chunk[k] for chunk in per_chunk]) for k in range(n_layers)]


def pooled(layers: list[np.ndarray]) -> np.ndarray:
    """(14, 1536): mean and std of each layer, then of the frame-level layer average."""
    stack = [*layers, np.mean(np.stack(layers), axis=0)]
    return np.stack([np.concatenate([m.mean(axis=0), m.std(axis=0)]) for m in stack])


def extract(
    name: str, material: str, limit: int | None, recompute: bool, files: list[str] | None = None
) -> dict[str, float]:
    index = common.index(material)
    if files:
        index = index[index["file_id"].isin(files)]
    if limit is not None:
        index = index.iloc[:limit]
    encoder = Encoder(name)
    cache = CACHE / name / material
    cache.mkdir(parents=True, exist_ok=True)
    timing = {"audio_s": 0.0, "compute_s": 0.0, "items": 0}
    for _, row in index.iterrows():
        target = cache / f"{row['file_id']}.npz"
        if target.is_file() and not recompute:
            continue
        signal, rate = common.load_interval(row)
        assert rate == RATE
        start = time.perf_counter()
        vectors = pooled(encoder.layers(signal))
        timing["compute_s"] += time.perf_counter() - start
        timing["audio_s"] += signal.size / RATE
        timing["items"] += 1
        if not np.isfinite(vectors).all():
            raise RuntimeError(f"{name} {material} {row['file_id']}: non-finite features")
        np.savez_compressed(target, pooled=vectors.astype(np.float64))
    return timing


def assemble(name: str, material: str) -> Path:
    """Gather the cached items of a material into features/out/<material>/<name>.npz."""
    index = common.index(material)
    cache = CACHE / name / material
    matrices = [np.load(cache / f"{file_id}.npz")["pooled"] for file_id in index["file_id"]]
    n_blocks, width = matrices[0].shape
    labels = [f"L{k:02d}" for k in range(n_blocks - 1)] + ["AVG"]
    half = width // 2
    columns = {
        label: [f"{label}_{stat}{d}" for stat in ("mean", "std") for d in range(half)]
        for label in labels
    }
    data = np.stack([m.reshape(-1) for m in matrices])
    names = [column for label in labels for column in columns[label]]
    table = index[["path", "group", "start_s", "end_s", common.PRIMARY_TARGET]].reset_index(drop=True)
    table = table.join(pd.DataFrame(data, columns=names))
    settings = {
        "model": common.plan()["feature_sets"][name]["model"],
        "model_dir": model_dir(name).relative_to(common.ROOT).as_posix(),
        "weights_sha256": {
            p.name: common.sha256(p)
            for p in sorted(model_dir(name).glob("*"))
            if p.suffix in (".safetensors", ".bin")
        },
        "revision": (model_dir(name) / "REVISION").read_text(encoding="utf-8").strip(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "chunk_s": CHUNK_S,
        "pooling": ["mean", "std"],
        "avg_block": "frame-level mean of the 13 hidden states, then pooled",
    }
    features = feature_set_from_table(
        table,
        target=common.PRIMARY_TARGET,
        blocks=columns,
        kind="ssl",
        extractor=common.provenance(__file__, settings),
    )
    destination = common.OUT / material / f"{name}.npz"
    save_features(destination, features)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=["A4", "A5", "A6"])
    parser.add_argument("--materials", nargs="+", default=["a"], choices=common.MATERIALS)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N items (trial run).")
    parser.add_argument("--recompute", action="store_true", help="Ignore the cache.")
    parser.add_argument("--files", nargs="*", help="Only these file ids (trial run).")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    for material in args.materials:
        timing = extract(args.model, material, args.limit, args.recompute, args.files)
        rate = timing["audio_s"] / timing["compute_s"] if timing["compute_s"] else float("nan")
        print(
            f"{args.model} {material}: {timing['items']} new items, "
            f"{timing['audio_s']:.1f} s audio in {timing['compute_s']:.1f} s ({rate:.1f}x real time)"
        )
        if args.limit is None and not args.files:
            print("written", assemble(args.model, material))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
