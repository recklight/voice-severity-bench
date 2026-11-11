"""Run folders and manifests: naming, key order, digests and JSON conventions."""

from __future__ import annotations

import json

import numpy as np

from voice_severity_bench.manifest import (
    array_sha256,
    build_manifest,
    file_sha256,
    run_directory,
    write_json,
)


def test_run_directory_adds_a_suffix_when_the_name_exists(tmp_path):
    first = run_directory(tmp_path, "evaluate")
    second = run_directory(tmp_path, "evaluate")
    assert first.name.startswith("evaluate_") and first.name.endswith("Z")
    # Same second: a suffix is added; a new second gives a later time stamp instead.
    assert second.name == f"{first.name}_1" or second.name > first.name
    assert first.is_dir() and second.is_dir() and first != second


def test_manifest_keys_and_order(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"abc")
    manifest = build_manifest(
        command="prepare",
        config={"seed": 0},
        inputs=[source, tmp_path / "gone.csv"],
        seed=0,
        extra={"n_items": np.int64(3)},
    )
    assert list(manifest) == [
        "command",
        "created_utc",
        "package_version",
        "git_commit",
        "seed",
        "versions",
        "inputs",
        "config",
        "details",
    ]
    assert list(manifest["versions"])[:2] == ["python", "platform"]
    assert "scikit-learn" in manifest["versions"] and "speechdsp" in manifest["versions"]
    assert manifest["inputs"][0] == {
        "path": source.as_posix(),
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "bytes": 3,
    }
    assert manifest["inputs"][1]["missing"] is True
    assert manifest["details"] == {"n_items": 3}
    assert manifest["created_utc"].endswith("+00:00")


def test_array_digest_depends_on_the_dtype():
    assert array_sha256(np.zeros(3)) == array_sha256(np.zeros(3))
    assert array_sha256(np.zeros(3)) != array_sha256(np.zeros(3, dtype=np.float32))


def test_json_turns_nan_into_null_and_ends_with_a_newline(tmp_path):
    path = write_json(tmp_path / "x.json", {"a": float("nan"), "b": np.array([1.0, np.inf])})
    text = path.read_bytes()
    assert text.endswith(b"\n") and b"\r\n" not in text
    assert json.loads(text) == {"a": None, "b": [1.0, None]}
    assert file_sha256(path) == file_sha256(path)
