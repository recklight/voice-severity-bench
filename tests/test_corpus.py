"""Batch extraction: audio-only discovery, one status row per file, the output formats
and the audio settings."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile
from speechdsp import read_htk, write_wav

from voice_severity_bench import (
    AudioConfig,
    Config,
    FrontendConfig,
    extract_corpus,
    frame_features,
    htk_parameter_kind,
    iter_audio_files,
    load_frame_features,
    make_vowel,
    read_audio,
    read_text_matrix,
)

RATE = 16_000


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    write_wav(root / "spk1" / "a.wav", make_vowel(10.0, rng=1)[:8000], RATE)
    write_wav(root / "spk2" / "deep" / "b.WAV", make_vowel(70.0, rng=2)[:6000], RATE)
    write_wav(root / "tiny.wav", np.zeros(50), RATE)
    (root / "spk1" / "notes.txt").write_text("not audio", encoding="utf-8")
    (root / "broken.wav").write_bytes(b"RIFF....garbage")
    return root


def test_only_audio_files_are_listed_in_a_stable_order(corpus):
    names = [path.relative_to(corpus).as_posix() for path in iter_audio_files(corpus)]
    assert names == ["broken.wav", "spk1/a.wav", "spk2/deep/b.WAV", "tiny.wav"]


@pytest.mark.parametrize("fmt", ["npz", "htk", "text"])
def test_every_file_gets_a_row_and_outputs_mirror_the_tree(tmp_path, corpus, fmt):
    table = extract_corpus(corpus, tmp_path / "out", Config(), fmt=fmt)
    status = dict(zip(table["path"], table["status"], strict=True))
    assert status == {
        "broken.wav": "error",
        "spk1/a.wav": "ok",
        "spk2/deep/b.WAV": "ok",
        "tiny.wav": "too_short",
    }
    assert table.loc[table["status"] == "error", "message"].iloc[0] != ""
    row = table[table["path"] == "spk1/a.wav"].iloc[0]
    written = tmp_path / "out" / row["output"]
    assert written.is_file() and written.parent.name == "spk1"
    features, _ = load_frame_features(written)
    signal, rate = read_audio(corpus / "spk1" / "a.wav", AudioConfig())
    expected = frame_features(signal, rate, FrontendConfig())
    tolerance = 1e-4 if fmt == "htk" else 1e-6
    assert features.shape == (row["n_frames"], 39)
    assert np.allclose(features, expected, rtol=tolerance, atol=tolerance)


def test_htk_header_describes_the_layout(tmp_path, corpus):
    table = extract_corpus(corpus, tmp_path / "out", Config(), fmt="htk")
    output = table.loc[table["path"] == "spk1/a.wav", "output"].iloc[0]
    _, header = read_htk(tmp_path / "out" / output)
    assert header["parm_kind"] == 8966  # MFCC_0_D_A
    assert header["samp_period"] == 150_000  # 15 ms in 100 ns units
    assert header["samp_size"] == 39 * 4


def test_parameter_kinds():
    assert htk_parameter_kind(FrontendConfig(delta_order=0)) == 6 | 0o20000
    assert (
        htk_parameter_kind(FrontendConfig(energy_term="log_energy", delta_order=1))
        == 6 | 0o100 | 0o400
    )
    assert (
        htk_parameter_kind(FrontendConfig(energy_term="none", delta_order=2)) == 6 | 0o400 | 0o1000
    )


def test_text_matrix_round_trip(tmp_path):
    matrix = np.arange(6.0).reshape(3, 2) / 7.0
    path = tmp_path / "m.txt"
    np.savetxt(path, matrix, fmt="%.8e")
    assert np.allclose(read_text_matrix(path), matrix, rtol=1e-8)


def test_audio_settings_resample_and_trim(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, 0.5 * np.sin(2 * np.pi * 200 * np.arange(44_100) / 44_100), 44_100)
    signal, rate = read_audio(path, AudioConfig(sample_rate=16_000, trim_s=0.25))
    assert rate == 16_000 and signal.size == 16_000 - 2 * 4_000
    with pytest.raises(ValueError, match="trim_s"):
        read_audio(path, AudioConfig(trim_s=0.6))


def test_non_finite_samples_are_rejected(tmp_path):
    signal = (0.5 * np.sin(np.arange(RATE) / 5.0)).astype(np.float32)
    signal[100] = np.nan
    (tmp_path / "rec").mkdir()
    wavfile.write(tmp_path / "rec" / "bad.wav", RATE, signal)
    with pytest.raises(ValueError, match=r"bad\.wav contains NaN or infinite samples"):
        read_audio(tmp_path / "rec" / "bad.wav", AudioConfig())
    table = extract_corpus(tmp_path / "rec", tmp_path / "out", Config())
    assert table[["status", "message"]].values.tolist() == [
        ["error", "bad.wav contains NaN or infinite samples"]
    ]


def test_missing_folder_and_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        iter_audio_files(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no audio files"):
        extract_corpus(tmp_path / "empty", tmp_path / "out", Config())


def test_an_interval_is_cut_at_the_file_rate_before_resampling(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, 0.5 * np.sin(2 * np.pi * 200 * np.arange(44_100) / 44_100), 44_100)
    whole, _ = read_audio(path, AudioConfig(sample_rate=16_000))
    part, rate = read_audio(path, AudioConfig(sample_rate=16_000), start_s=0.25, end_s=0.75)
    assert rate == 16_000 and part.size == 8_000
    # Away from the cut edges the two signals agree up to the resampling filter.
    assert np.allclose(part[1000:7000], whole[5000:11000], atol=1e-3)
    open_end, _ = read_audio(path, AudioConfig(), start_s=0.5, end_s=float("nan"))
    assert open_end.size == 44_100 - 22_050


def test_intervals_outside_the_recording_are_rejected(tmp_path):
    path = tmp_path / "x.wav"
    write_wav(path, 0.5 * np.sin(np.arange(RATE) / 5.0), RATE)
    for start, end in ((0.5, 1.5), (0.8, 0.2), (2.0, None)):
        with pytest.raises(ValueError, match="does not fit the recording"):
            read_audio(path, AudioConfig(), start_s=start, end_s=end)


def test_recordings_are_scaled_to_the_requested_level(tmp_path):
    loud, quiet = tmp_path / "loud.wav", tmp_path / "quiet.wav"
    tone = np.sin(2 * np.pi * 200 * np.arange(RATE) / RATE)
    write_wav(loud, 0.5 * tone, RATE)
    write_wav(quiet, 0.01 * tone, RATE)
    audio = AudioConfig(rms_dbfs=-26.0)
    a, _ = read_audio(loud, audio)
    b, _ = read_audio(quiet, audio)
    assert np.allclose(a, b, atol=2e-4)
    assert np.isclose(20 * np.log10(np.sqrt(np.mean(a**2))), -26.0)
    write_wav(tmp_path / "silent.wav", np.zeros(RATE), RATE)
    with pytest.raises(ValueError, match="silent"):
        read_audio(tmp_path / "silent.wav", audio)
