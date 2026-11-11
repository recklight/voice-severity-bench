"""Synthetic checks of segment.py with known boundaries (python -m pytest, in its environment)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

import segment

RATE = 44100  # written at a PVQD rate so the resampling path runs too
FORMANTS = {
    # Peterson & Barney (1952), adult male averages: (F1, F2, F3) Hz and bandwidths
    "a": ((730, 1090, 2440), (90, 110, 170)),
    "i": ((270, 2290, 3010), (60, 100, 150)),
    "e": ((530, 1840, 2480), (70, 100, 160)),
    "o": ((570, 840, 2410), (80, 90, 160)),
    "u": ((300, 870, 2240), (60, 90, 150)),
}


def resonator(x: np.ndarray, freq: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / RATE)
    theta = 2 * np.pi * freq / RATE
    a = [1.0, -2 * r * np.cos(theta), r * r]
    return lfilter([1 - r], a, x)


def vowel(name: str, seconds: float, rng: np.random.Generator, f0: float = 120.0,
          jitter: float = 0.01, breath: float = 0.02, f0_end: float = None) -> np.ndarray:
    n = int(seconds * RATE)
    source = np.zeros(n)
    t = 0.0
    while t < n:
        source[int(t)] = 1.0
        pitch = f0 if f0_end is None else f0 + (f0_end - f0) * t / n
        t += RATE / (pitch * (1 + jitter * rng.standard_normal()))
    source = lfilter([1.0], [1.0, -0.95], source)  # spectral tilt of the glottal pulse
    source += breath * rng.standard_normal(n)
    out = source
    for freq, bw in zip(*FORMANTS[name]):
        out = resonator(out, freq, bw)
    ramp = int(0.03 * RATE)
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    out = out * env
    return out / np.sqrt(np.mean(out**2)) * 0.1


def fricative(seconds: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(int(seconds * RATE))
    noise = noise - lfilter([0.5, 0.5], [1.0], noise)  # crude high-pass
    return noise / np.sqrt(np.mean(noise**2)) * 0.04


def sentences(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Syllable-rate speech stand-in: short vowels of changing quality, frication, closures."""
    parts, total, names = [], 0.0, list(FORMANTS)
    while total < seconds:
        for _ in range(int(rng.integers(5, 9))):
            dur = float(rng.uniform(0.10, 0.22))
            parts.append(vowel(names[int(rng.integers(len(names)))], dur, rng,
                               f0=float(rng.uniform(100, 150))))
            if rng.random() < 0.5:
                parts.append(fricative(float(rng.uniform(0.05, 0.10)), rng))
            parts.append(np.zeros(int(rng.uniform(0.03, 0.07) * RATE)))
            total += dur + 0.1
        parts.append(np.zeros(int(0.4 * RATE)))
        total += 0.4
    return np.concatenate(parts)[: int(seconds * RATE)]


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE))


def write(tmp_path: Path, pieces, rng) -> tuple:
    y = np.concatenate(pieces)
    y = y + 10 ** (-55 / 20) * rng.standard_normal(y.size)
    sf.write(tmp_path / "SYN001.wav", y, RATE, subtype="PCM_16")
    return y


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(segment, "AUDIO", tmp_path)

    def _run():
        return segment.segment_file("SYN001", "SYN001.wav", write_cache=False)

    return _run


def tokens_of(result, material):
    return [t for t in result["tokens"] if t["material"] == material]


def interval(result, material):
    return next(r for r in result["intervals"] if r["material"] == material)


def test_vowels_then_sentences_are_found_at_their_boundaries(tmp_path, run):
    rng = np.random.default_rng(1)
    write(tmp_path, [silence(0.5), vowel("a", 3.0, rng), silence(0.6), vowel("i", 3.0, rng),
                     silence(0.6), sentences(6.0, rng), silence(0.5)], rng)
    result = run()
    (a,) = tokens_of(result, "a")
    (i,) = tokens_of(result, "i")
    # Boundary error under 0.1 s, the S1 acceptance criterion.
    assert abs(a["start_s"] - 0.5) < 0.1 and abs(a["end_s"] - 3.5) < 0.1
    assert abs(i["start_s"] - 4.1) < 0.1 and abs(i["end_s"] - 7.1) < 0.1
    # The analysis intervals, after trimming, lie inside the vowels.
    ia, ii = interval(result, "a"), interval(result, "i")
    assert 0.5 <= ia["start_s"] < ia["end_s"] <= 3.5
    assert 4.1 <= ii["start_s"] < ii["end_s"] <= 7.1
    sentence = interval(result, "sentences")
    assert sentence["status"] == "valid" and sentence["start_s"] >= i["end_s"]
    assert interval(result, "a")["status"] == "valid" and interval(result, "i")["status"] == "valid"


def test_an_a_running_straight_into_an_i_is_split_near_the_change(tmp_path, run):
    rng = np.random.default_rng(2)
    a, i = vowel("a", 2.5, rng), vowel("i", 2.5, rng)
    fade = int(0.04 * RATE)
    joined = np.concatenate([a[:-fade], a[-fade:] + i[:fade], i[fade:]])
    write(tmp_path, [silence(0.5), joined, silence(0.5), sentences(5.0, rng)], rng)
    result = run()
    (ta,) = tokens_of(result, "a")
    (ti,) = tokens_of(result, "i")
    change = 0.5 + 2.5 - 0.02
    assert abs(ta["end_s"] - change) < 0.1 and abs(ti["start_s"] - change) < 0.1


def test_sentences_first_and_the_vowel_last_still_work(tmp_path, run):
    rng = np.random.default_rng(3)
    write(tmp_path, [silence(0.3), sentences(6.0, rng), silence(0.5), vowel("a", 2.0, rng), silence(0.3)], rng)
    result = run()
    (a,) = tokens_of(result, "a")
    assert a["start_s"] > 6.5
    sentence = interval(result, "sentences")
    assert sentence["status"] == "valid" and sentence["end_s"] <= a["start_s"]


def test_a_breathy_vowel_is_still_found(tmp_path, run):
    rng = np.random.default_rng(4)
    write(tmp_path, [silence(0.5), vowel("a", 2.5, rng, jitter=0.03, breath=0.6), silence(0.5),
                     sentences(5.0, rng)], rng)
    result = run()
    assert len(tokens_of(result, "a")) == 1


def test_noise_alone_gives_no_vowel(tmp_path, run):
    rng = np.random.default_rng(5)
    write(tmp_path, [0.02 * rng.standard_normal(int(5 * RATE))], rng)
    result = run()
    assert not tokens_of(result, "a") and not tokens_of(result, "i")
    assert interval(result, "a")["status"] == "invalid"


def test_the_segmenter_never_reads_the_ratings():
    source = Path(segment.__file__).read_text(encoding="utf-8")
    for name in ("targets.csv", "ratings_long", "Ratings_both_scales", '"ratings"', "_only.xlsx", "read_excel"):
        assert name not in source


def test_a_steady_vowel_is_preferred_over_a_longer_pitch_glide(tmp_path, run):
    rng = np.random.default_rng(6)
    write(tmp_path, [silence(0.5), vowel("a", 2.0, rng), silence(0.6),
                     vowel("a", 4.0, rng, f0=110.0, f0_end=220.0), silence(0.6), sentences(5.0, rng)], rng)
    result = run()
    chosen = interval(result, "a")
    assert chosen["status"] == "valid" and "glide" not in chosen["reason"]
    assert 0.5 <= chosen["start_s"] < chosen["end_s"] <= 2.5


def test_a_short_vowel_is_kept_and_flagged(tmp_path, run):
    rng = np.random.default_rng(7)
    write(tmp_path, [silence(0.4), sentences(5.0, rng), silence(0.5), vowel("a", 0.9, rng), silence(0.4)], rng)
    result = run()
    chosen = interval(result, "a")
    assert chosen["status"] == "valid" and "under_1s" in chosen["reason"]


def test_sentences_survive_when_one_speech_span_also_holds_the_vowel(tmp_path, run):
    rng = np.random.default_rng(8)
    # 0.15 s gap: shorter than the VAD's min_silence, so one span covers both.
    write(tmp_path, [silence(0.4), sentences(6.0, rng), silence(0.15), vowel("a", 3.0, rng), silence(0.4)], rng)
    result = run()
    (a,) = tokens_of(result, "a")
    sentence = interval(result, "sentences")
    assert sentence["status"] == "valid" and sentence["end_s"] <= a["start_s"]


def test_a_rise_and_fall_glide_is_still_a_glide(tmp_path, run):
    rng = np.random.default_rng(9)
    up = vowel("a", 2.0, rng, f0=110.0, f0_end=200.0)
    down = vowel("a", 2.0, rng, f0=200.0, f0_end=110.0)
    fade = int(0.04 * RATE)
    arch = np.concatenate([up[:-fade], up[-fade:] + down[:fade], down[fade:]])
    write(tmp_path, [silence(0.5), vowel("a", 2.0, rng), silence(0.6), arch, silence(0.6), sentences(5.0, rng)], rng)
    chosen = interval(run(), "a")
    assert 0.5 <= chosen["start_s"] < chosen["end_s"] <= 2.5


def test_tremor_is_not_a_glide():
    f0 = 150 * 2 ** (1.5 / 12 * np.sin(2 * np.pi * 6 * np.arange(300) * 0.01))  # 6 Hz, +-1.5 semitones
    assert segment.f0_slow_range_st(f0, 0, 300) < 1.0
