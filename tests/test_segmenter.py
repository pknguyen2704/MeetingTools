import numpy as np

from meetingtool.config import SAMPLE_RATE
from meetingtool.segmenter import UtteranceSegmenter


def _voice(sec: float) -> np.ndarray:
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    # crude vowel-like signal: harmonics of 140 Hz with syllable-rate AM
    sig = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
    return (0.2 * sig * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


def _feed(seg, audio, chunk=1234):
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.push(audio[i : i + chunk])
    return out + seg.flush()


def test_splits_on_pauses_with_accurate_times():
    seg = UtteranceSegmenter(aggressiveness=1, end_silence_ms=600)
    audio = np.concatenate([_silence(1.0), _voice(2.0), _silence(1.2), _voice(1.5), _silence(1.0)])
    utts = _feed(seg, audio)
    assert len(utts) == 2
    assert abs(utts[0].start - 1.0) < 0.35 and abs(utts[1].start - 4.2) < 0.35
    assert utts[0].end < 3.5


def test_long_speech_is_cut_below_whisper_window():
    seg = UtteranceSegmenter(aggressiveness=1, soft_max_s=10, hard_max_s=12)
    utts = _feed(seg, np.concatenate([_voice(30.0), _silence(1.0)]))
    assert len(utts) >= 3
    assert all(u.end - u.start <= 12.05 for u in utts)
    # contiguous: hard cuts must not drop audio
    for a, b in zip(utts, utts[1:]):
        assert abs(b.start - a.end) < 0.05


def test_ignores_low_level_noise():
    seg = UtteranceSegmenter(min_rms=0.01)
    noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3) * 0.002).astype(np.float32)
    assert _feed(seg, noise) == []
