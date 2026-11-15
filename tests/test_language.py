import numpy as np

from meetingtool.config import AsrSettings, is_english_only, multilingual_counterpart
from meetingtool.engine import TranscriptionEngine, pick_language
from meetingtool.segmenter import Utterance


def test_pick_language_restricts_to_en_vi():
    probs = [("zh", 0.6), ("vi", 0.3), ("en", 0.1)]
    assert pick_language(probs, "en", duration=3.0) == "vi"


def test_pick_language_sticky_on_short_ambiguous():
    probs = [("vi", 0.5), ("en", 0.4)]
    assert pick_language(probs, "en", duration=0.8) == "en"
    assert pick_language(probs, "en", duration=3.0) == "vi"
    assert pick_language([("vi", 0.9), ("en", 0.05)], "en", duration=0.8) == "vi"


def test_model_name_helpers():
    assert is_english_only("small.en") and not is_english_only("small")
    assert multilingual_counterpart("small.en") == "small"
    assert multilingual_counterpart("distil-medium.en") == "medium"
    assert multilingual_counterpart("large-v3-turbo") == "large-v3-turbo"


class _Seg:
    def __init__(self, text):
        self.text, self.start, self.end = text, 0.0, 1.0
        self.no_speech_prob, self.avg_logprob, self.compression_ratio = 0.0, -0.2, 1.2


class _Inner:
    def __init__(self, multilingual):
        self.is_multilingual = multilingual


class FakeModel:
    """Says each utterance is whatever language `langs` yields next."""

    def __init__(self, langs, multilingual=True):
        self.model = _Inner(multilingual)
        self.langs = list(langs)
        self.calls = []

    def detect_language(self, audio):
        lang = self.langs.pop(0)
        other = "en" if lang == "vi" else "vi"
        return lang, 0.9, [(lang, 0.9), (other, 0.05)]

    def transcribe(self, audio, language, initial_prompt=None, **_):
        self.calls.append((language, initial_prompt))
        text = "xin chào mọi người" if language == "vi" else "hello everyone"
        return [_Seg(text)], None


def _run(model, cfg, n):
    finals = []
    eng = TranscriptionEngine(cfg, on_final=finals.append)
    eng.final_model = model
    for i in range(n):
        audio = np.zeros(16000 * 2, dtype=np.float32)
        eng._do_final(Utterance(id=i, start_sample=i * 3 * 16000, audio=audio))
    return finals


def test_auto_switches_language_per_utterance():
    model = FakeModel(["en", "vi", "vi"])
    finals = _run(model, AsrSettings(language="auto"), 3)
    assert [f.language for f in finals] == ["en", "vi", "vi"]
    assert finals[1].text == "Xin chào mọi người."
    # Context prompt only carries same-language history.
    assert "hello" not in (model.calls[1][1] or "").lower()
    assert "xin chào" in model.calls[2][1].lower()


def test_fixed_language_skips_detection():
    model = FakeModel([])  # detect_language would raise IndexError if called
    finals = _run(model, AsrSettings(language="vi"), 2)
    assert [f.language for f in finals] == ["vi", "vi"]


def test_english_only_model_forces_english():
    model = FakeModel([], multilingual=False)
    finals = _run(model, AsrSettings(language="auto"), 1)
    assert finals[0].language == "en"
