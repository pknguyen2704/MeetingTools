"""Realtime transcription engine.

One worker thread owns VAD segmentation and both Whisper models so CPU use stays
bounded on a 4-core Intel laptop:

* **partial** pass – a tiny model re-decodes the utterance in progress about once
  per second (greedy, no timestamps) and the UI shows it greyed out;
* **final** pass – when the speaker pauses, the bigger model decodes the whole
  utterance with beam search, context prompt and timestamps; the result is
  normalised (punctuation, casing) and replaces the partial.

Finals always take priority; partials are skipped while there is a backlog.

Language: with ``language="auto"`` every utterance is classified as English or
Vietnamese (Whisper's language-ID restricted to those two), so a meeting can
switch language from one sentence to the next. English-only (``.en``) models
always decode English.
"""

from __future__ import annotations

import bisect
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import AUTO_LANGUAGES, SAMPLE_RATE, AsrSettings
from .segmenter import Utterance, UtteranceSegmenter
from .textnorm import clean_partial, is_hallucination, normalize

log = logging.getLogger(__name__)

_EOF = object()


@dataclass
class Cue:
    start: float  # seconds on the recording's audio timeline
    end: float
    text: str
    language: str = ""


@dataclass
class PartialResult:
    utt_id: int
    start: float
    text: str
    language: str = ""


@dataclass
class FinalResult:
    utt_id: int
    start: float
    end: float
    text: str  # normalised, may be "" when nothing usable was said
    cues: list[Cue] = field(default_factory=list)
    asr_seconds: float = 0.0
    latency: float | None = None  # wall-clock: end of speech -> final text
    language: str = ""


def is_multilingual(model) -> bool:
    inner = getattr(model, "model", None)
    return bool(getattr(inner, "is_multilingual", True))


def pick_language(
    probs: list[tuple[str, float]],
    previous: str,
    *,
    duration: float,
    allowed: tuple[str, ...] = AUTO_LANGUAGES,
) -> str:
    """Choose among ``allowed`` from Whisper's language probabilities.

    Short, ambiguous utterances ("OK", "yes", names) keep the previous language
    instead of flip-flopping.
    """
    scores = {lang: p for lang, p in probs if lang in allowed}
    if not scores:
        return previous
    best = max(scores, key=scores.get)
    if best != previous and previous in scores and duration < 1.5:
        if scores[best] < 2.0 * scores[previous]:
            return previous
    return best


class ModelPool:
    """Keeps loaded WhisperModel instances alive across recording sessions."""

    _lock = threading.Lock()
    _models: dict[tuple, object] = {}

    @classmethod
    def get(cls, name: str, cfg: AsrSettings):
        from faster_whisper import WhisperModel

        key = (name, cfg.device, cfg.compute_type, cfg.cpu_threads, str(cfg.model_dir))
        with cls._lock:
            if key not in cls._models:
                cls._models[key] = cls._load(WhisperModel, name, cfg)
            return cls._models[key]

    @staticmethod
    def _load(WhisperModel, name: str, cfg: AsrSettings):  # noqa: N803
        attempts = [(cfg.device, cfg.compute_type), (cfg.device, "default")]
        if cfg.device != "cpu":
            attempts += [("cpu", cfg.compute_type), ("cpu", "default")]
        last: Exception | None = None
        for device, ctype in attempts:
            try:
                model = WhisperModel(
                    name,
                    device=device,
                    compute_type=ctype,
                    cpu_threads=cfg.cpu_threads,
                    num_workers=1,
                    download_root=str(cfg.model_dir) if cfg.model_dir else None,
                )
                if (device, ctype) != attempts[0]:
                    log.warning("%s: using device=%s compute_type=%s", name, device, ctype)
                model.device_used = device
                return model
            except (ValueError, RuntimeError) as exc:
                last = exc
        raise RuntimeError(f"cannot load {name}: {last}")


class TranscriptionEngine:
    def __init__(
        self,
        cfg: AsrSettings,
        *,
        on_partial: Callable[[PartialResult], None] | None = None,
        on_final: Callable[[FinalResult], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.on_partial = on_partial or (lambda r: None)
        self.on_final = on_final or (lambda r: None)
        self.on_status = on_status or (lambda s: None)
        self.on_error = on_error or (lambda s: None)
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="asr", daemon=True)
        self._abort = threading.Event()
        self.ready = threading.Event()
        self._segmenter = UtteranceSegmenter(
            aggressiveness=cfg.vad_aggressiveness,
            end_silence_ms=cfg.end_silence_ms,
            soft_max_s=cfg.soft_max_s,
            hard_max_s=cfg.hard_max_s,
            min_rms=cfg.min_rms,
        )
        self._finals: deque[Utterance] = deque()
        self._history: deque[tuple[str, str]] = deque(maxlen=12)  # (language, text)
        self._last_lang = "vi" if cfg.language == "vi" else "en"
        self._partial_lang: tuple[int, str] | None = None  # (utt_id, language)
        self._last_partial_pos = 0.0
        self._last_partial_text = ""
        self._eof = False
        # wall-clock bookkeeping for latency measurement
        self._fed_samples = 0
        self._feed_marks: list[tuple[int, float]] = []
        self._feed_lock = threading.Lock()
        self.final_model = None
        self.partial_model = None
        # stats
        self.asr_time = 0.0
        self.latencies: list[float] = []

    # ---------------------------------------------------------------- public
    def start(self) -> None:
        self._thread.start()

    def feed(self, audio: np.ndarray) -> None:
        """Thread-safe, never blocks (called from the ffmpeg reader thread)."""
        with self._feed_lock:
            self._fed_samples += len(audio)
            self._feed_marks.append((self._fed_samples, time.monotonic()))
            if len(self._feed_marks) > 20_000:
                del self._feed_marks[:10_000]
        self._q.put(audio)

    def finish(self) -> None:
        """Signal end of audio; pending speech is still transcribed."""
        self._q.put(_EOF)

    def abort(self) -> None:
        self._abort.set()
        self._q.put(_EOF)

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def backlog_seconds(self) -> float:
        pending = sum(len(u.audio) for u in self._finals) / SAMPLE_RATE
        with self._feed_lock:
            fed = self._fed_samples / SAMPLE_RATE
        return pending + max(0.0, fed - self._segmenter.position)

    # -------------------------------------------------------------- internal
    def _load(self) -> None:
        cfg = self.cfg
        self.on_status(f"Đang tải mô hình {cfg.final_model}…")
        self.final_model = ModelPool.get(cfg.final_model, cfg)
        pm = cfg.partial_model
        if pm == "same":
            self.partial_model = self.final_model
        elif pm:
            self.on_status(f"Đang tải mô hình nháp {pm}…")
            self.partial_model = ModelPool.get(pm, cfg)
        self.ready.set()
        dev = getattr(self.final_model, "device_used", "cpu")
        lang = cfg.language
        if lang != "en" and not is_multilingual(self.final_model):
            log.warning("%s is English-only; language=%s ignored", cfg.final_model, lang)
            self.on_status(f"⚠ {cfg.final_model} chỉ hỗ trợ tiếng Anh – chọn mô hình không có '.en' cho tiếng Việt")
            return
        self.on_status(f"Mô hình sẵn sàng ({cfg.final_model} trên {dev}, ngôn ngữ: {lang})")

    def _run(self) -> None:
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            log.exception("model load failed")
            self.on_error(f"Không tải được mô hình: {exc}")
            self.ready.set()
            return
        try:
            self._loop()
        except Exception as exc:  # noqa: BLE001
            log.exception("ASR loop crashed")
            self.on_error(f"Lỗi chép lời: {exc}")

    def _accept(self, item) -> None:
        if item is _EOF:
            self._eof = True
            self._finals.extend(self._segmenter.flush())
        else:
            self._finals.extend(self._segmenter.push(item))

    def _drain(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            self._accept(item)

    def _loop(self) -> None:
        while not self._abort.is_set():
            self._drain()
            if self._finals:
                self._do_final(self._finals.popleft())
                continue
            if self._eof:
                break
            if self._should_partial():
                self._do_partial()
                continue
            try:
                self._accept(self._q.get(timeout=0.05))
            except queue.Empty:
                pass

    def _should_partial(self) -> bool:
        if self.partial_model is None or not self._segmenter.active:
            return False
        if self._segmenter.current_duration < 0.6:
            return False
        if self._q.qsize() > 10:  # ~1 s of audio waiting: catch up first
            return False
        return self._segmenter.position - self._last_partial_pos >= self.cfg.partial_interval_s

    def _language(self, model, audio: np.ndarray, *, detect: bool, duration: float) -> str:
        """Language to decode ``audio`` with; runs language-ID only when ``detect``."""
        if not is_multilingual(model):
            return "en"
        if self.cfg.language != "auto":
            return self.cfg.language
        if not detect:
            return self._last_lang
        try:
            _, _, probs = model.detect_language(audio=audio)
        except Exception:  # noqa: BLE001 - never lose a sentence over language-ID
            log.exception("language detection failed")
            return self._last_lang
        return pick_language(probs, self._last_lang, duration=duration)

    def _prompt(self, lang: str) -> str | None:
        parts = []
        if self.cfg.glossary.strip():
            label = "Thuật ngữ" if lang == "vi" else "Glossary"
            parts.append(f"{label}: {self.cfg.glossary.strip()}.")
        # Only same-language context: an English prompt nudges Whisper to translate
        # Vietnamese speech into English (and vice versa).
        ctx = " ".join(t for lg, t in self._history if lg == lang)
        if ctx:
            parts.append(ctx[-400:])
        elif self.cfg.style_prompts.get(lang):
            parts.append(self.cfg.style_prompts[lang])
        return " ".join(parts) or None

    def _do_partial(self) -> None:
        utt = self._segmenter.current()
        if utt is None:
            return
        self._last_partial_pos = self._segmenter.position
        t0 = time.perf_counter()
        # Language-ID on the draft model until ~2 s of speech, then keep that answer.
        dur = len(utt.audio) / SAMPLE_RATE
        cached = self._partial_lang if self._partial_lang and self._partial_lang[0] == utt.id else None
        if cached and dur >= 2.0:
            lang = cached[1]
        else:
            lang = self._language(self.partial_model, utt.audio, detect=True, duration=dur)
            self._partial_lang = (utt.id, lang)
        segments, _ = self.partial_model.transcribe(
            utt.audio,
            language=lang,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self._prompt(lang),
            vad_filter=False,
        )
        text = clean_partial(" ".join(s.text.strip() for s in segments))
        self.asr_time += time.perf_counter() - t0
        if text and text != self._last_partial_text and not is_hallucination(text):
            self._last_partial_text = text
            self.on_partial(PartialResult(utt.id, utt.start, text, language=lang))

    def _latency_for(self, end_sample: int) -> float | None:
        with self._feed_lock:
            marks = self._feed_marks
            idx = bisect.bisect_left(marks, (end_sample, 0.0))
            if idx >= len(marks):
                return None
            fed_at = marks[idx][1]
        return time.monotonic() - fed_at

    def _do_final(self, utt: Utterance) -> None:
        cfg = self.cfg
        t0 = time.perf_counter()
        # Falling behind (slow CPU, long monologue)? Trade a bit of accuracy for speed.
        behind = self.backlog_seconds > cfg.degrade_backlog_s
        beam = 1 if behind else cfg.beam_size
        if behind and self._partial_lang and self._partial_lang[0] == utt.id:
            lang = self._partial_lang[1]  # skip the extra encoder pass, reuse the draft's guess
        else:
            lang = self._language(
                self.final_model, utt.audio, detect=not behind, duration=utt.end - utt.start
            )
        self._last_lang = lang
        segments, _ = self.final_model.transcribe(
            utt.audio,
            language=lang,
            beam_size=beam,
            temperature=[0.0, 0.2, 0.4, 0.6],
            condition_on_previous_text=False,
            initial_prompt=self._prompt(lang),
            without_timestamps=False,
            vad_filter=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        cues: list[Cue] = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
                continue
            if seg.compression_ratio > 2.6:
                continue
            raw = seg.text.strip()
            if is_hallucination(raw) and (seg.avg_logprob < -0.6 or utt.end - utt.start < 1.5):
                continue
            text = normalize(raw, remove_fillers=cfg.remove_fillers, language=lang)
            if not text:
                continue
            start = utt.start + max(0.0, seg.start)
            end = min(utt.end, utt.start + max(seg.end, seg.start + 0.3))
            cues.append(Cue(start, end, text, lang))
        dt = time.perf_counter() - t0
        self.asr_time += dt
        full = " ".join(c.text for c in cues)
        if full:
            self._history.append((lang, full))
        latency = self._latency_for(utt.end_sample)
        if latency is not None:
            self.latencies.append(latency)
        self._last_partial_text = ""
        self._last_partial_pos = self._segmenter.position
        self.on_final(
            FinalResult(
                utt.id, utt.start, utt.end, full, cues,
                asr_seconds=dt, latency=latency, language=lang,
            )  # fmt: skip
        )
