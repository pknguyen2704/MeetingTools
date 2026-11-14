"""Streaming utterance segmentation (endpointing) with WebRTC VAD.

Audio comes in as float32 mono @16 kHz in arbitrary chunk sizes. The segmenter
detects speech start, accumulates the utterance, and closes it when the speaker
pauses (``end_silence_ms``) or the utterance becomes too long. Times are counted
in samples since the start of the stream, so they line up exactly with the audio
track recorded into the video file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import webrtcvad

from .config import SAMPLE_RATE

FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples


@dataclass
class Utterance:
    id: int
    start_sample: int
    audio: np.ndarray  # float32

    @property
    def start(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def end(self) -> float:
        return (self.start_sample + len(self.audio)) / SAMPLE_RATE

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.audio)


class UtteranceSegmenter:
    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        end_silence_ms: int = 700,
        soft_max_s: float = 15.0,
        hard_max_s: float = 24.0,
        min_rms: float = 0.004,
        preroll_ms: int = 300,
        start_window: int = 8,
        start_voiced: int = 5,
        min_voiced_ms: int = 250,
        keep_tail_ms: int = 300,
        soft_cut_silence_ms: int = 210,
    ) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)
        self.end_silence = end_silence_ms // FRAME_MS
        self.soft_max = int(soft_max_s * 1000) // FRAME_MS
        self.hard_max = int(hard_max_s * 1000) // FRAME_MS
        self.soft_cut_silence = soft_cut_silence_ms // FRAME_MS
        self.min_rms = min_rms
        self.min_voiced = max(1, min_voiced_ms // FRAME_MS)
        self.keep_tail = keep_tail_ms // FRAME_MS
        self.start_voiced = start_voiced
        self._history: deque[bool] = deque(maxlen=start_window)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self._utt_start = 0  # sample index of first frame in _frames
        self._voiced = 0
        self._silence_run = 0
        self._in_speech = False
        self._sample_pos = 0  # samples consumed into frames
        self._next_id = 0

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._in_speech

    @property
    def position(self) -> float:
        """Seconds of audio processed so far."""
        return self._sample_pos / SAMPLE_RATE

    def current(self) -> Utterance | None:
        """Snapshot of the utterance in progress (for partial hypotheses)."""
        if not self._in_speech or not self._frames:
            return None
        return Utterance(self._next_id, self._utt_start, np.concatenate(self._frames))

    @property
    def current_duration(self) -> float:
        return len(self._frames) * FRAME_MS / 1000 if self._in_speech else 0.0

    # --------------------------------------------------------------- feeding
    def push(self, audio: np.ndarray) -> list[Utterance]:
        if self._pending.size:
            audio = np.concatenate([self._pending, audio])
        n = len(audio) // FRAME
        done: list[Utterance] = []
        for i in range(n):
            utt = self._frame(audio[i * FRAME : (i + 1) * FRAME])
            if utt is not None:
                done.append(utt)
        self._pending = audio[n * FRAME :].copy()
        return done

    def flush(self) -> list[Utterance]:
        """End of stream: close whatever is in progress."""
        out = []
        if self._in_speech:
            utt = self._close(trim_tail=True)
            if utt is not None:
                out.append(utt)
        self._pending = np.zeros(0, dtype=np.float32)
        return out

    # -------------------------------------------------------------- internal
    def _is_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < self.min_rms:
            return False
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return self._vad.is_speech(pcm, SAMPLE_RATE)

    def _frame(self, frame: np.ndarray) -> Utterance | None:
        frame_start = self._sample_pos
        self._sample_pos += FRAME
        speech = self._is_speech(frame)

        if not self._in_speech:
            self._preroll.append(frame)
            self._history.append(speech)
            if sum(self._history) >= self.start_voiced:
                self._in_speech = True
                self._frames = list(self._preroll)
                self._utt_start = frame_start + FRAME - len(self._frames) * FRAME
                self._voiced = sum(self._history)
                self._silence_run = 0
                self._preroll.clear()
                self._history.clear()
            return None

        self._frames.append(frame)
        if speech:
            self._voiced += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        length = len(self._frames)
        if self._silence_run >= self.end_silence:
            return self._close(trim_tail=True)
        if length >= self.soft_max and self._silence_run >= self.soft_cut_silence:
            return self._close(trim_tail=True, keep_listening=True)
        if length >= self.hard_max:
            return self._close(trim_tail=False, keep_listening=True)
        return None

    def _close(self, *, trim_tail: bool, keep_listening: bool = False) -> Utterance | None:
        frames = self._frames
        if trim_tail and self._silence_run > self.keep_tail:
            frames = frames[: len(frames) - (self._silence_run - self.keep_tail)]
        utt = None
        if self._voiced >= self.min_voiced and frames:
            utt = Utterance(self._next_id, self._utt_start, np.concatenate(frames))
            self._next_id += 1
        cont = keep_listening and self._silence_run < self.soft_cut_silence
        self._frames = []
        self._voiced = 0
        self._silence_run = 0
        self._history.clear()
        self._preroll.clear()
        if cont:
            # Hard cut in the middle of speech: the next utterance starts right here.
            self._in_speech = True
            self._utt_start = self._sample_pos
        else:
            self._in_speech = False
        return utt
