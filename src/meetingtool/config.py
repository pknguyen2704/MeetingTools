"""Settings dataclasses shared by the recorder, the ASR engine and the UI."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000  # Whisper's native sample rate; the ASR tap is resampled to this.

# Output resolution presets: label -> target height (None = keep source size).
RESOLUTIONS: dict[str, int | None] = {
    "Gốc (native)": None,
    "1440p": 1440,
    "1080p": 1080,
    "720p": 720,
    "480p": 480,
}

# Default H.264 bitrate per target height (screen content compresses well).
_BITRATE_BY_HEIGHT = {None: "6M", 1440: "6M", 1080: "4M", 720: "2500k", 480: "1200k"}

VIDEO_ENCODERS = {
    "H.264 (VideoToolbox - phần cứng)": "h264_videotoolbox",
    "HEVC (VideoToolbox - phần cứng)": "hevc_videotoolbox",
    "H.264 (x264 - CPU)": "libx264",
}

# Transcription language: label -> code. "auto" detects English vs Vietnamese
# for every utterance (speakers may switch language mid-meeting).
LANGUAGES: dict[str, str] = {
    "Tự động (Anh + Việt)": "auto",
    "Tiếng Anh": "en",
    "Tiếng Việt": "vi",
}
AUTO_LANGUAGES = ("en", "vi")  # candidates considered in "auto" mode

# Final (punctuated) models. Multilingual models (no ".en") handle English *and*
# Vietnamese; ".en" models are English-only but a bit more accurate for English.
FINAL_MODELS = [
    "small",
    "base",
    "medium",
    "large-v3-turbo",
    "small.en",
    "distil-small.en",
    "base.en",
    "distil-medium.en",
    "medium.en",
]
PARTIAL_MODELS = ["tiny", "base", "tiny.en", "base.en", "(dùng mô hình chính)", "(tắt bản nháp)"]

# Per-language prompts that set the expected style (punctuation, casing).
STYLE_PROMPTS = {
    "en": "Hello, everyone. Thanks for joining the meeting today.",
    "vi": "Xin chào mọi người. Cảm ơn mọi người đã tham gia cuộc họp hôm nay.",
}


def is_english_only(model: str | None) -> bool:
    return bool(model) and model.endswith(".en")


def multilingual_counterpart(model: str | None) -> str | None:
    """'small.en' -> 'small', 'distil-medium.en' -> 'medium'; others unchanged."""
    if not is_english_only(model):
        return model
    return model.removesuffix(".en").removeprefix("distil-")

IS_INTEL = platform.machine() in ("x86_64", "AMD64", "i386")
DEVICES = ["cpu", "mps"]  # mps = Metal GPU backend of CTranslate2 (experimental)


def physical_cores() -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, check=True
        ).stdout
        return max(1, int(out.strip()))
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class RecordSettings:
    screen_index: str = "1"
    mic_index: str = "0"
    system_audio_index: str | None = None  # e.g. BlackHole for remote participants
    height: int | None = 1080
    fps: int = 30
    encoder: str = "h264_videotoolbox"
    bitrate: str | None = None  # None -> preset for the chosen height
    capture_cursor: bool = True
    container: str = "mp4"  # final container after remux ("mp4" or "mkv")
    embed_subtitles: bool = False  # soft subtitle track (never burned into pixels)

    @property
    def effective_bitrate(self) -> str:
        return self.bitrate or _BITRATE_BY_HEIGHT.get(self.height, "4M")


@dataclass
class AsrSettings:
    final_model: str = "small"
    partial_model: str | None = "tiny"  # None -> no partials; "same" -> reuse final model
    language: str = "auto"  # "auto" (English/Vietnamese per utterance), "en" or "vi"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = field(default_factory=physical_cores)
    # Intel CPUs are ~3-5x slower than Apple Silicon for Whisper: lighter defaults.
    beam_size: int = 2 if IS_INTEL else 5
    degrade_backlog_s: float = 8.0  # drop to greedy decoding if we fall this far behind
    # Voice activity / endpointing
    vad_aggressiveness: int = 2  # webrtcvad 0..3
    end_silence_ms: int = 700  # pause that closes a sentence
    soft_max_s: float = 15.0  # after this, cut at the next short pause
    hard_max_s: float = 24.0  # never exceed (Whisper window is 30 s)
    partial_interval_s: float = 1.5 if IS_INTEL else 1.0
    min_rms: float = 0.004  # energy gate to ignore keyboard/fan noise
    remove_fillers: bool = True
    glossary: str = ""  # names / jargon to bias the decoder
    style_prompts: dict[str, str] = field(default_factory=lambda: dict(STYLE_PROMPTS))
    model_dir: Path | None = None  # cache dir for downloaded models


@dataclass
class SessionPaths:
    folder: Path
    stem: str = "meeting"

    def p(self, suffix: str) -> Path:
        return self.folder / f"{self.stem}{suffix}"

    @property
    def raw_video(self) -> Path:
        return self.p(".rec.mkv")

    @property
    def ffmpeg_log(self) -> Path:
        return self.folder / "ffmpeg.log"

    @property
    def live_transcript(self) -> Path:
        return self.p(".live.txt")
