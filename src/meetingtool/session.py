"""One recording session = ffmpeg recorder + transcription engine + exports."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import AsrSettings, RecordSettings, SessionPaths
from .engine import FinalResult, PartialResult, TranscriptionEngine
from .recorder import Recorder, probe_audio_offset, remux
from .transcript import TranscriptStore

log = logging.getLogger(__name__)


@dataclass
class SessionResult:
    folder: Path
    video: Path | None
    transcripts: dict[str, Path] = field(default_factory=dict)
    offset: float = 0.0
    error: str | None = None


class MeetingSession:
    def __init__(
        self,
        rec: RecordSettings,
        asr: AsrSettings,
        out_root: Path,
        *,
        on_partial: Callable[[PartialResult], None] | None = None,
        on_final: Callable[[FinalResult], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_recorder_died: Callable[[str], None] | None = None,
        input_args: list[str] | None = None,
    ) -> None:
        self.rec = rec
        self.asr = asr
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.paths = SessionPaths(Path(out_root).expanduser() / f"Meeting_{stamp}", stem=f"meeting_{stamp}")
        self.paths.folder.mkdir(parents=True, exist_ok=True)
        self.store = TranscriptStore(self.paths.live_transcript)
        self._on_final = on_final or (lambda r: None)
        self._on_recorder_died = on_recorder_died or (lambda s: None)
        self.engine = TranscriptionEngine(
            asr,
            on_partial=on_partial,
            on_final=self._handle_final,
            on_status=on_status,
            on_error=on_error,
        )
        self.recorder = Recorder(
            rec,
            self.paths.raw_video,
            self.paths.ffmpeg_log,
            on_audio=self.engine.feed,
            on_eof=self._handle_eof,
            input_args=input_args,
        )

    def _handle_final(self, r: FinalResult) -> None:
        self.store.add(r)
        self._on_final(r)

    def _handle_eof(self) -> None:
        self.engine.finish()
        if not self.recorder.stopping:
            self._on_recorder_died(self.recorder.log_tail())

    @property
    def elapsed(self) -> float:
        return self.recorder.elapsed

    def start(self) -> None:
        self.engine.start()
        self.recorder.start()

    def stop(self, on_status: Callable[[str], None] | None = None) -> SessionResult:
        """Blocking: stop capture, finish pending transcription, write files."""
        say = on_status or (lambda s: None)
        say("Đang dừng ghi hình…")
        self.recorder.stop()
        say("Đang hoàn tất chép lời các câu cuối…")
        self.engine.join()
        self.store.close()

        raw = self.paths.raw_video
        result = SessionResult(self.paths.folder, None)
        if not raw.exists() or raw.stat().st_size == 0:
            result.error = "Không có file ghi hình. Xem ffmpeg.log:\n" + self.recorder.log_tail()
            result.transcripts = self.store.write_all(self.paths.p(".srt"))
            return result

        result.offset = probe_audio_offset(raw)
        result.transcripts = self.store.write_all(self.paths.p(".srt"), offset=result.offset)

        if self.rec.container == "mp4" or self.rec.embed_subtitles:
            say("Đang đóng gói video…")
            dst = self.paths.p(f".{self.rec.container}")
            subs = result.transcripts["srt"] if self.rec.embed_subtitles and self.store.cues else None
            if remux(raw, dst, subs):
                raw.unlink(missing_ok=True)
                result.video = dst
            else:
                result.video = raw
        else:
            dst = self.paths.p(".mkv")
            raw.rename(dst)
            result.video = dst
        say("Đã lưu")
        return result
