"""Transcript storage and export (SRT / WebVTT / TXT / JSON)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .engine import Cue, FinalResult


def fmt_ts(t: float, sep: str = ",") -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def fmt_clock(t: float) -> str:
    s = max(0, int(t))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class TranscriptStore:
    """Collects final results; appends each one to a live .txt for crash safety."""

    def __init__(self, live_file: Path | None = None) -> None:
        self.cues: list[Cue] = []
        self._lock = threading.Lock()
        self._live = open(live_file, "a", encoding="utf-8") if live_file else None

    def add(self, r: FinalResult) -> None:
        if not r.cues:
            return
        with self._lock:
            self.cues.extend(r.cues)
            if self._live:
                self._live.write(f"[{fmt_clock(r.start)}] {r.text}\n")
                self._live.flush()

    def close(self) -> None:
        if self._live:
            self._live.close()
            self._live = None

    def shifted(self, offset: float) -> list[Cue]:
        with self._lock:
            cues = sorted(self.cues, key=lambda c: c.start)
        out: list[Cue] = []
        for c in cues:
            start, end = c.start + offset, c.end + offset
            if out and start < out[-1].end:  # never overlap the previous cue
                out[-1].end = max(out[-1].start + 0.1, start)
            out.append(Cue(start, max(end, start + 0.3), c.text, c.language))
        return out

    # ------------------------------------------------------------- exporters
    def write_all(self, stem: Path, offset: float = 0.0) -> dict[str, Path]:
        cues = self.shifted(offset)
        paths = {
            "srt": stem.with_suffix(".srt"),
            "vtt": stem.with_suffix(".vtt"),
            "txt": stem.with_suffix(".txt"),
            "json": stem.with_suffix(".json"),
        }
        paths["srt"].write_text(to_srt(cues), encoding="utf-8")
        paths["vtt"].write_text(to_vtt(cues), encoding="utf-8")
        paths["txt"].write_text(to_txt(cues), encoding="utf-8")
        paths["json"].write_text(
            json.dumps(
                [
                    {"start": round(c.start, 3), "end": round(c.end, 3), "text": c.text, "lang": c.language}
                    for c in cues
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return paths


def to_srt(cues: list[Cue]) -> str:
    return "".join(
        f"{i}\n{fmt_ts(c.start)} --> {fmt_ts(c.end)}\n{c.text}\n\n" for i, c in enumerate(cues, 1)
    )


def to_vtt(cues: list[Cue]) -> str:
    body = "".join(f"{fmt_ts(c.start, '.')} --> {fmt_ts(c.end, '.')}\n{c.text}\n\n" for c in cues)
    return "WEBVTT\n\n" + body


def to_txt(cues: list[Cue]) -> str:
    return "".join(f"[{fmt_clock(c.start)}] {c.text}\n" for c in cues)
