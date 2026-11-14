"""Locate ffmpeg and enumerate AVFoundation capture devices."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CANDIDATES = ["/usr/local/bin", "/opt/homebrew/bin", "/opt/local/bin", str(Path.home() / "bin")]


@lru_cache(maxsize=None)
def find_tool(name: str) -> str:
    env = os.environ.get(f"MEETINGTOOL_{name.upper()}")
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if found:
        return found
    for d in _CANDIDATES:
        p = Path(d) / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Không tìm thấy {name}. Cài bằng `brew install ffmpeg` hoặc đặt biến môi trường "
        f"MEETINGTOOL_{name.upper()}=/đường/dẫn/{name}"
    )


@dataclass
class Device:
    index: str
    name: str

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}"


_LINE = re.compile(r"\[AVFoundation[^\]]*\]\s+\[(\d+)\]\s+(.+)$")


def list_devices() -> tuple[list[Device], list[Device]]:
    """Return (screens, audio_inputs) as seen by ffmpeg's avfoundation demuxer."""
    proc = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    video, audio, section = [], [], None
    for line in proc.stderr.splitlines():
        if "video devices" in line:
            section = video
            continue
        if "audio devices" in line:
            section = audio
            continue
        m = _LINE.search(line)
        if m and section is not None:
            section.append(Device(m.group(1), m.group(2).strip()))
    screens = [d for d in video if d.name.lower().startswith("capture screen")]
    return screens, audio


def encoder_available(encoder: str) -> bool:
    try:
        out = subprocess.run(
            [find_tool("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return False
    return f" {encoder} " in out
