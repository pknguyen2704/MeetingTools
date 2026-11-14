"""Screen + audio recorder built on a single ffmpeg process.

Design notes (performance and A/V sync):

* Screen and microphone are opened in **one** avfoundation input, so they share
  the same capture clock. An optional second audio device (e.g. BlackHole for
  the remote participants' voices) is mixed in with ``amix``.
* Video is encoded with VideoToolbox (Intel Quick Sync / Apple media engine), so
  the CPU is left for speech recognition. Down-scaling is optional.
* The mixed audio is split in the filter graph: one branch goes to the video
  file (AAC), the other is resampled to 16 kHz mono PCM and streamed to stdout
  for the transcriber. ``aresample=async:first_pts=0`` pads/stretches so the
  sample count of the PCM tap *is* the timeline of the recorded audio track —
  transcript timestamps therefore match the video without any drift.
* Recording goes to Matroska first (survives crashes), then is remuxed (copy,
  no re-encode) to MP4 when stopped.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE, RecordSettings
from .devices import find_tool

log = logging.getLogger(__name__)

CHUNK_SAMPLES = SAMPLE_RATE // 10  # 100 ms per read


def build_filtergraph(s: RecordSettings) -> str:
    if s.height:
        v = f"[0:v]scale=-2:'min({s.height},ih)':flags=fast_bilinear,format=nv12[v]"
    else:
        v = "[0:v]format=nv12[v]"
    if s.system_audio_index is not None:
        a = "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,"
    else:
        a = "[0:a]"
    a += "aresample=async=1000:first_pts=0,asplit=2[arec][atap]"
    tap = "[atap]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[asr]"
    return ";".join([v, a, tap])


def build_command(s: RecordSettings, out_file: Path, input_args: list[str] | None = None) -> list[str]:
    """``input_args`` replaces the avfoundation inputs (used by tests: lavfi sources)."""
    ff = find_tool("ffmpeg")
    cmd = [ff, "-hide_banner", "-nostats", "-loglevel", "warning", "-y"]
    if input_args is not None:
        cmd += input_args
    else:
        cmd += _avfoundation_inputs(s)
    cmd += ["-filter_complex", build_filtergraph(s)]
    cmd += _output_args(s, out_file)
    return cmd


def _avfoundation_inputs(s: RecordSettings) -> list[str]:
    cmd = [
        "-f", "avfoundation",
        "-capture_cursor", "1" if s.capture_cursor else "0",
        "-pixel_format", "nv12",
        "-framerate", str(s.fps),
        "-thread_queue_size", "4096",
        "-i", f"{s.screen_index}:{s.mic_index}",
    ]  # fmt: skip
    if s.system_audio_index is not None:
        cmd += ["-f", "avfoundation", "-thread_queue_size", "4096", "-i", f":{s.system_audio_index}"]
    return cmd


def _output_args(s: RecordSettings, out_file: Path) -> list[str]:
    # Output 1: the recording
    cmd = ["-map", "[v]", "-map", "[arec]", "-c:v", s.encoder]
    br = s.effective_bitrate
    if s.encoder.endswith("_videotoolbox"):
        cmd += ["-realtime", "1", "-allow_sw", "1", "-b:v", br, "-maxrate", br, "-bufsize", br]
        if s.encoder.startswith("hevc"):
            cmd += ["-tag:v", "hvc1"]
    else:  # libx264 fallback, tuned for low CPU
        cmd += ["-preset", "veryfast", "-tune", "zerolatency", "-crf", "26", "-maxrate", br, "-bufsize", br]
    cmd += ["-g", str(s.fps * 2), "-fps_mode", "passthrough"]
    cmd += ["-c:a", "aac", "-b:a", "128k", str(out_file)]

    # Output 2: 16 kHz mono PCM tap for the transcriber (flushed per packet => low latency)
    cmd += ["-map", "[asr]", "-c:a", "pcm_s16le", "-f", "s16le", "-flush_packets", "1", "pipe:1"]
    return cmd


class Recorder:
    def __init__(
        self,
        settings: RecordSettings,
        out_file: Path,
        log_file: Path,
        on_audio: Callable[[np.ndarray], None],
        on_eof: Callable[[], None] | None = None,
        input_args: list[str] | None = None,
    ) -> None:
        self.input_args = input_args
        self.settings = settings
        self.out_file = out_file
        self.log_file = log_file
        self.on_audio = on_audio
        self.on_eof = on_eof or (lambda: None)
        self.proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._log_fh = None
        self.stopping = False
        self.samples = 0

    @property
    def elapsed(self) -> float:
        """Seconds of audio captured (== position on the recording timeline)."""
        return self.samples / SAMPLE_RATE

    def start(self) -> None:
        cmd = build_command(self.settings, self.out_file, self.input_args)
        log.info("ffmpeg: %s", subprocess.list2cmdline(cmd))
        self._log_fh = open(self.log_file, "wb")
        self._log_fh.write((subprocess.list2cmdline(cmd) + "\n\n").encode())
        self._log_fh.flush()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_fh,
            bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_pcm, name="pcm-reader", daemon=True)
        self._reader.start()

    def _read_pcm(self) -> None:
        assert self.proc and self.proc.stdout
        nbytes = CHUNK_SAMPLES * 2
        buf = b""
        try:
            while True:
                data = self.proc.stdout.read(nbytes)
                if not data:
                    break
                buf += data
                usable = len(buf) - (len(buf) % 2)
                if not usable:
                    continue
                pcm = np.frombuffer(buf[:usable], dtype="<i2").astype(np.float32) / 32768.0
                buf = buf[usable:]
                self.samples += len(pcm)
                self.on_audio(pcm)
        finally:
            self.on_eof()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 15.0) -> int:
        """Gracefully stop ffmpeg (sends 'q') and wait for the file to be finalised."""
        self.stopping = True
        p = self.proc
        if p is None:
            return -1
        if p.poll() is None:
            try:
                p.stdin.write(b"q")
                p.stdin.flush()
                p.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                p.wait(timeout)
            except subprocess.TimeoutExpired:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        if self._reader:
            self._reader.join(5)
        if self._log_fh:
            self._log_fh.close()
        return p.returncode

    def log_tail(self, n: int = 15) -> str:
        try:
            return "\n".join(self.log_file.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""


# ---------------------------------------------------------------- post-process
def probe_audio_offset(path: Path) -> float:
    """Start of the audio track relative to the file's presentation start.

    Transcript times are measured on the audio track, so this offset is added to
    every cue to land exactly on the video timeline (normally ~0 thanks to
    ``first_pts=0``; non-zero only if the muxer shifted streams).
    """
    try:
        out = subprocess.run(
            [
                find_tool("ffprobe"), "-v", "error",
                "-show_entries", "format=start_time:stream=codec_type,start_time",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        info = json.loads(out)
        fmt_start = float(info.get("format", {}).get("start_time", 0) or 0)
        for st in info.get("streams", []):
            if st.get("codec_type") == "audio":
                return max(0.0, float(st.get("start_time", 0) or 0) - fmt_start)
    except Exception as exc:  # noqa: BLE001
        log.warning("ffprobe failed: %s", exc)
    return 0.0


def remux(src: Path, dst: Path, subtitles: Path | None = None) -> bool:
    """Copy streams into the final container (no re-encode). Optional soft subtitle track."""
    cmd = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if subtitles:
        cmd += ["-i", str(subtitles)]
    cmd += ["-map", "0:v", "-map", "0:a"]
    if subtitles:
        cmd += ["-map", "1:s"]
    cmd += ["-c", "copy"]
    if subtitles:
        cmd += ["-c:s", "mov_text" if dst.suffix == ".mp4" else "srt",
                "-metadata:s:s:0", "language=eng", "-disposition:s:0", "0"]  # fmt: skip
    if dst.suffix == ".mp4":
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(dst))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("remux failed: %s", res.stderr)
    return res.returncode == 0 and dst.exists()
