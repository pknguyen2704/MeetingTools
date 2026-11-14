"""End-to-end: synthetic screen + speech through the real ffmpeg recorder and ASR.

Checks that the recording is produced, scaled, and that transcript timestamps
line up with speech onsets measured on the *recorded* audio track.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from meetingtool.config import AsrSettings, RecordSettings
from meetingtool.session import MeetingSession

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("say")), reason="needs ffmpeg + macOS say"
)

SENTENCES = [
    "Good morning everyone, thanks for joining the weekly sync.",
    "First, the release is scheduled for next Tuesday.",
    "Does anyone have questions about the budget?",
]


def _make_speech(path: Path) -> None:
    text = "[[slnc 2000]] " + " [[slnc 1500]] ".join(SENTENCES)
    subprocess.run(["say", "-o", str(path), text], check=True)


def _speech_onsets(media: Path) -> list[float]:
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(media), "-map", "0:a", "-af",
         "silencedetect=noise=-40dB:d=0.6", "-f", "null", "-"],
        capture_output=True, text=True,
    )  # fmt: skip
    return [float(x) for x in re.findall(r"silence_end: ([\d.]+)", res.stderr)]


def test_record_and_transcribe_in_sync(tmp_path: Path) -> None:
    speech = tmp_path / "speech.aiff"
    _make_speech(speech)
    dur = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(speech)],
            capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    graph = f"testsrc2=s=2880x1800:r=30[out0];amovie={speech},aresample=48000,apad[out1]"
    inputs = ["-re", "-f", "lavfi", "-i", graph]

    rec = RecordSettings(height=720, fps=30, encoder="h264_videotoolbox", container="mp4")
    asr = AsrSettings(final_model="base.en", partial_model="tiny.en", beam_size=2)
    partials, finals = [], []
    sess = MeetingSession(
        rec, asr, tmp_path / "out", on_partial=partials.append, on_final=finals.append, input_args=inputs
    )
    sess.start()
    time.sleep(dur + 1.5)
    res = sess.stop()

    assert res.error is None, res.error
    assert res.video and res.video.suffix == ".mp4" and res.video.exists()
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(res.video)],
            capture_output=True, text=True, check=True,
        ).stdout
    )  # fmt: skip
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert v["codec_name"] == "h264" and int(v["height"]) == 720
    assert any(s["codec_type"] == "audio" for s in probe["streams"])

    texts = [f.text for f in finals if f.text]
    assert len(texts) == len(SENTENCES), texts
    for t in texts:
        assert t[0].isupper() and t[-1] in ".?!", t
    assert "?" in texts[-1]
    assert partials, "expected live partial hypotheses"

    # Sync: cue starts (from SRT on disk) vs speech onsets measured in the video file.
    cues = json.loads(res.transcripts["json"].read_text())
    onsets = _speech_onsets(res.video)
    assert len(onsets) >= len(SENTENCES)
    print("\nsync deltas (cue.start - onset):", [round(c["start"] - o, 3) for c, o in zip(cues, onsets)])
    print("texts:", texts)
    for cue, onset in zip(cues, onsets):
        # VAD pre-roll starts a cue up to ~0.3 s before the onset; allow small jitter.
        assert -0.45 <= cue["start"] - onset <= 0.25, (cue, onset)
