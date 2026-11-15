"""Command line entry point.

    meetingtool                 # desktop app (default)
    meetingtool devices         # list screens / audio inputs
    meetingtool record ...      # headless recording + live transcript in the terminal
    meetingtool transcribe F    # stream a media file through the realtime pipeline
    meetingtool bench           # measure model speed on this machine
"""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .config import FINAL_MODELS, RESOLUTIONS, SAMPLE_RATE, AsrSettings, RecordSettings, is_english_only
from .devices import find_tool, list_devices


def _asr_from_args(a: argparse.Namespace) -> AsrSettings:
    cfg = AsrSettings(
        final_model=a.model, beam_size=a.beam, glossary=a.glossary or "", device=a.device, language=a.language
    )
    cfg.partial_model = None if a.partial_model == "none" else a.partial_model
    if a.language != "en" and is_english_only(a.model):
        print(f"Cảnh báo: {a.model} chỉ chép tiếng Anh; dùng --model small để chép cả tiếng Việt.", file=sys.stderr)
    if a.threads:
        cfg.cpu_threads = a.threads
    return cfg


class _TermPrinter:
    """Shows the partial hypothesis on one rewritable line, finals as permanent lines."""

    def __init__(self, show_partials: bool = True) -> None:
        self.show_partials = show_partials and sys.stdout.isatty()
        self.lock = threading.Lock()
        self.width = shutil.get_terminal_size((100, 20)).columns

    def partial(self, r) -> None:
        if not self.show_partials:
            return
        from .transcript import fmt_clock

        line = f"[{fmt_clock(r.start)}] … {r.text}"
        if len(line) > self.width - 1:
            line = "…" + line[-(self.width - 2) :]
        with self.lock:
            sys.stdout.write("\r\033[2K\033[90m" + line + "\033[0m")
            sys.stdout.flush()

    def final(self, r) -> None:
        from .transcript import fmt_clock

        with self.lock:
            if self.show_partials:
                sys.stdout.write("\r\033[2K")
            if r.text:
                lat = f"  \033[90m(asr {r.asr_seconds:.2f}s" + (
                    f", trễ {r.latency:.2f}s)\033[0m" if r.latency is not None else ")\033[0m"
                )
                sys.stdout.write(f"[{fmt_clock(r.start)}] {r.text}{lat}\n")
            sys.stdout.flush()


# --------------------------------------------------------------------- commands
def cmd_devices(_: argparse.Namespace) -> int:
    screens, audio = list_devices()
    print("Màn hình (--screen):")
    for d in screens:
        print(f"  {d}")
    print("Âm thanh (--mic / --sys):")
    for d in audio:
        print(f"  {d}")
    return 0


def cmd_record(a: argparse.Namespace) -> int:
    from .session import MeetingSession

    rec = RecordSettings(
        screen_index=a.screen,
        mic_index=a.mic,
        system_audio_index=a.sys,
        height=RESOLUTIONS.get(a.res, None) if a.res != "native" else None,
        fps=a.fps,
        encoder=a.encoder,
        container=a.container,
        embed_subtitles=a.embed_subs,
    )
    printer = _TermPrinter()
    died = threading.Event()
    sess = MeetingSession(
        rec,
        _asr_from_args(a),
        Path(a.out),
        on_partial=printer.partial,
        on_final=printer.final,
        on_status=lambda s: print(f"\033[90m· {s}\033[0m"),
        on_error=lambda s: print(f"LỖI: {s}", file=sys.stderr),
        on_recorder_died=lambda tail: (print("ffmpeg dừng bất ngờ:\n" + tail, file=sys.stderr), died.set()),
    )
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    sess.start()
    print(f"Đang ghi → {sess.paths.folder}  (Ctrl+C để dừng)")
    t0 = time.monotonic()
    while not stop.is_set() and not died.is_set():
        if a.duration and time.monotonic() - t0 >= a.duration:
            break
        stop.wait(0.2)
    res = sess.stop(on_status=lambda s: print(f"\033[90m· {s}\033[0m"))
    print(f"Video: {res.video}")
    for k, p in res.transcripts.items():
        print(f"{k.upper():5}: {p}")
    if res.offset:
        print(f"Bù lệch audio/video: {res.offset * 1000:.1f} ms")
    if res.error:
        print(res.error, file=sys.stderr)
        return 1
    return 0


def stream_file(path: Path, cfg: AsrSettings, *, realtime: bool, printer: _TermPrinter | None):
    """Feed a media file through the realtime engine exactly as the recorder would."""
    from .engine import TranscriptionEngine
    from .transcript import TranscriptStore

    store = TranscriptStore()
    finals = []

    def on_final(r):
        store.add(r)
        finals.append(r)
        if printer:
            printer.final(r)

    eng = TranscriptionEngine(
        cfg,
        on_partial=printer.partial if printer else None,
        on_final=on_final,
        on_error=lambda s: print(f"LỖI: {s}", file=sys.stderr),
    )
    eng.start()
    eng.ready.wait()
    proc = subprocess.Popen(
        [find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "pipe:1"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    chunk = SAMPLE_RATE // 10
    fed = 0
    t_start = time.monotonic()
    while True:
        data = proc.stdout.read(chunk * 2)
        if not data:
            break
        pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        eng.feed(pcm)
        fed += len(pcm)
        if realtime:
            ahead = fed / SAMPLE_RATE - (time.monotonic() - t_start)
            if ahead > 0:
                time.sleep(ahead)
    proc.wait()
    eng.finish()
    eng.join()
    wall = time.monotonic() - t_start
    return store, finals, eng, fed / SAMPLE_RATE, wall


def cmd_transcribe(a: argparse.Namespace) -> int:
    cfg = _asr_from_args(a)
    path = Path(a.file)
    store, finals, eng, dur, wall = stream_file(path, cfg, realtime=a.realtime, printer=_TermPrinter())
    stem = Path(a.out) / path.stem if a.out else path.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = store.write_all(stem.with_suffix(".srt"))
    print(
        f"\nAudio {dur:.1f}s · thời gian ASR {eng.asr_time:.1f}s (RTF {eng.asr_time / max(dur, 1e-6):.2f})"
        f" · wall {wall:.1f}s · {len(finals)} câu"
    )
    if eng.latencies:
        lat = np.array(eng.latencies)
        print(f"Độ trễ từ lúc ngừng nói → câu chuẩn: TB {lat.mean():.2f}s, p90 {np.percentile(lat, 90):.2f}s")
    print("Đã ghi:", ", ".join(str(p) for p in paths.values()))
    return 0


_BENCH_TEXT_VI = (
    "Chào buổi sáng mọi người, cảm ơn đã tham gia. Hôm nay chúng ta sẽ xem lại kế hoạch quý. "
    "Thứ nhất, bản phát hành di động bị chậm hai tuần vì phần tích hợp thanh toán. "
    "Thứ hai, chúng ta cần quyết định chuyển cơ sở dữ liệu trước hay sau khi ra mắt. "
    "Mọi người có lo ngại gì về tiến độ không? "
    "Được rồi, chúng ta chuyển sang kế hoạch tuyển dụng cho quý tới."
)

_BENCH_TEXT = (
    "Good morning everyone, thanks for joining. Today we will review the quarterly roadmap. "
    "First, the mobile release slipped by two weeks because of the payment integration. "
    "Second, we need to decide whether to migrate the database before or after the launch. "
    "Does anyone have concerns about the timeline? "
    "Okay, let's move on to the hiring plan for the next quarter."
)


def cmd_bench(a: argparse.Namespace) -> int:
    if a.file:
        sample = Path(a.file)
    else:
        if not shutil.which("say"):
            print("Cần --file trên hệ điều hành không có lệnh `say`.", file=sys.stderr)
            return 2
        tmp = Path(tempfile.mkdtemp())
        sample = tmp / "bench.aiff"
        # Pauses between sentences so the segmenter sees separate utterances.
        vi = a.language == "vi"
        base = _BENCH_TEXT_VI if vi else _BENCH_TEXT
        text = base.replace(". ", ". [[slnc 900]] ").replace("? ", "? [[slnc 900]] ")
        voice = ["-v", "Linh"] if vi else []  # macOS Vietnamese voice (System Settings → Spoken Content)
        subprocess.run(["say", *voice, "-o", str(sample), text], check=True)
    if a.models:
        models = a.models.split(",")
    elif a.language == "en":
        models = ["base.en", "distil-small.en", "small.en"]
    else:
        models = ["base", "small"]
    print(f"Mẫu: {sample}  ·  cpu_threads={a.threads or AsrSettings().cpu_threads}\n")
    rows = []
    for m in models:
        partial = None if a.no_partial else ("tiny.en" if is_english_only(m) else "tiny")
        cfg = AsrSettings(
            final_model=m, partial_model=partial, beam_size=a.beam, device=a.device, language=a.language
        )
        if a.threads:
            cfg.cpu_threads = a.threads
        from .engine import ModelPool

        t = time.perf_counter()
        ModelPool.get(m, cfg)
        load = time.perf_counter() - t
        _, finals, eng, dur, _ = stream_file(sample, cfg, realtime=False, printer=None)
        per = [f.asr_seconds for f in finals if f.text]
        text = " ".join(f.text for f in finals)
        rows.append((m, load, eng.asr_time / dur, max(per) if per else 0.0))
        print(f"== {m}: {text}\n")
    print(f"{'model':18} {'load(s)':>8} {'RTF*':>6} {'câu chậm nhất(s)':>17}")
    for m, load, rtf, worst in rows:
        print(f"{m:18} {load:8.1f} {rtf:6.2f} {worst:17.2f}")
    print("\n*RTF gồm cả bản nháp; < 0.5 là thoải mái chạy realtime song song với ghi hình.")
    return 0


# ------------------------------------------------------------------------ main
def _add_language_arg(p: argparse.ArgumentParser, default: str = "auto") -> None:
    p.add_argument(
        "--language", default=default, choices=["auto", "en", "vi"],
        help="auto = tự nhận diện Anh/Việt từng câu (mặc định); en; vi",
    )  # fmt: skip


def _add_asr_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="small", help=f"mô hình chính ({', '.join(FINAL_MODELS)})")
    p.add_argument("--partial-model", default="tiny", help="mô hình bản nháp, 'same' hoặc 'none'")
    _add_language_arg(p)
    p.add_argument("--beam", type=int, default=AsrSettings().beam_size)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="mps = GPU Metal (thử nghiệm)")
    p.add_argument("--threads", type=int, default=0, help="số luồng CPU (mặc định = số nhân vật lý)")
    p.add_argument("--glossary", default="", help="tên riêng / thuật ngữ")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="meetingtool", description="Ghi hình cuộc họp + chép lời realtime")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="mở giao diện (mặc định)")
    sub.add_parser("devices", help="liệt kê thiết bị ghi")

    p = sub.add_parser("record", help="ghi hình không giao diện")
    p.add_argument("--screen", required=True, help="chỉ số màn hình (xem `devices`)")
    p.add_argument("--mic", required=True, help="chỉ số micro")
    p.add_argument("--sys", default=None, help="chỉ số thiết bị âm thanh hệ thống (BlackHole)")
    p.add_argument("--res", default="1080p", choices=["native", *[k for k in RESOLUTIONS if k[0].isdigit()]])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--encoder", default="h264_videotoolbox")
    p.add_argument("--container", default="mp4", choices=["mp4", "mkv"])
    p.add_argument("--embed-subs", action="store_true")
    p.add_argument("--duration", type=float, default=0, help="tự dừng sau N giây")
    p.add_argument("--out", default=str(Path.home() / "Movies" / "MeetingTool"))
    _add_asr_args(p)

    p = sub.add_parser("transcribe", help="chạy pipeline realtime trên một file audio/video")
    p.add_argument("file")
    p.add_argument("--realtime", action="store_true", help="đẩy audio theo tốc độ thật")
    p.add_argument("--out", default=None)
    _add_asr_args(p)

    p = sub.add_parser("bench", help="đo tốc độ các mô hình trên máy này")
    p.add_argument("--models", default=None, help="vd: base,small hoặc base.en,small.en")
    _add_language_arg(p, default="en")
    p.add_argument("--file", default=None)
    p.add_argument("--beam", type=int, default=AsrSettings().beam_size)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-partial", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd in (None, "gui"):
        from .ui import run_gui

        return run_gui()
    return {"devices": cmd_devices, "record": cmd_record, "transcribe": cmd_transcribe, "bench": cmd_bench}[
        a.cmd
    ](a)


if __name__ == "__main__":
    sys.exit(main())
