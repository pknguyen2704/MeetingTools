"""PySide6 desktop UI."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import macos
from .config import (
    DEVICES,
    FINAL_MODELS,
    PARTIAL_MODELS,
    RESOLUTIONS,
    VIDEO_ENCODERS,
    AsrSettings,
    RecordSettings,
)
from .devices import encoder_available, list_devices
from .engine import FinalResult, ModelPool, PartialResult
from .session import MeetingSession, SessionResult
from .transcript import fmt_clock

log = logging.getLogger(__name__)

NO_DEVICE = "(không dùng)"


class Bridge(QObject):
    """Marshals callbacks from worker threads onto the Qt GUI thread."""

    partial = Signal(object)
    final = Signal(object)
    status = Signal(str)
    error = Signal(str)
    died = Signal(str)
    finished = Signal(object)


class TranscriptView(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        f = self.font()
        f.setPointSize(15)
        self.setFont(f)
        self._partial_start: int | None = None
        self._has_content = False
        self.ts_fmt = QTextCharFormat()
        self.ts_fmt.setForeground(QColor("#3b82f6"))
        self.ts_fmt.setFontFamilies(["Menlo", "Monaco", "monospace"])
        self.ts_fmt.setFontPointSize(12)
        self.text_fmt = QTextCharFormat()
        self.text_fmt.setFontPointSize(15)
        self.partial_fmt = QTextCharFormat()
        self.partial_fmt.setForeground(QColor("#8a8a8a"))
        self.partial_fmt.setFontItalic(True)
        self.partial_fmt.setFontPointSize(15)

    def clear_all(self) -> None:
        self.clear()
        self._partial_start = None
        self._has_content = False

    def _at_bottom(self) -> bool:
        sb = self.verticalScrollBar()
        return sb.value() >= sb.maximum() - 8

    def _remove_partial(self) -> None:
        if self._partial_start is None:
            return
        cur = QTextCursor(self.document())
        cur.setPosition(self._partial_start)
        cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cur.removeSelectedText()
        self._partial_start = None

    def _append(self, start: float, text: str, fmt: QTextCharFormat) -> int:
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        pos = cur.position()
        if self._has_content:
            cur.insertBlock()
        cur.insertText(f"[{fmt_clock(start)}]  ", self.ts_fmt)
        cur.insertText(text, fmt)
        return pos

    def set_partial(self, r: PartialResult) -> None:
        stick = self._at_bottom()
        self._remove_partial()
        had = self._has_content
        self._partial_start = self._append(r.start, r.text + " …", self.partial_fmt)
        self._has_content = had  # a partial does not count as committed content
        if stick:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def add_final(self, r: FinalResult) -> None:
        stick = self._at_bottom()
        self._remove_partial()
        if r.text:
            self._append(r.start, r.text, self.text_fmt)
            self._has_content = True
        if stick:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MeetingTool – Ghi hình & chép lời cuộc họp")
        self.resize(980, 760)
        self.settings = QSettings("MeetingTool", "MeetingTool")
        self.bridge = Bridge()
        self.session: MeetingSession | None = None
        self._stopping = False
        self._last_result: SessionResult | None = None
        self._latencies: list[float] = []

        self._build_ui()
        self._wire()
        self.refresh_devices()
        self._restore()
        self._preload_models()

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        box = QGroupBox("Cấu hình")
        grid = QGridLayout(box)
        left, right = QFormLayout(), QFormLayout()
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)

        self.cb_screen = QComboBox()
        self.cb_mic = QComboBox()
        self.cb_sys = QComboBox()
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(36)
        self.btn_refresh.setToolTip("Quét lại thiết bị")
        row = QHBoxLayout()
        row.addWidget(self.cb_screen, 1)
        row.addWidget(self.btn_refresh)
        left.addRow("Màn hình:", row)
        left.addRow("Micro:", self.cb_mic)
        left.addRow("Âm thanh hệ thống:", self.cb_sys)
        self.cb_sys.setToolTip(
            "Tiếng của người họp từ xa (Zoom/Meet/Teams). Cần BlackHole – xem README."
        )

        self.cb_res = QComboBox()
        self.cb_res.addItems(list(RESOLUTIONS))
        self.cb_fps = QComboBox()
        self.cb_fps.addItems(["10", "15", "24", "30", "60"])
        self.cb_enc = QComboBox()
        self.cb_enc.addItems(list(VIDEO_ENCODERS))
        self.cb_container = QComboBox()
        self.cb_container.addItems(["mp4", "mkv"])
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(self.cb_res, 2)
        fmt_row.addWidget(QLabel("FPS"))
        fmt_row.addWidget(self.cb_fps, 1)
        fmt_row.addWidget(self.cb_container, 1)
        left.addRow("Độ phân giải:", fmt_row)
        left.addRow("Bộ mã hóa:", self.cb_enc)

        self.cb_model = QComboBox()
        self.cb_model.addItems(FINAL_MODELS)
        self.cb_model.setToolTip(
            "Mô hình tạo câu chuẩn (có dấu câu). small.en: chính xác nhất mà vẫn realtime trên i7 2017."
        )
        self.cb_partial = QComboBox()
        self.cb_partial.addItems(PARTIAL_MODELS)
        self.sp_beam = QSpinBox()
        self.sp_beam.setRange(1, 5)
        self.sp_beam.setValue(AsrSettings().beam_size)
        mrow = QHBoxLayout()
        mrow.addWidget(self.cb_model, 2)
        mrow.addWidget(QLabel("beam"))
        mrow.addWidget(self.sp_beam)
        right.addRow("Mô hình chép lời:", mrow)
        right.addRow("Bản nháp realtime:", self.cb_partial)
        self.ed_glossary = QLineEdit()
        self.ed_glossary.setPlaceholderText("Tên riêng, thuật ngữ… (vd: Kubernetes, OKR, Nguyen)")
        right.addRow("Từ khóa:", self.ed_glossary)
        self.ed_out = QLineEdit(str(Path.home() / "Movies" / "MeetingTool"))
        self.btn_out = QPushButton("…")
        self.btn_out.setFixedWidth(36)
        orow = QHBoxLayout()
        orow.addWidget(self.ed_out, 1)
        orow.addWidget(self.btn_out)
        right.addRow("Thư mục lưu:", orow)

        self.ck_hide = QCheckBox("Ẩn cửa sổ này khỏi video")
        self.ck_hide.setChecked(True)
        self.ck_embed = QCheckBox("Nhúng phụ đề mềm vào video")
        self.ck_cursor = QCheckBox("Quay con trỏ chuột")
        self.ck_cursor.setChecked(True)
        self.ck_top = QCheckBox("Luôn ở trên cùng")
        ckrow = QHBoxLayout()
        for w in (self.ck_hide, self.ck_embed, self.ck_cursor, self.ck_top):
            ckrow.addWidget(w)
        ckrow.addStretch(1)
        grid.addLayout(ckrow, 1, 0, 1, 2)
        grid.addWidget(self._build_advanced(), 2, 0, 1, 2)
        root.addWidget(box)
        self.config_box = box

        ctl = QHBoxLayout()
        self.btn_start = QPushButton("●  Bắt đầu ghi")
        self.btn_start.setMinimumHeight(40)
        f = self.btn_start.font()
        f.setPointSize(14)
        f.setBold(True)
        self.btn_start.setFont(f)
        self.lbl_time = QLabel("00:00:00")
        tf = QFont("Menlo")
        tf.setPointSize(20)
        self.lbl_time.setFont(tf)
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: gray")
        self.btn_open = QPushButton("Mở thư mục")
        self.btn_open.setEnabled(False)
        ctl.addWidget(self.btn_start, 2)
        ctl.addSpacing(12)
        ctl.addWidget(self.lbl_time)
        ctl.addSpacing(12)
        ctl.addWidget(self.lbl_stats, 3)
        ctl.addWidget(self.btn_open)
        root.addLayout(ctl)

        self.view = TranscriptView()
        root.addWidget(self.view, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Sẵn sàng")

    def _build_advanced(self) -> QGroupBox:
        d = AsrSettings()
        adv = QGroupBox("Nâng cao")
        adv.setCheckable(True)
        adv.setChecked(False)
        inner = QWidget()
        inner.setVisible(False)
        adv.toggled.connect(inner.setVisible)
        QVBoxLayout(adv).addWidget(inner)
        g = QGridLayout(inner)
        g.setContentsMargins(0, 0, 0, 0)
        left, right = QFormLayout(), QFormLayout()
        g.addLayout(left, 0, 0)
        g.addLayout(right, 0, 1)

        self.cb_bitrate = QComboBox()
        self.cb_bitrate.setEditable(True)
        self.cb_bitrate.addItems(["Tự động", "1200k", "2500k", "4M", "6M", "10M"])
        self.cb_bitrate.setToolTip("Bitrate video. 'Tự động' chọn theo độ phân giải.")
        left.addRow("Bitrate video:", self.cb_bitrate)
        self.cb_device = QComboBox()
        self.cb_device.addItems(DEVICES)
        self.cb_device.setToolTip("mps = GPU qua Metal (thử nghiệm). Lỗi sẽ tự quay về CPU.")
        left.addRow("Thiết bị suy luận:", self.cb_device)
        self.sp_threads = QSpinBox()
        self.sp_threads.setRange(1, os.cpu_count() or 8)
        self.sp_threads.setValue(d.cpu_threads)
        self.sp_threads.setToolTip("Mặc định = số nhân vật lý (i7 2017: 4).")
        left.addRow("Số luồng CPU:", self.sp_threads)
        self.ck_fillers = QCheckBox("Bỏ từ đệm (um, uh, erm…)")
        self.ck_fillers.setChecked(d.remove_fillers)
        left.addRow("", self.ck_fillers)

        self.sp_silence = QSpinBox()
        self.sp_silence.setRange(300, 2000)
        self.sp_silence.setSingleStep(100)
        self.sp_silence.setSuffix(" ms")
        self.sp_silence.setValue(d.end_silence_ms)
        self.sp_silence.setToolTip("Người nói dừng bao lâu thì chốt câu và chuẩn hóa.")
        right.addRow("Khoảng lặng chốt câu:", self.sp_silence)
        self.sp_maxlen = QDoubleSpinBox()
        self.sp_maxlen.setRange(5, 20)
        self.sp_maxlen.setSuffix(" s")
        self.sp_maxlen.setValue(d.soft_max_s)
        self.sp_maxlen.setToolTip("Câu dài hơn sẽ được cắt ở lần ngắt nghỉ ngắn tiếp theo.")
        right.addRow("Độ dài câu tối đa:", self.sp_maxlen)
        self.sp_partial = QDoubleSpinBox()
        self.sp_partial.setRange(0.5, 5)
        self.sp_partial.setSingleStep(0.5)
        self.sp_partial.setSuffix(" s")
        self.sp_partial.setValue(d.partial_interval_s)
        right.addRow("Cập nhật bản nháp mỗi:", self.sp_partial)
        self.sp_vad = QSpinBox()
        self.sp_vad.setRange(0, 3)
        self.sp_vad.setValue(d.vad_aggressiveness)
        self.sp_vad.setToolTip("0 = nhạy nhất, 3 = lọc ồn mạnh nhất.")
        self.sp_rms = QDoubleSpinBox()
        self.sp_rms.setDecimals(3)
        self.sp_rms.setRange(0, 0.05)
        self.sp_rms.setSingleStep(0.001)
        self.sp_rms.setValue(d.min_rms)
        self.sp_rms.setToolTip("Âm lượng tối thiểu được coi là giọng nói (lọc tiếng gõ phím, quạt).")
        vrow = QHBoxLayout()
        vrow.addWidget(self.sp_vad)
        vrow.addWidget(QLabel("ngưỡng âm lượng"))
        vrow.addWidget(self.sp_rms)
        right.addRow("Lọc ồn (VAD):", vrow)

        btn_reset = QPushButton("Khôi phục mặc định")
        btn_reset.clicked.connect(self._reset_advanced)
        g.addWidget(btn_reset, 1, 1, Qt.AlignmentFlag.AlignRight)
        self.adv_box = adv
        return adv

    def _reset_advanced(self) -> None:
        d = AsrSettings()
        self.cb_bitrate.setCurrentText("Tự động")
        self.cb_device.setCurrentText(d.device)
        self.sp_threads.setValue(d.cpu_threads)
        self.ck_fillers.setChecked(d.remove_fillers)
        self.sp_silence.setValue(d.end_silence_ms)
        self.sp_maxlen.setValue(d.soft_max_s)
        self.sp_partial.setValue(d.partial_interval_s)
        self.sp_vad.setValue(d.vad_aggressiveness)
        self.sp_rms.setValue(d.min_rms)
        self.sp_beam.setValue(d.beam_size)

    def _wire(self) -> None:
        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_out.clicked.connect(self._choose_dir)
        self.btn_start.clicked.connect(self._toggle)
        self.btn_open.clicked.connect(self._open_folder)
        self.ck_hide.toggled.connect(self._apply_capture_visibility)
        self.ck_top.toggled.connect(self._apply_on_top)
        self.cb_model.currentTextChanged.connect(lambda _: self._preload_models())
        self.cb_partial.currentTextChanged.connect(lambda _: self._preload_models())
        self.cb_device.currentTextChanged.connect(lambda _: self._preload_models())
        b = self.bridge
        b.partial.connect(self.view.set_partial)
        b.final.connect(self._on_final)
        b.status.connect(self.statusBar().showMessage)
        b.error.connect(self._on_error)
        b.died.connect(self._on_recorder_died)
        b.finished.connect(self._on_finished)

    # ------------------------------------------------------------- settings
    _PERSIST = {
        "res": "cb_res", "fps": "cb_fps", "enc": "cb_enc", "container": "cb_container",
        "model": "cb_model", "partial": "cb_partial", "screen": "cb_screen",
        "mic": "cb_mic", "sys": "cb_sys", "bitrate": "cb_bitrate", "device": "cb_device",
    }  # fmt: skip
    _PERSIST_NUM = {
        "beam": "sp_beam", "threads": "sp_threads", "silence": "sp_silence", "maxlen": "sp_maxlen",
        "partial_iv": "sp_partial", "vad": "sp_vad", "rms": "sp_rms",
    }  # fmt: skip
    _PERSIST_CK = {
        "hide": "ck_hide", "embed": "ck_embed", "cursor": "ck_cursor", "top": "ck_top",
        "fillers": "ck_fillers", "advanced": "adv_box",
    }  # fmt: skip

    def _restore(self) -> None:
        s = self.settings
        for key, attr in self._PERSIST.items():
            val = s.value(key)
            if val:
                cb: QComboBox = getattr(self, attr)
                i = cb.findText(val)
                if i >= 0:
                    cb.setCurrentIndex(i)
        if not s.value("res"):
            self.cb_res.setCurrentText("1080p")
            self.cb_fps.setCurrentText("30")
        self.ed_glossary.setText(s.value("glossary", ""))
        self.ed_out.setText(s.value("out", self.ed_out.text()))
        for key, attr in self._PERSIST_NUM.items():
            v = s.value(key)
            if v is not None:
                w = getattr(self, attr)
                try:
                    w.setValue(type(w.value())(float(v)))
                except (TypeError, ValueError):
                    pass
        for key, attr in self._PERSIST_CK.items():
            v = s.value(key)
            if v is not None:
                getattr(self, attr).setChecked(str(v).lower() == "true")

    def _save(self) -> None:
        s = self.settings
        for key, attr in self._PERSIST.items():
            s.setValue(key, getattr(self, attr).currentText())
        s.setValue("glossary", self.ed_glossary.text())
        s.setValue("out", self.ed_out.text())
        for key, attr in self._PERSIST_NUM.items():
            s.setValue(key, getattr(self, attr).value())
        for key, attr in self._PERSIST_CK.items():
            s.setValue(key, getattr(self, attr).isChecked())

    # -------------------------------------------------------------- devices
    def refresh_devices(self) -> None:
        try:
            screens, audio = list_devices()
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Thiếu ffmpeg", str(exc))
            return
        for cb, items, allow_none in (
            (self.cb_screen, screens, False),
            (self.cb_mic, audio, False),
            (self.cb_sys, audio, True),
        ):
            cur = cb.currentText()
            cb.clear()
            if allow_none:
                cb.addItem(NO_DEVICE, None)
            for d in items:
                cb.addItem(d.name, d.index)
            i = cb.findText(cur)
            if i >= 0:
                cb.setCurrentIndex(i)
        if self.cb_sys.currentText() == NO_DEVICE:
            i = self.cb_sys.findText("BlackHole", Qt.MatchFlag.MatchContains)
            if i >= 0:
                self.cb_sys.setCurrentIndex(i)
        if not screens:
            self.statusBar().showMessage("Không thấy màn hình nào – kiểm tra quyền Screen Recording")

    def _choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    # ------------------------------------------------------------- settings → objects
    def _record_settings(self) -> RecordSettings:
        return RecordSettings(
            screen_index=self.cb_screen.currentData(),
            mic_index=self.cb_mic.currentData(),
            system_audio_index=self.cb_sys.currentData(),
            height=RESOLUTIONS[self.cb_res.currentText()],
            fps=int(self.cb_fps.currentText()),
            encoder=VIDEO_ENCODERS[self.cb_enc.currentText()],
            capture_cursor=self.ck_cursor.isChecked(),
            container=self.cb_container.currentText(),
            embed_subtitles=self.ck_embed.isChecked(),
            bitrate=None if self.cb_bitrate.currentText() in ("", "Tự động") else self.cb_bitrate.currentText().strip(),
        )

    def _asr_settings(self) -> AsrSettings:
        p = self.cb_partial.currentText()
        partial = {"(dùng mô hình chính)": "same", "(tắt bản nháp)": None}.get(p, p)
        return AsrSettings(
            final_model=self.cb_model.currentText(),
            partial_model=partial,
            device=self.cb_device.currentText(),
            cpu_threads=self.sp_threads.value(),
            beam_size=self.sp_beam.value(),
            glossary=self.ed_glossary.text(),
            end_silence_ms=self.sp_silence.value(),
            soft_max_s=self.sp_maxlen.value(),
            partial_interval_s=self.sp_partial.value(),
            vad_aggressiveness=self.sp_vad.value(),
            min_rms=self.sp_rms.value(),
            remove_fillers=self.ck_fillers.isChecked(),
        )

    def _preload_models(self) -> None:
        cfg = self._asr_settings()
        names = [cfg.final_model] + ([cfg.partial_model] if cfg.partial_model not in (None, "same") else [])

        def work() -> None:
            for n in names:
                self.bridge.status.emit(f"Đang nạp mô hình {n} (lần đầu sẽ tải về)…")
                try:
                    ModelPool.get(n, cfg)
                except Exception as exc:  # noqa: BLE001
                    self.bridge.error.emit(f"Không tải được mô hình {n}: {exc}")
                    return
            self.bridge.status.emit("Mô hình sẵn sàng")

        threading.Thread(target=work, daemon=True, name="preload").start()

    # ---------------------------------------------------------- start/stop
    def _toggle(self) -> None:
        if self.session is None:
            self._start()
        elif not self._stopping:
            self._stop()

    def _start(self) -> None:
        if self.cb_screen.currentData() is None or self.cb_mic.currentData() is None:
            QMessageBox.warning(self, "Thiếu thiết bị", "Chọn màn hình và micro trước khi ghi.")
            return
        rec = self._record_settings()
        if not encoder_available(rec.encoder):
            QMessageBox.warning(self, "Bộ mã hóa", f"ffmpeg không hỗ trợ {rec.encoder}.")
            return
        self._save()
        self.view.clear_all()
        self._latencies.clear()
        self._ticks = 0
        b = self.bridge
        try:
            self.session = MeetingSession(
                rec,
                self._asr_settings(),
                Path(self.ed_out.text()),
                on_partial=b.partial.emit,
                on_final=b.final.emit,
                on_status=b.status.emit,
                on_error=b.error.emit,
                on_recorder_died=b.died.emit,
            )
            self.session.start()
        except Exception as exc:  # noqa: BLE001
            self.session = None
            QMessageBox.critical(self, "Lỗi", f"Không bắt đầu được: {exc}")
            return
        self.config_box.setEnabled(False)
        self.btn_open.setEnabled(True)
        self.btn_start.setText("■  Dừng ghi")
        self.btn_start.setStyleSheet("background:#dc2626;color:white;border-radius:6px")
        self.statusBar().showMessage(f"Đang ghi → {self.session.paths.folder}")
        self.timer.start()

    def _stop(self) -> None:
        if not self.session or self._stopping:
            return
        self._stopping = True
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Đang hoàn tất…")
        sess = self.session

        def work() -> None:
            try:
                res = sess.stop(on_status=self.bridge.status.emit)
            except Exception as exc:  # noqa: BLE001
                log.exception("stop failed")
                res = SessionResult(sess.paths.folder, None, error=str(exc))
            self.bridge.finished.emit(res)

        threading.Thread(target=work, daemon=True, name="stopper").start()

    def _on_finished(self, res: SessionResult) -> None:
        self.timer.stop()
        self._tick()
        self.session = None
        self._stopping = False
        self._last_result = res
        self.config_box.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_start.setText("●  Bắt đầu ghi")
        self.btn_start.setStyleSheet("")
        if res.error:
            QMessageBox.warning(self, "Có lỗi khi ghi", res.error)
        msg = f"Đã lưu: {res.video.name if res.video else '(không có video)'}"
        if res.offset:
            msg += f"  (bù lệch audio {res.offset * 1000:.0f} ms)"
        self.statusBar().showMessage(msg)

    def _on_final(self, r: FinalResult) -> None:
        self.view.add_final(r)
        if r.latency is not None:
            self._latencies = (self._latencies + [r.latency])[-10:]

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg)
        QMessageBox.warning(self, "Lỗi", msg)

    def _on_recorder_died(self, tail: str) -> None:
        QMessageBox.critical(
            self,
            "ffmpeg đã dừng",
            "Quá trình ghi hình kết thúc bất ngờ. Kiểm tra quyền Screen Recording / Microphone "
            "trong System Settings → Privacy & Security.\n\n" + tail,
        )
        self._stop()

    def _tick(self) -> None:
        s = self.session
        if s is None:
            return
        self.lbl_time.setText(fmt_clock(s.elapsed))
        self._ticks = getattr(self, "_ticks", 0) + 1
        if s.elapsed == 0 and self._ticks == 16 and s.recorder.is_running():  # ~8 s, no audio at all
            self.statusBar().showMessage(
                "⚠ Chưa nhận được dữ liệu từ ffmpeg – kiểm tra quyền Screen Recording/Microphone "
                "(System Settings → Privacy & Security) rồi khởi động lại ứng dụng."
            )
        parts = []
        if not s.engine.ready.is_set():
            parts.append("đang nạp mô hình…")
        else:
            parts.append(f"tồn đọng {s.engine.backlog_seconds:.1f}s")
        if self._latencies:
            parts.append(f"độ trễ câu ~{sum(self._latencies) / len(self._latencies):.1f}s")
        self.lbl_stats.setText(" · ".join(parts))
        if self.ck_hide.isChecked():
            macos.set_windows_capturable(False)  # also covers newly opened dialogs

    # ------------------------------------------------------------- misc
    def _apply_capture_visibility(self) -> None:
        macos.set_windows_capturable(not self.ck_hide.isChecked())

    def _apply_on_top(self) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.ck_top.isChecked())
        self.show()
        QTimer.singleShot(0, self._apply_capture_visibility)

    def _open_folder(self) -> None:
        folder = self.session.paths.folder if self.session else (
            self._last_result.folder if self._last_result else Path(self.ed_out.text())
        )
        subprocess.run(["open", str(folder)])

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        QTimer.singleShot(0, self._apply_capture_visibility)

    def closeEvent(self, e) -> None:  # noqa: N802
        self._save()
        if self._stopping:
            self.statusBar().showMessage("Đang hoàn tất file, vui lòng đợi rồi thoát…")
            e.ignore()
            return
        if self.session is not None:
            if (
                QMessageBox.question(self, "Đang ghi", "Dừng ghi và lưu file trước khi thoát?")
                != QMessageBox.StandardButton.Yes
            ):
                e.ignore()
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.session.stop()
            finally:
                QApplication.restoreOverrideCursor()
        e.accept()


def run_gui() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MeetingTool")
    w = MainWindow()
    if w.ck_top.isChecked():
        w.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    w.show()
    return app.exec()
