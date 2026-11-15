# MeetingTool

Ghi hình toàn bộ cuộc họp (màn hình + tiếng) và chép lời **tiếng Anh và tiếng Việt** **realtime**:

- Chế độ **Tự động (Anh + Việt)** nhận diện ngôn ngữ cho từng câu, nên cuộc họp có thể chuyển qua lại giữa hai thứ tiếng. Cũng có thể cố định Tiếng Anh hoặc Tiếng Việt.

- Bản nháp hiện ngay khi đang nói (chữ xám nghiêng), cập nhật khoảng mỗi 1–1.5 s.
- Khi người nói dừng hoặc chuyển câu, câu được chép lại bằng mô hình chính xác hơn và **chuẩn hóa** (dấu chấm, dấu phẩy, dấu hỏi, viết hoa, bỏ "um/uh").
- Lời chép chỉ hiện trong ứng dụng, **không in lên video**. Cửa sổ ứng dụng cũng được ẩn khỏi bản ghi màn hình.
- Sau khi dừng sẽ xuất `.mp4` cùng `.srt`, `.vtt`, `.txt`, `.json`. Timestamp khớp với video vì lấy từ chính luồng audio được ghi vào video.

Máy đích: MacBook Pro 2017 (Intel i7, 16 GB, Radeon), macOS 12–13.

## Cài đặt

```bash
# 1. uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. ffmpeg (cần bản có avfoundation + videotoolbox)
brew install ffmpeg
#   Nếu Homebrew trên macOS cũ báo lỗi hoặc build quá lâu: tải bản static Intel
#   (vd. evermeet.cx/ffmpeg) rồi trỏ tới nó:
#   export MEETINGTOOL_FFMPEG=/path/ffmpeg  MEETINGTOOL_FFPROBE=/path/ffprobe

# 3. Cài dependencies (uv tự tải Python 3.12)
cd MeetingTool
uv sync

# 4. Chạy
uv run meetingtool
```

Lần đầu chạy, ứng dụng tự tải mô hình về `~/.cache/huggingface`: `small` khoảng 480 MB, `tiny` khoảng 75 MB.

### Quyền macOS (bắt buộc)

Vào **System Settings → Privacy & Security** và cấp cho ứng dụng chạy lệnh (Terminal, iTerm hoặc VS Code):

- **Screen Recording**
- **Microphone**

Sau khi cấp quyền, khởi động lại ứng dụng đó. Nếu thiếu quyền, ffmpeg sẽ treo mà không báo lỗi; ứng dụng sẽ cảnh báo *"Chưa nhận được dữ liệu từ ffmpeg"* sau khoảng 8 s.

### Thu cả tiếng người họp từ xa (Zoom/Meet/Teams)

Micro chỉ thu được giọng của bạn. Để thu cả tiếng của những người khác trong cuộc họp:

1. `brew install blackhole-2ch`
2. Mở **Audio MIDI Setup** → `+` → **Create Multi-Output Device**, tick *loa/tai nghe của bạn* và *BlackHole 2ch*. Chọn thiết bị này làm Output của macOS (hoặc làm loa trong Zoom) để bạn vẫn nghe được bình thường.
3. Trong MeetingTool, ở mục **Âm thanh hệ thống**, chọn `BlackHole 2ch` (ứng dụng tự chọn nếu tìm thấy).

ffmpeg sẽ trộn micro và âm thanh hệ thống, rồi đưa bản trộn đó vào cả video lẫn bộ chép lời.

## Sử dụng

| Mục | Ý nghĩa |
|---|---|
| Màn hình / Micro / Âm thanh hệ thống | Thiết bị ghi (nút ↻ để quét lại) |
| Độ phân giải, FPS, mp4/mkv | Chất lượng đầu ra: Gốc, 1440p, 1080p, 720p hoặc 480p |
| Bộ mã hóa | VideoToolbox (phần cứng, khuyên dùng) hoặc x264 (dùng CPU) |
| Ngôn ngữ | Tự động (Anh + Việt, nhận diện từng câu), Tiếng Anh hoặc Tiếng Việt. Ở chế độ tự động, mỗi dòng có nhãn `EN`/`VI` |
| Mô hình chép lời, beam | Mô hình tạo câu chuẩn; beam cao thì chính xác hơn nhưng chậm hơn |
| Bản nháp realtime | Mô hình nhỏ cho chữ chạy trực tiếp, có thể tắt |
| Từ khóa | Tên riêng hoặc thuật ngữ giúp mô hình viết đúng (vd: `Kubernetes, OKR, Hanoi`) |
| Ẩn cửa sổ này khỏi video | Cửa sổ chép lời không xuất hiện trong bản ghi |
| Nhúng phụ đề mềm | Thêm track phụ đề bật/tắt được vào mp4 (không in lên hình) |
| **Nâng cao** | Bitrate, CPU/GPU, số luồng, khoảng lặng chốt câu, độ dài câu tối đa, chu kỳ cập nhật bản nháp, lọc ồn VAD, ngưỡng âm lượng, bỏ từ đệm |

Mọi cấu hình được lưu lại cho lần mở sau.

Mỗi phiên ghi lưu vào `~/Movies/MeetingTool/Meeting_<ngày-giờ>/`:

```
meeting_<..>.mp4        video + audio
meeting_<..>.srt/.vtt   phụ đề; đặt cùng tên với video nên VLC/IINA tự nạp
meeting_<..>.txt        [00:01:23] Câu đã chuẩn hóa.
meeting_<..>.json       start/end/text/lang từng câu
meeting_<..>.live.txt   ghi dần trong lúc họp, an toàn nếu ứng dụng bị tắt đột ngột
ffmpeg.log
```

### Dòng lệnh

```bash
uv run meetingtool devices                                  # liệt kê màn hình / micro
uv run meetingtool record --screen 1 --mic 0 --sys 2 --res 720p   # ghi không giao diện, Ctrl+C để dừng
uv run meetingtool transcribe meeting.mp4 --realtime        # chạy pipeline realtime trên một file
uv run meetingtool transcribe meeting.mp4 --language vi     # cố định tiếng Việt (auto | en | vi)
uv run meetingtool bench                                    # đo tốc độ các mô hình trên máy này
uv run meetingtool bench --models small.en --device mps     # thử GPU (Metal)
uv run meetingtool bench --language vi                      # đo trên giọng đọc tiếng Việt (giọng Linh của macOS)
```

## Vì sao chọn các công nghệ này

### Chép lời: faster-whisper (CTranslate2, int8), chạy 2 tầng

- Whisper tự sinh **dấu câu và viết hoa**, nên đáp ứng yêu cầu chuẩn hóa mà không cần thêm mô hình khôi phục dấu câu. Các mô hình `.en` chỉ dành cho tiếng Anh và chính xác hơn bản đa ngôn ngữ cùng cỡ. Để chép **tiếng Việt** phải dùng mô hình đa ngôn ngữ (không có `.en`); ứng dụng tự đổi `small.en` → `small` khi bạn chọn Tiếng Việt hoặc Tự động.
- **Hai ngôn ngữ:** ở chế độ tự động, trước khi giải mã mỗi câu, mô hình chính chạy nhận diện ngôn ngữ nhưng chỉ chọn giữa `en` và `vi`. Câu ngắn và mơ hồ (vd. "OK") giữ ngôn ngữ của câu trước. Ngữ cảnh và prompt cũng tách riêng theo ngôn ngữ, để Whisper không dịch câu tiếng Việt sang tiếng Anh. Bước nhận diện tốn thêm một lượt encoder cho mỗi câu; khi máy bị tồn đọng, ứng dụng bỏ qua bước này và dùng lại ngôn ngữ mà bản nháp đã đoán. Hạn chế: nếu một câu trộn Anh lẫn Việt mà không ngắt nghỉ, Whisper chỉ chép theo một ngôn ngữ, và phần nói bằng ngôn ngữ còn lại có thể bị bỏ hoặc bị dịch.
- CTranslate2 int8 là cách chạy Whisper trên CPU Intel nhanh nhất hiện nay (dùng AVX2), tiêu tốn RAM ít hơn PyTorch khoảng 4 lần. Card AMD Radeon không chạy được CUDA hay ROCm trên macOS, nên mặc định dùng CPU. Có tùy chọn `mps` (Metal, thử nghiệm); nếu không chạy được, ứng dụng tự quay về CPU.
- **Hai tầng:**
  - `tiny` giải mã nhanh (greedy) phần câu đang nói để hiện bản nháp (`base` cho bản nháp tiếng Việt dễ đọc hơn, đổi lại chậm hơn).
  - Khi VAD phát hiện người nói ngừng khoảng 700 ms, `small` giải mã lại cả câu bằng beam search, kèm ngữ cảnh các câu trước, rồi chuẩn hóa văn bản.
  - Câu chuẩn luôn được ưu tiên xử lý trước. Bản nháp tự bỏ qua khi máy đang bị tồn đọng, và nếu tồn đọng quá 8 s thì beam tự giảm về 1 để không bị trễ dồn.
- Gợi ý chọn mô hình trên i7 2017 (hãy chạy `meetingtool bench` để có số liệu thật trên máy bạn):

  | Mô hình | Độ chính xác | Tốc độ | Khi nào dùng |
  |---|---|---|---|
  | `small` (mặc định) | tốt, Anh + Việt | đủ realtime với beam 2 | họp tiếng Việt hoặc song ngữ |
  | `medium`, `large-v3-turbo` | tiếng Việt tốt hơn nhiều | nặng, khó realtime trên i7 2017 | máy mạnh (Apple Silicon) |
  | `small.en` | tốt nhất cho tiếng Anh ở cỡ này | đủ realtime với beam 2 | họp chỉ có tiếng Anh |
  | `distil-small.en` | gần bằng small | nhanh hơn | máy nóng hoặc chậm |
  | `base.en` | khá | rất nhanh | ưu tiên tiết kiệm CPU |
  | `medium.en`, `distil-medium.en` | tốt nhất | nặng | máy mạnh hơn |

### Ghi hình: ffmpeg + VideoToolbox, một tiến trình duy nhất

- Màn hình và micro được mở trong **cùng một input avfoundation**, nên dùng chung một đồng hồ.
- Mã hóa video bằng **phần cứng** (Quick Sync qua VideoToolbox), CPU gần như để dành hết cho chép lời. Thu nhỏ độ phân giải dùng `fast_bilinear`, chi phí thấp.
- Audio được tách đôi ngay trong filter graph: một nhánh AAC ghi vào video, một nhánh PCM 16 kHz chuyển thẳng cho bộ chép lời. `aresample=async=1000:first_pts=0` đảm bảo **số mẫu PCM chính là trục thời gian của track audio trong video**, nên phụ đề không bị trôi dù họp nhiều giờ. Khi dừng, ứng dụng dùng `ffprobe` đo độ lệch đầu file (nếu có) và bù vào phụ đề.
- Ghi ra MKV trước để chịu được crash, sau đó remux sang MP4 (copy, không mã hóa lại).
- Cửa sổ ứng dụng được đặt `NSWindowSharingNone` nên không lọt vào bản ghi. Cách này có tác dụng với avfoundation trên macOS 12–13. Trên macOS mới hơn, nếu cửa sổ vẫn bị ghi lại, hãy đặt nó sang màn hình phụ không ghi.

## Phát triển

```bash
uv sync
uv run pytest            # unit + end-to-end (cần ffmpeg và lệnh `say` của macOS)
```

Test end-to-end dùng màn hình giả lập 2880×1800 và giọng đọc có khoảng lặng biết trước, chạy qua đúng pipeline ffmpeg → VAD → Whisper. Test kiểm tra rằng file mp4 720p được tạo ra, câu được chuẩn hóa dấu câu, và timestamp phụ đề khớp với lúc bắt đầu nói đo được trên file video.

```
src/meetingtool/
  cli.py         lệnh: gui / devices / record / transcribe / bench
  ui.py          giao diện PySide6
  session.py     điều phối một phiên ghi: recorder + engine + xuất file
  recorder.py    lệnh ffmpeg, tap PCM, remux, đo lệch
  segmenter.py   VAD (WebRTC) + ngắt câu theo khoảng lặng
  engine.py      faster-whisper: bản nháp + bản chuẩn, ModelPool
  textnorm.py    chuẩn hóa dấu câu, viết hoa (Unicode, có tiếng Việt), lọc ảo giác
  transcript.py  xuất SRT / VTT / TXT / JSON
  devices.py     tìm ffmpeg, liệt kê thiết bị avfoundation
  macos.py       ẩn cửa sổ khỏi bản ghi màn hình
```
