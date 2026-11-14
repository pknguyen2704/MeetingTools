"""macOS helpers."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def set_windows_capturable(capturable: bool) -> bool:
    """Hide (or show) all of this app's windows from screen capture.

    Uses NSWindow.sharingType = NSWindowSharingNone, which AVFoundation screen
    capture (used by ffmpeg) honours: the transcript window is visible to you
    but does not appear in the recorded video.
    """
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApplication, NSWindowSharingNone, NSWindowSharingReadOnly

        mode = NSWindowSharingReadOnly if capturable else NSWindowSharingNone
        for w in NSApplication.sharedApplication().windows():
            w.setSharingType_(mode)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("cannot change window sharing type: %s", exc)
        return False
