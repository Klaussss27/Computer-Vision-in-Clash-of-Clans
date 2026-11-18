"""Window capture and desktop coordinate mapping for Windows."""

import ctypes
import threading
import time
from ctypes import wintypes


def enable_dpi_awareness():
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))


def list_windows():
    import win32gui

    windows = []

    def visit(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if title and win32gui.IsWindowVisible(hwnd):
            windows.append((hwnd, title))

    win32gui.EnumWindows(visit, None)
    return windows


class GameWindow:
    def __init__(self, title, expected_size):
        import win32gui
        from windows_capture import WindowsCapture

        self.gui = win32gui
        matches = [(hwnd, name) for hwnd, name in list_windows() if name == title]
        if len(matches) != 1:
            raise RuntimeError(
                "Game window not found. Use --list-windows and set window_title exactly."
            )
        self.hwnd = matches[0][0]
        self.expected_size = tuple(expected_size)
        self.condition = threading.Condition()
        self.frame = None
        self.updated = 0
        self.closed = False
        self.capture = WindowsCapture(window_hwnd=self.hwnd, cursor_capture=False)

        @self.capture.event
        def on_frame_arrived(frame, capture_control):
            with self.condition:
                self.frame = frame.frame_buffer.copy()
                self.updated = time.monotonic()
                self.condition.notify_all()

        @self.capture.event
        def on_closed():
            with self.condition:
                self.closed = True
                self.condition.notify_all()

        self.control = self.capture.start_free_threaded()

    def screenshot(self):
        from PIL import Image

        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.frame is not None, timeout=10)
            if self.closed or self.frame is None or time.monotonic() - self.updated > 5:
                raise RuntimeError(
                    "Game capture closed or stopped updating. Keep the window open and restored."
                )
            frame = self.frame.copy()
        image = Image.fromarray(frame[:, :, [2, 1, 0]])
        if image.size != self.expected_size:
            image.save("preview.png")
            raise RuntimeError(
                f"Captured size is {image.size}; set capture_size to match and recalibrate. Saved preview.png."
            )
        return image

    def desktop_point(self, point):
        if not self.gui.IsWindow(self.hwnd) or self.gui.IsIconic(self.hwnd):
            raise RuntimeError("Game window closed or minimized")
        if self.gui.GetForegroundWindow() != self.hwnd:
            raise RuntimeError("Focus the game before live input")
        rect = wintypes.RECT()
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(self.hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if result != 0:
            raise RuntimeError("Cannot read game window bounds")
        if (rect.right - rect.left, rect.bottom - rect.top) != self.expected_size:
            raise RuntimeError("Window size changed; recalibrate before continuing")
        target = (rect.left + point[0], rect.top + point[1])
        hit = self.gui.WindowFromPoint(target)
        if hit != self.hwnd and not self.gui.IsChild(self.hwnd, hit):
            raise RuntimeError("Another window covers the click target")
        return target

    def close(self):
        self.control.stop()
