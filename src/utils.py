"""Provide system-level utility functions for environment validation and interaction."""

from __future__ import annotations

import ctypes
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import pygetwindow as gw
import pyperclip

if TYPE_CHECKING:
    from collections.abc import Generator


def prevent_system_sleep(enable: bool = True) -> None:  # noqa: FBT001, FBT002
    """Prevent or allow the system to enter sleep/screen saver mode.

    ES_CONTINUOUS: Stay in the state until next call.
    ES_SYSTEM_REQUIRED: Prevent system sleep.
    ES_DISPLAY_REQUIRED: Prevent monitor/screen saver.
    """
    ES_CONTINUOUS = 0x80000000  # noqa: N806
    ES_SYSTEM_REQUIRED = 0x00000001  # noqa: N806
    ES_DISPLAY_REQUIRED = 0x00000002  # noqa: N806

    if enable:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED,
        )
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def validate_environment(
    required_assets: list[Path] | None = None,
    tesseract_path: str | None = None,
) -> None:
    """Perform pre-flight checks to ensure the runner environment is compatible.

    Args:
        target_res: The expected screen resolution (width, height).
        required_assets: List of paths to required files (e.g., icons).
        tesseract_path: Path to the Tesseract executable to verify.

    Raises:
        RuntimeError: If Admin rights are missing or resolution is mismatched.
        FileNotFoundError: If required assets or dependencies are missing.

    """
    if not ctypes.windll.shell32.IsUserAnAdmin():
        msg = "Automation must be run with Administrative privileges."
        raise RuntimeError(msg)

    if required_assets:
        for asset in required_assets:
            if not asset.exists():
                msg = f"Missing Critical Asset: {asset}"
                raise FileNotFoundError(msg)
    print("[INIT] Environment and Asset Integrity Verified.")

    if tesseract_path and not Path(tesseract_path).exists():
        msg = f"Tesseract OCR not found at: {tesseract_path}"
        raise FileNotFoundError(msg)


@contextmanager
def preserve_clipboard() -> Generator[None]:
    """Context manager to backup and restore the system clipboard."""
    backup = pyperclip.paste()
    try:
        yield
    finally:
        pyperclip.copy(backup)


def wait_for_window_to_disappear(window_title: str, timeout: float = 5.0) -> bool:
    """Wait until all windows with a specific title are closed.

    Args:
        window_title: The title of the window to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        bool: True if the window disappeared, False if it timed out.

    """
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
        if not [w for w in gw.getWindowsWithTitle(window_title) if w.visible]:
            return True
        time.sleep(0.2)
    return False


def kill_bot_process_only(bot_pid: int | None) -> None:
    """Terminate the specific process started by the bot, leaving others alone."""
    if bot_pid is None:
        return
    try:
        proc = psutil.Process(bot_pid)
        if proc.is_running():
            proc.terminate()
            print(f"[CLEANUP] Terminated bot-owned process PID: {bot_pid}")
    except psutil.NoSuchProcess, psutil.AccessDenied:
        pass


def get_pid_from_hwnd(hwnd: int) -> int:
    """Identify the PID from a specific window handle."""
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value
