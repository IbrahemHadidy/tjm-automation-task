"""Provide system-level utility functions for environment validation and interaction."""

from __future__ import annotations

import ctypes
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import pyautogui
import pygetwindow as gw
import pyperclip

if TYPE_CHECKING:
    from collections.abc import Generator

    from core import StructuredLogger
    from screenshot_service import ScreenshotService


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


def _normalize_pos(pos: pyautogui.Point) -> tuple[int, int]:
    """Convert a pyautogui.Point to a plain (x, y) tuple."""
    return (int(pos.x), int(pos.y))


def check_for_interference(
    expected_mouse_pos: pyautogui.Point,
    *,
    screen_service: ScreenshotService | None = None,
    logger: StructuredLogger | None = None,
    drift_px: int = 2,
    stable_seconds: float = 2.0,
    poll_interval: float = 0.25,
    timeout: float | None = 60.0,
    wait_for_clear: bool = True,
) -> None:
    """Watchdog: detect and handle user mouse interference.

    Behavior:
    - If mouse moved more than `drift_px` from `expected_mouse_pos`, and
      `wait_for_clear` is False: raises InterruptedError immediately.
    - If `wait_for_clear` is True: takes a workspace snapshot (if provided),
      then blocks until the mouse has been *stable* (movement <= `drift_px`)
      for `stable_seconds`. After stability is observed, it restores the
      workspace (if provided) and returns.
    - If stability is not achieved within `timeout` seconds, raises InterruptedError.

    Args:
        expected_mouse_pos: previous mouse (x, y) coordinates to compare against.
        screen_service: optional ScreenshotService used to snapshot/restore workspace.
        logger: optional StructuredLogger for structured events.
        drift_px: allowed pixel drift before considering movement interference.
        stable_seconds: required uninterrupted stable period to resume.
        poll_interval: how often to poll the mouse position while waiting.
        timeout: maximum time to wait for stability (None => wait indefinitely).
        wait_for_clear: if False, raise on first detection instead of waiting.

    Raises:
        InterruptedError: if user interference is detected and not cleared within timeout.

    """
    prev = _normalize_pos(expected_mouse_pos)
    curr = _normalize_pos(pyautogui.position())

    dx = abs(curr[0] - prev[0])
    dy = abs(curr[1] - prev[1])

    if max(dx, dy) <= drift_px:
        # no meaningful interference
        return

    # interference detected
    msg = f"User interference detected (dx={dx}, dy={dy})."
    if logger:
        logger.warning("user_interference_detected", dx=dx, dy=dy)
    else:
        # fallback to printing minimal info if no logger is provided
        print(f"[WARN] {msg}")

    if not wait_for_clear:
        raise InterruptedError(msg)

    # Save workspace snapshot (if available)
    if screen_service:
        try:
            if logger:
                logger.info("snapshot_workspace_on_interference")
            screen_service.snapshot_workspace()
        except Exception as e:
            if logger:
                logger.debug("snapshot_failed", reason=str(e))

    # Wait until mouse is stable for `stable_seconds`
    stable_started_at: float | None = None
    start_wait = time.time()

    while True:
        now_pos_raw = pyautogui.position()
        now_pos = _normalize_pos(now_pos_raw)
        if now_pos is None:
            # treat as instability; reset stability timer
            stable_started_at = None
        else:
            ddx = abs(now_pos[0] - curr[0])
            ddy = abs(now_pos[1] - curr[1])

            if max(ddx, ddy) <= drift_px:
                if stable_started_at is None:
                    stable_started_at = time.time()
                elif (time.time() - stable_started_at) >= stable_seconds:
                    # stable long enough -> done
                    break
            else:
                # movement detected; update current baseline and reset stability timer
                curr = now_pos
                stable_started_at = None

        # timeout check
        if timeout is not None and (time.time() - start_wait) > timeout:
            err = "User interference persisted past timeout; aborting."
            if logger:
                logger.error("interference_timeout", timeout=timeout)
            raise InterruptedError(err)

        time.sleep(poll_interval)

    # restore workspace if we saved it
    if screen_service:
        try:
            if logger:
                logger.info("restore_workspace_after_interference")
            screen_service.restore_workspace()
        except Exception as e:
            if logger:
                logger.debug("restore_failed", reason=str(e))

    # finally, function returns and caller can continue


def wait_for_pixel_color(
    x: int,
    y: int,
    expected_rgb: tuple[int, int, int],
    timeout: float = 5.0,
    tolerance: int = 10,
) -> bool:
    """Wait for a specific screen coordinate to match a target color.

    Args:
        x: Screen X.
        y: Screen Y.
        expected_rgb: Target (R, G, B) tuple.
        timeout: Max wait time.
        tolerance: Allowed variance in color values.

    Returns:
        bool: True if color matched, False otherwise.

    """
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        current_rgb = pyautogui.pixel(x, y)

        # Check tolerance for each channel
        match = all(
            abs(c - e) <= tolerance
            for c, e in zip(current_rgb, expected_rgb, strict=False)
        )
        if match:
            return True
        time.sleep(0.1)
    return False


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


def kill_processes_by_name(process_name: str) -> int:
    """Force terminate all processes matching a specific name."""
    count = 0
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"].lower() == process_name.lower():
                proc.terminate()
                count += 1
        except psutil.NoSuchProcess, psutil.AccessDenied:
            continue
    return count


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
