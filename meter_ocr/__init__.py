"""Meter OCR pipeline: RTSP camera -> EasyOCR -> hourly CSV."""

import sys

__version__ = "1.0.0"


def _fix_console_encoding() -> None:
    """Make stdout/stderr UTF-8 safe on Windows.

    The default Windows console codepage (cp1252) cannot encode the block
    characters EasyOCR uses in its model-download progress bar, which raises
    UnicodeEncodeError and kills the process mid-download. Reconfiguring with
    errors='replace' makes any such character harmless.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # already fine, or redirected to something unreconfigurable


_fix_console_encoding()
