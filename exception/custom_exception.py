import sys
import traceback
from typing import Optional


class BEVException(Exception):
    """
    Custom exception for the BEV Occupancy project.
    Automatically captures file name, line number,
    and full traceback — just like DocumentPortalException.

    Usage:
        try:
            img = cv2.imread(path)
        except Exception as e:
            raise BEVException("Failed to load image", e) from e
    """

    def __init__(self,
                 error_message: str,
                 error_detail: Optional[Exception] = None):

        # ── Capture traceback info ──────────────────────
        exc_type  = None
        exc_value = None
        exc_tb    = None

        if error_detail is not None:
            # Use the passed exception's traceback
            exc_type  = type(error_detail)
            exc_value = error_detail
            exc_tb    = error_detail.__traceback__
        else:
            # Use current exception context
            exc_type, exc_value, exc_tb = sys.exc_info()

        # ── Walk to the last frame (most relevant line) ─
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        # ── Extract file + line info ────────────────────
        if last_tb:
            self.file_name    = last_tb.tb_frame.f_code.co_filename
            self.line_number  = last_tb.tb_lineno
        else:
            self.file_name    = "<unknown>"
            self.line_number  = -1

        self.error_message = str(error_message)

        # ── Full traceback string ───────────────────────
        if exc_type and exc_tb:
            self.traceback_str = ''.join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        base = (
            f"\n{'='*60}\n"
            f"  BEV Exception\n"
            f"{'='*60}\n"
            f"  File    : {self.file_name}\n"
            f"  Line    : {self.line_number}\n"
            f"  Message : {self.error_message}\n"
            f"{'='*60}"
        )
        if self.traceback_str:
            base += f"\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return (
            f"BEVException("
            f"file={self.file_name!r}, "
            f"line={self.line_number}, "
            f"message={self.error_message!r})"
        )