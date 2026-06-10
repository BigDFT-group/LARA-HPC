"""Logging functionality for laraq agents."""

from pathlib import Path

DEFAULT_LOGFILE = "lara_chat.log"


class LoggingMixin:
    """Mixin class for logging agent interactions."""

    _logfile: str | Path = DEFAULT_LOGFILE
    source_name: str = "LARA-HPC"

    @property
    def log_file(self) -> Path | None:
        """Get the log file path."""
        if self._logfile is None:
            return None
        return Path(self._logfile)

    def clear_log(self) -> None:
        """Clear the log file by deleting it."""
        if self.log_file and self.log_file.exists():
            self.log_file.unlink()

    def write_log(self, content: str, append: bool = True) -> None:
        """Log a message to the logfile.

        Args:
            content: Content to write to the log
            append: append to the log if True, otherwise overwrite

        Returns:
            None
        """
        if not self.log_file:
            return

        mode = "a+" if append else "w+"

        with self.log_file.open(mode) as f:
            f.write(content)
