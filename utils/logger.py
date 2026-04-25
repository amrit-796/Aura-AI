"""
utils/logger.py — Centralised structured logging setup.
 
Creates a logger that writes:
  - INFO+   to stdout (coloured for terminals)
  - WARNING+ to logs/aura.log (plain text, rotating)
 
Usage:
    from utils.logger import setup_logger
    logger = setup_logger("aura.mymodule")
"""
 
import logging
import logging.handlers
import os
import sys
 
# Create logs dir if needed
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
 
_LOG_FILE = os.path.join(_LOG_DIR, "aura.log")
 
# ── ANSI colour codes for terminal ────────────────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[90m",   # grey
    "INFO":     "\033[94m",   # blue
    "WARNING":  "\033[93m",   # yellow
    "ERROR":    "\033[91m",   # red
    "CRITICAL": "\033[95m",   # magenta
}
_RESET = "\033[0m"
 
 
class _ColourFormatter(logging.Formatter):
    """Apply ANSI colour to the level name in terminal output."""
 
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)
 
 
def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Build and return a configured logger for the given module name.
 
    Idempotent — calling setup_logger with the same name twice returns
    the same logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
 
    if logger.handlers:
        return logger  # Already configured
 
    logger.setLevel(level)
    logger.propagate = False
 
    # ── Console handler (coloured) ────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(_ColourFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)
 
    # ── File handler (rotating, plain text) ───────────────────────────────
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning("Could not set up file logging: %s", exc)
 
    return logger
 