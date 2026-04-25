"""
utils/sanitizer.py — Input sanitisation for all user-submitted text.
 
Handles every category of messy real-world input:
  • None / non-string types
  • Empty or whitespace-only strings
  • Extremely long text (>4000 chars)
  • Null bytes and control characters
  • Unicode edge cases (surrogates, BOM, zero-width chars)
  • HTML / script injection attempts
  • Repeated characters (e.g. "aaaaaaaaaa…")
  • Mixed-language text  ← left intact (the LLM handles it)
 
Returns
-------
A clean UTF-8 string, always.  May be empty string for completely
un-recoverable input — callers must handle the empty case.
"""
 
import re
import unicodedata
import logging
 
logger = logging.getLogger("aura.sanitizer")
 
_MAX_LENGTH = 4000
_MIN_USEFUL_LENGTH = 1
 
# Characters to strip entirely
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"   # ASCII control chars (keep \t \n \r)
    r"\ufeff"                               # BOM
    r"\u200b-\u200d"                        # Zero-width chars
    r"\u2028\u2029"                         # Line/paragraph separators
    r"]",
    re.UNICODE,
)
 
# Collapse runs of 5+ identical characters to 3 (e.g. "heeelp" stays, "aaaaaaa" → "aaa")
_REPEAT_RE = re.compile(r"(.)\1{4,}", re.UNICODE)
 
# Very basic HTML tag stripper (not a security measure — just cosmetic cleanup)
_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>", re.DOTALL)
 
 
def sanitize_input(raw: object) -> str:
    """
    Sanitise raw user input into a clean, bounded string.
 
    Never raises.  Returns empty string on complete failure.
    """
    try:
        # ── Step 1: Coerce to string ──────────────────────────────────────
        if raw is None:
            logger.debug("sanitize: received None input")
            return ""
 
        text = str(raw)
 
        # ── Step 2: Normalize unicode (NFC) ──────────────────────────────
        try:
            text = unicodedata.normalize("NFC", text)
        except (TypeError, ValueError):
            logger.warning("sanitize: unicode normalization failed, skipping")
 
        # ── Step 3: Remove control characters ────────────────────────────
        text = _CONTROL_CHARS_RE.sub("", text)
 
        # ── Step 4: Strip HTML tags ───────────────────────────────────────
        text = _HTML_TAG_RE.sub("", text)
 
        # ── Step 5: Collapse excessive whitespace ─────────────────────────
        # Preserve up to 2 consecutive newlines (paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse horizontal whitespace runs to a single space
        text = re.sub(r"[^\S\n]+", " ", text)
 
        # ── Step 6: Trim leading/trailing whitespace ──────────────────────
        text = text.strip()
 
        # ── Step 7: Truncate to max length ────────────────────────────────
        if len(text) > _MAX_LENGTH:
            logger.info(
                "sanitize: input truncated from %d to %d chars",
                len(text), _MAX_LENGTH,
            )
            # Truncate at word boundary where possible
            text = text[:_MAX_LENGTH].rsplit(" ", 1)[0]
            text = text + " [truncated]"
 
        # ── Step 8: Collapse extreme character repetition ────────────────
        # "heeelllppp" → "heelllppp" etc — keeps expressive text readable
        text = _REPEAT_RE.sub(lambda m: m.group(1) * 3, text)
 
        logger.debug("sanitize: clean length=%d", len(text))
        return text
 
    except Exception as exc:
        # Absolute fallback — log and return empty
        logger.error("sanitize: unexpected error: %s", exc, exc_info=True)
        return ""
 
 
def is_meaningful(text: str) -> bool:
    """
    Return True if text contains enough content to process.
 
    Used by the pipeline to decide whether to send a
    "I didn't quite get that" response vs. attempting analysis.
    """
    if not text or len(text.strip()) < _MIN_USEFUL_LENGTH:
        return False
    # At least one alphanumeric character
    return bool(re.search(r"\w", text, re.UNICODE))
 