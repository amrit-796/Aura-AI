"""
utils/display.py — Terminal UI helpers.

Keeps all print/colour logic in one place so the rest of the codebase
stays clean.  Works on any ANSI-capable terminal (macOS, Linux, modern
Windows Terminal).
"""

import textwrap

# ── ANSI colour codes ──────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_GREY   = "\033[90m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"

_WRAP_WIDTH = 80


def _wrap(text: str) -> str:
    """Wrap text to _WRAP_WIDTH columns, preserving paragraph breaks."""
    paragraphs = text.split("\n\n")
    wrapped = []
    for para in paragraphs:
        # Preserve intentional single newlines (e.g. bullet-like lines)
        lines = para.split("\n")
        rewrapped_lines = [
            textwrap.fill(line, width=_WRAP_WIDTH) if line.strip() else ""
            for line in lines
        ]
        wrapped.append("\n".join(rewrapped_lines))
    return "\n\n".join(wrapped)


def print_banner():
    """Print the welcome banner."""
    banner = f"""
{_CYAN}{_BOLD}╔══════════════════════════════════════════════════════╗
║         ✦  A U R A  —  Your Emotional Companion  ✦       ║
║                                                          ║
║   A safe space to think, feel, and find your footing.   ║
║   Type 'quit' or 'bye' at any time to exit.             ║
╚══════════════════════════════════════════════════════════╝{_RESET}
"""
    print(banner)


def print_assistant(message: str):
    """Print an assistant message with formatting."""
    prefix = f"{_CYAN}{_BOLD}Aura ›{_RESET} "
    wrapped = _wrap(message)
    # Indent continuation lines to align with text after "Aura › "
    indent = " " * 7
    lines = wrapped.splitlines()
    formatted_lines = [prefix + lines[0]] if lines else [prefix]
    for line in lines[1:]:
        formatted_lines.append(indent + line)
    print("\n" + "\n".join(formatted_lines) + "\n")


def print_user_prompt() -> str:
    """
    Display the user prompt and return their input.
    Returns the stripped input string.
    """
    try:
        user_input = input(f"{_GREEN}{_BOLD}You  ›{_RESET} ")
        return user_input.strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"


def print_divider():
    """Print a subtle horizontal rule."""
    print(f"{_GREY}{'─' * _WRAP_WIDTH}{_RESET}")


def print_info(message: str):
    """Print an informational/system message."""
    print(f"{_YELLOW}[info]{_RESET} {message}")
