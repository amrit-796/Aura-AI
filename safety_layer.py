"""
safety_layer.py — Crisis detection and override responses.

This module is the FIRST thing checked on every user message.
If a crisis is detected, it returns a pre-written, empathetic response
and signals the main loop to skip the normal LLM pipeline.

IMPORTANT DESIGN NOTE
---------------------
This system is NOT a crisis counsellor. It will never try to fully
handle a mental-health emergency. Its only job is to:
  1. Acknowledge the person with warmth and seriousness.
  2. Provide real hotline/resource information.
  3. Encourage the person to reach out to a human immediately.
"""

import re
from typing import Tuple

# ── Status codes returned by check_safety() ───────────────────────────────
SAFE        = "safe"
CONCERNING  = "concerning"   # Distress, but not immediate crisis
CRISIS      = "crisis"       # Possible self-harm / suicidal intent

# ── Crisis keyword patterns ────────────────────────────────────────────────
_CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bend (my|this) life\b",
    r"\bsuicid(e|al)\b",
    r"\bwant to die\b",
    r"\bbetter off dead\b",
    r"\bno reason to live\b",
    r"\bself.?harm\b",
    r"\bcutting myself\b",
    r"\bhurt myself\b",
    r"\boverdos(e|ing)\b",
    r"\b(take|took) (too many|all (my|the)) pills\b",
    r"\bdon'?t want to (be here|exist|wake up)\b",
    r"\bcan'?t go on\b",
]

# ── Concerning (not immediate crisis) patterns ────────────────────────────
_CONCERNING_PATTERNS = [
    r"\bgive up (on (life|everything))?\b",
    r"\bno reason (to|for)\b",
    r"\bwhy (am|do) i (even )?(exist|bother)\b",
    r"\bnothing (left|matters)\b",
    r"\bfeeling (so )?(dead|empty|nothing)\b",
]

# ── Pre-written crisis response ───────────────────────────────────────────
_CRISIS_RESPONSE = """\
I want you to know I'm taking what you just said very seriously, \
and I'm genuinely glad you're still here talking.

What you're feeling right now sounds incredibly heavy — and you don't \
have to carry it alone.

Please reach out to someone who can truly be there for you right now:

  🆘  iCall (India):          9152987821
  🆘  Vandrevala Foundation:  1860-2662-345 (24/7)
  🆘  NIMHANS Helpline:       080-46110007
  🆘  International (iSOS):   https://www.iasp.info/resources/Crisis_Centres/

If you're in immediate danger, please call emergency services (112 in India).

You matter. This moment is not the whole story. \
A real human who cares is just one call away. 💙\
"""

# ── Pre-written concerning response ──────────────────────────────────────
_CONCERNING_RESPONSE = """\
It sounds like you're going through something really painful right now, \
and I want you to know I'm here with you in this moment.

Those feelings of wanting to give up can feel overwhelming — \
but they're also a signal that something inside you needs care and attention.

Would you like to talk about what's brought you to this place? \
I'm listening, and there's no rush.\
"""


def check_safety(text: str) -> Tuple[str, str]:
    """
    Scan user input for crisis or concerning signals.

    Parameters
    ----------
    text : Raw user message.

    Returns
    -------
    (status, response_text)
      status        : SAFE | CONCERNING | CRISIS
      response_text : Pre-written response (empty string when SAFE)
    """
    lower = text.lower()

    # Crisis takes priority
    for pattern in _CRISIS_PATTERNS:
        if re.search(pattern, lower):
            return CRISIS, _CRISIS_RESPONSE

    # Concerning but not immediate crisis
    for pattern in _CONCERNING_PATTERNS:
        if re.search(pattern, lower):
            return CONCERNING, _CONCERNING_RESPONSE

    return SAFE, ""
