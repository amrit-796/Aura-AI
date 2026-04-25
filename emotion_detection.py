"""
emotion_detection.py — Detects the dominant emotion in user input.

Uses a two-tier approach:
  Tier 1 (default): Rule-based keyword matching — fast, no dependencies.
  Tier 2 (optional): Transformer-based classification via the
                     `transformers` library — more accurate but requires
                     additional installation.

The active tier is selected automatically based on what is installed.
"""

import re
from typing import Optional

# ── Emotion labels used throughout the system ──────────────────────────────
EMOTION_ANXIETY    = "anxiety"
EMOTION_SADNESS    = "sadness"
EMOTION_ANGER      = "anger"
EMOTION_LONELINESS = "loneliness"
EMOTION_CONFUSION  = "confusion"
EMOTION_STRESS     = "stress"
EMOTION_HOPELESS   = "hopelessness"
EMOTION_GUILT      = "guilt"
EMOTION_FEAR       = "fear"
EMOTION_NEUTRAL    = "neutral"

# ── Keyword map for rule-based detection ──────────────────────────────────
# Each key is an emotion label; each value is a list of trigger patterns.
KEYWORD_MAP: dict[str, list[str]] = {
    EMOTION_ANXIETY: [
        r"\banxious\b", r"\bworried?\b", r"\bworrying\b", r"\bnervous\b",
        r"\bpanic\b", r"\bpanicking\b", r"\boverwhelmed?\b", r"\bscared\b",
        r"\bterrified\b", r"\buneasy\b", r"\brestless\b", r"\btense\b",
        r"\bcan'?t breathe\b", r"\bheart racing\b", r"\bwhat if\b",
    ],
    EMOTION_SADNESS: [
        r"\bsad\b", r"\bunhappy\b", r"\bdown\b", r"\bdepressed\b",
        r"\bdepress(ing|ion)?\b", r"\bcry(ing)?\b", r"\btears?\b",
        r"\bhurt\b", r"\bbroken\b", r"\bheartbroken\b", r"\bgrie(f|ving)\b",
        r"\bmiserable\b", r"\bnumb\b", r"\bempty\b", r"\blost\b",
        r"\bworthless\b", r"\bno point\b",
    ],
    EMOTION_ANGER: [
        r"\bangry\b", r"\banger\b", r"\bfurious\b", r"\brage\b",
        r"\birritated\b", r"\bannoy(ed|ing)?\b", r"\bfrustr(ated|ation)\b",
        r"\bhatred?\b", r"\bresentment\b", r"\bfed up\b", r"\bpissed\b",
    ],
    EMOTION_LONELINESS: [
        r"\blonely\b", r"\balone\b", r"\bisolated\b", r"\bno ?one\b",
        r"\bnobody\b", r"\bno friends?\b", r"\bno one cares\b",
        r"\bfeel invisible\b", r"\bdisconnected\b", r"\bignored\b",
    ],
    EMOTION_CONFUSION: [
        r"\bconfused?\b", r"\bconfusion\b", r"\blost\b", r"\bdon'?t know\b",
        r"\bnot sure\b", r"\buncertain\b", r"\bwhat should i\b",
        r"\bcan'?t decide\b", r"\bwhich way\b", r"\bno direction\b",
        r"\bdon'?t understand\b",
    ],
    EMOTION_STRESS: [
        r"\bstress(ed|ful)?\b", r"\bburnt? out\b", r"\bburnout\b",
        r"\bexhausted\b", r"\bexhaustion\b", r"\btired\b", r"\bdrained\b",
        r"\btoo much\b", r"\bpressure\b", r"\bdeadline\b", r"\bno time\b",
        r"\boverwhelming\b",
    ],
    EMOTION_HOPELESS: [
        r"\bhopeless\b", r"\bgive up\b", r"\bgiving up\b", r"\bno hope\b",
        r"\bnever get better\b", r"\bpointless\b", r"\bwhat'?s the point\b",
        r"\bno future\b", r"\bnothing matters\b",
    ],
    EMOTION_GUILT: [
        r"\bguilty\b", r"\bguilt\b", r"\bashamed?\b", r"\bshame\b",
        r"\bmy fault\b", r"\bi failed\b", r"\bi messed up\b",
        r"\bi ruined\b", r"\bsorry for being\b", r"\bshould have\b",
        r"\bshouldn'?t have\b",
    ],
    EMOTION_FEAR: [
        r"\bafraid\b", r"\bfear(ful)?\b", r"\bscared\b", r"\bphobia\b",
        r"\bpetrified\b", r"\bdread(ful)?\b", r"\bnightmare\b",
        r"\bterror\b", r"\bthreatened\b",
    ],
}


def detect_emotion_rulebased(text: str) -> str:
    """
    Tier-1 detection: scan text for keyword patterns.

    Returns the emotion with the highest match count, or NEUTRAL
    if nothing matches.
    """
    text_lower = text.lower()
    scores: dict[str, int] = {emotion: 0 for emotion in KEYWORD_MAP}

    for emotion, patterns in KEYWORD_MAP.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                scores[emotion] += 1

    best_emotion = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_emotion if scores[best_emotion] > 0 else EMOTION_NEUTRAL


# ── Optional Tier-2: transformer-based detection ──────────────────────────
_transformer_classifier = None
_transformer_available = False

try:
    from transformers import pipeline  # type: ignore

    # Lazy-load the model on first use to keep startup fast.
    _transformer_available = True
except ImportError:
    pass  # Transformers not installed — that's fine, Tier 1 will be used.


def _load_transformer():
    """Load the zero-shot or sentiment classifier (once)."""
    global _transformer_classifier
    if _transformer_classifier is None:
        from transformers import pipeline  # type: ignore
        # Using a lightweight sentiment model as a proxy.
        _transformer_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
        )
    return _transformer_classifier


# Map from model label → our internal emotion labels
_TRANSFORMER_LABEL_MAP = {
    "anger":    EMOTION_ANGER,
    "disgust":  EMOTION_ANGER,
    "fear":     EMOTION_FEAR,
    "joy":      EMOTION_NEUTRAL,
    "neutral":  EMOTION_NEUTRAL,
    "sadness":  EMOTION_SADNESS,
    "surprise": EMOTION_NEUTRAL,
}


def detect_emotion_transformer(text: str) -> Optional[str]:
    """
    Tier-2 detection: use a fine-tuned transformer model.

    Returns an emotion label or None on failure.
    """
    try:
        clf = _load_transformer()
        results = clf(text[:512])  # Truncate to model max length
        label = results[0][0]["label"].lower()  # type: ignore
        return _TRANSFORMER_LABEL_MAP.get(label, EMOTION_NEUTRAL)
    except Exception:
        return None


def detect_emotion(text: str, use_transformer: bool = False) -> str:
    """
    Public API: detect the dominant emotion in `text`.

    Parameters
    ----------
    text            : The raw user message.
    use_transformer : If True AND transformers is installed, use Tier-2.
                      Falls back to Tier-1 on any error.

    Returns
    -------
    One of the EMOTION_* constants defined at the top of this module.
    """
    if use_transformer and _transformer_available:
        result = detect_emotion_transformer(text)
        if result:
            return result

    return detect_emotion_rulebased(text)
