"""
pipeline.py — Central orchestrator for the Aura processing pipeline.
 
Pipeline stages (in order):
  1. Sanitize input
  2. Detect emotion
  3. Safety check          (may short-circuit to crisis response)
  4. Retrieve wisdom hint
  5. Fetch memory context
  6. Generate LLM response (with rule-based fallback)
  7. Store turn in memory
  8. Return structured result
 
Every stage is wrapped in its own try/except.  A failure in any single
stage degrades gracefully — the pipeline never raises to the caller.
"""
 
import logging
import time
from typing import Optional
 
from utils.sanitizer import sanitize_input
from emotion_detection import detect_emotion
from safety_layer import check_safety, SAFE, CONCERNING, CRISIS
from philosophy_engine import get_wisdom
from memory_module import MemoryStore
from response_generator import generate_response
 
logger = logging.getLogger("aura.pipeline")
 
# ── Safe default used if the entire pipeline collapses ───────────────────────
_HARD_FALLBACK = (
    "I'm here with you. Something didn't process correctly on my end, "
    "but that doesn't matter right now — you do. "
    "What's on your mind?"
)
 
 
def run_pipeline(
    raw_input: str,
    session_id: str,
    memory_store: MemoryStore,
    config: dict,
) -> dict:
    """
    Execute the full processing pipeline for a single user turn.
 
    Parameters
    ----------
    raw_input    : Raw string from the API request (may be empty / garbage).
    session_id   : Unique identifier for this conversation session.
    memory_store : Shared MemoryStore instance from app.py.
    config       : Application configuration dict.
 
    Returns
    -------
    dict with keys:
      response    (str)  — assistant's reply
      emotion     (str)  — detected emotion label
      safety_flag (bool) — True if a safety override was triggered
    """
    start_time = time.perf_counter()
    emotion = "neutral"
    safety_flag = False
 
    try:
        # ── Stage 1: Sanitize ─────────────────────────────────────────────
        clean_input = _stage_sanitize(raw_input)
        logger.debug("stage=sanitize | result_len=%d", len(clean_input))
 
        # ── Stage 2: Emotion detection ────────────────────────────────────
        emotion = _stage_detect_emotion(clean_input)
        logger.debug("stage=emotion | result=%s", emotion)
 
        # ── Stage 3: Safety check ─────────────────────────────────────────
        safety_status, safety_response = _stage_safety(clean_input)
        logger.debug("stage=safety | status=%s", safety_status)
 
        if safety_status == CRISIS:
            safety_flag = True
            _store_turn(memory_store, session_id, raw_input, safety_response, "crisis")
            return _result(safety_response, "crisis", safety_flag=True)
 
        # ── Stage 4: Wisdom ───────────────────────────────────────────────
        wisdom = _stage_wisdom(emotion, clean_input)
        logger.debug("stage=wisdom | found=%s", wisdom is not None)
 
        # ── Stage 5: Memory context ───────────────────────────────────────
        history = _stage_memory(memory_store, session_id)
        logger.debug("stage=memory | turns=%d", len(history))
 
        # ── Stage 6: Response generation ──────────────────────────────────
        response = generate_response(
            user_input=user_input,
            emotion=emotion,
            history=history,
            philosophy=wisdom,
            safety_status=safety_status,
        )

        logger.debug("stage=generate | response_len=%d", len(response))

        return response
 
        # ── Stage 7: Store turn ───────────────────────────────────────────
        _store_turn(memory_store, session_id, raw_input, response, emotion)
 
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            "pipeline complete | session=%s emotion=%s safety=%s elapsed=%.1fms",
            session_id, emotion, safety_status, elapsed,
        )
 
        return _result(response, emotion, safety_flag=safety_status == CONCERNING)
 
    except Exception as exc:
        # Absolute last resort — nothing escapes to the caller as an exception
        logger.error("pipeline hard failure: %s", exc, exc_info=True)
        return _result(_HARD_FALLBACK, emotion, safety_flag=False)
 
 
# ─── Stage helpers ────────────────────────────────────────────────────────────
 
def _stage_sanitize(raw: str) -> str:
    try:
        return sanitize_input(raw)
    except Exception as exc:
        logger.warning("sanitize stage failed: %s", exc)
        # Return a safe empty string — later stages handle empty input
        return ""
 
 
def _stage_detect_emotion(text: str) -> str:
    try:
        return detect_emotion(text)
    except Exception as exc:
        logger.warning("emotion detection failed: %s", exc)
        return "neutral"
 
 
def _stage_safety(text: str):
    try:
        return check_safety(text)
    except Exception as exc:
        logger.warning("safety check failed: %s", exc)
        return SAFE, ""
 
 
def _stage_wisdom(emotion: str, text: str) -> Optional[dict]:
    try:
        return get_wisdom(emotion, text)
    except Exception as exc:
        logger.warning("wisdom retrieval failed: %s", exc)
        return None
 
 
def _stage_memory(store: MemoryStore, session_id: str) -> list[dict]:
    try:
        return store.get_history(session_id)
    except Exception as exc:
        logger.warning("memory retrieval failed: %s", exc)
        return []
 
 
def _stage_generate(
    user_input: str,
    emotion: str,
    wisdom: Optional[dict],
    history: list[dict],
    safety_status: str,
    config: dict,
) -> str:
    try:
        return generate_response(
    user_input=user_input,
    emotion=emotion,
    history=history,
    philosophy=wisdom,
    safety_status=safety_status,
)
    except Exception as exc:
        logger.warning("response generation failed: %s", exc)
        return _HARD_FALLBACK
 
 
def _store_turn(
    store: MemoryStore,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    emotion: str,
):
    try:
        store.add_turn(session_id, user_msg, assistant_msg, emotion)
    except Exception as exc:
        logger.warning("memory store failed: %s", exc)
 
 
def _result(response: str, emotion: str, *, safety_flag: bool) -> dict:
    return {
        "response": response,
        "emotion": emotion,
        "safety_flag": safety_flag,
    }
 