"""
tests/test_pipeline.py — Core test suite for Aura backend.
 
Tests cover:
  - Input sanitisation (edge cases, injection, encoding)
  - Emotion detection (keywords, empty input)
  - Safety layer (crisis detection, false-positive avoidance)
  - Philosophy engine (returns valid dict or None)
  - Memory module (add/get/clear/overflow)
  - Pipeline (end-to-end, graceful failure, empty input)
  - API endpoints (health, chat, session history)
 
Run with:
    pytest tests/test_pipeline.py -v
"""
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
 
import pytest
from fastapi.testclient import TestClient
 
# ─── Sanitizer ────────────────────────────────────────────────────────────────
 
from utils.sanitizer import sanitize_input, is_meaningful
 
 
class TestSanitizer:
    def test_none_returns_empty(self):
        assert sanitize_input(None) == ""
 
    def test_empty_string(self):
        assert sanitize_input("") == ""
 
    def test_whitespace_only(self):
        assert sanitize_input("   \n\t  ") == ""
 
    def test_normal_text_unchanged(self):
        result = sanitize_input("I feel really anxious today")
        assert "anxious" in result
 
    def test_truncates_long_input(self):
        long = "a " * 5000
        result = sanitize_input(long)
        assert len(result) <= 4050  # max + "[truncated]"
 
    def test_strips_null_bytes(self):
        result = sanitize_input("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
 
    def test_strips_html_tags(self):
        result = sanitize_input("<script>alert(1)</script>hello")
        assert "<script>" not in result
        assert "hello" in result
 
    def test_collapses_repeated_chars(self):
        result = sanitize_input("heeelllpppppp me")
        # 5+ repetitions should collapse to 3
        assert "ppppp" not in result
 
    def test_integer_input_coerced(self):
        result = sanitize_input(42)
        assert result == "42"
 
    def test_unicode_preserved(self):
        result = sanitize_input("こんにちは I'm feeling sad")
        assert "こんにちは" in result
 
    def test_is_meaningful_empty(self):
        assert is_meaningful("") is False
 
    def test_is_meaningful_normal(self):
        assert is_meaningful("I feel sad") is True
 
    def test_is_meaningful_whitespace(self):
        assert is_meaningful("   ") is False
 
 
# ─── Emotion detection ────────────────────────────────────────────────────────
 
from emotion_detection import detect_emotion, NEUTRAL, ANXIETY, SADNESS, ANGER
 
 
class TestEmotionDetection:
    def test_always_returns_string(self):
        result = detect_emotion("")
        assert isinstance(result, str)
        assert result == NEUTRAL
 
    def test_anxiety_detection(self):
        assert detect_emotion("I'm so worried and panicking") == ANXIETY
 
    def test_sadness_detection(self):
        assert detect_emotion("I feel so sad and empty inside") == SADNESS
 
    def test_anger_detection(self):
        assert detect_emotion("I'm furious and so frustrated") == ANGER
 
    def test_neutral_on_empty(self):
        assert detect_emotion("") == NEUTRAL
 
    def test_neutral_on_garbage(self):
        result = detect_emotion("asdf 1234 !@#$")
        assert isinstance(result, str)  # Never crashes
 
    def test_never_returns_none(self):
        for text in [None, "", "   ", "xyz123", "的确 très bon"]:
            result = detect_emotion(str(text) if text is not None else "")
            assert result is not None
 
 
# ─── Safety layer ─────────────────────────────────────────────────────────────
 
from safety_layer import check_safety, SAFE, CONCERNING, CRISIS
 
 
class TestSafetyLayer:
    def test_safe_on_normal_message(self):
        status, _ = check_safety("I'm feeling a bit stressed today")
        assert status == SAFE
 
    def test_crisis_on_suicidal_language(self):
        status, response = check_safety("I want to kill myself")
        assert status == CRISIS
        assert len(response) > 50  # Substantial response
 
    def test_crisis_response_contains_hotline(self):
        _, response = check_safety("I'm thinking about ending my life")
        assert "9152987821" in response or "112" in response
 
    def test_concerning_on_hopelessness(self):
        status, _ = check_safety("I feel like giving up on life")
        assert status in (CONCERNING, CRISIS)
 
    def test_safe_on_empty(self):
        status, _ = check_safety("")
        assert status == SAFE
 
    def test_never_raises(self):
        for text in [None, "", "a" * 10000, "\x00\x01\x02"]:
            try:
                check_safety(str(text) if text else "")
            except Exception as e:
                pytest.fail(f"check_safety raised: {e}")
 
 
# ─── Philosophy engine ────────────────────────────────────────────────────────
 
from philosophy_engine import get_wisdom
 
 
class TestPhilosophyEngine:
    def test_returns_dict_or_none(self):
        result = get_wisdom("anxiety", "I'm so worried about the future")
        assert result is None or isinstance(result, dict)
 
    def test_dict_has_required_keys(self):
        result = get_wisdom("sadness", "I feel so sad and lost")
        if result is not None:
            assert "text" in result
            assert "source" in result
 
    def test_never_raises(self):
        for emotion in ["anxiety", "sadness", "anger", "neutral", "xyz", ""]:
            try:
                get_wisdom(emotion, "test message")
            except Exception as e:
                pytest.fail(f"get_wisdom raised: {e}")
 
    def test_returns_something_for_all_known_emotions(self):
        emotions = ["anxiety", "sadness", "anger", "loneliness",
                    "confusion", "stress", "hopelessness", "guilt", "fear", "neutral"]
        for emotion in emotions:
            result = get_wisdom(emotion, "I need help")
            assert result is not None, f"No wisdom found for emotion={emotion}"
 
 
# ─── Memory module ────────────────────────────────────────────────────────────
 
from memory_module import MemoryStore
 
 
class TestMemoryModule:
    def test_empty_history_on_new_session(self):
        store = MemoryStore()
        assert store.get_history("new-session") == []
 
    def test_add_and_retrieve(self):
        store = MemoryStore()
        store.add_turn("s1", "hello", "hi there", "neutral")
        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
 
    def test_max_turns_overflow(self):
        store = MemoryStore(max_turns=2)
        for i in range(5):
            store.add_turn("s1", f"msg {i}", f"reply {i}", "neutral")
        history = store.get_history("s1")
        # Should keep only last 2 pairs = 4 messages
        assert len(history) <= 4
 
    def test_clear_session(self):
        store = MemoryStore()
        store.add_turn("s1", "hello", "hi", "neutral")
        store.clear("s1")
        assert store.get_history("s1") == []
 
    def test_multiple_sessions_isolated(self):
        store = MemoryStore()
        store.add_turn("s1", "session one message", "reply", "neutral")
        store.add_turn("s2", "session two message", "reply", "neutral")
        h1 = store.get_history("s1")
        h2 = store.get_history("s2")
        assert "session one message" in h1[0]["content"]
        assert "session two message" in h2[0]["content"]
 
 
# ─── Pipeline ────────────────────────────────────────────────────────────────
 
from pipeline import run_pipeline
from memory_module import MemoryStore
 
 
class TestPipeline:
    def _run(self, message: str, config: dict = None):
        store = MemoryStore()
        cfg = config or {"provider": "openai", "api_key": "INVALID_KEY_FOR_TEST"}
        return run_pipeline(message, "test-session", store, cfg)
 
    def test_never_raises_on_normal_input(self):
        result = self._run("I'm feeling very anxious today")
        assert "response" in result
        assert "emotion" in result
 
    def test_never_raises_on_empty_input(self):
        result = self._run("")
        assert "response" in result
        assert isinstance(result["response"], str)
 
    def test_never_raises_on_garbage(self):
        result = self._run("🤯💥🔥" * 100)
        assert "response" in result
 
    def test_never_raises_on_very_long_input(self):
        result = self._run("I feel anxious " * 1000)
        assert "response" in result
 
    def test_crisis_sets_safety_flag(self):
        result = self._run("I want to end my life")
        assert result.get("safety_flag") is True
 
    def test_emotion_is_always_string(self):
        for msg in ["", "hello", "I'm sad", "asdf 123"]:
            result = self._run(msg)
            assert isinstance(result["emotion"], str)
 
    def test_response_is_always_string(self):
        for msg in ["", None.__str__(), "stress deadline work", "💔"]:
            result = self._run(msg)
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
 
 
# ─── API endpoints ────────────────────────────────────────────────────────────
 
from app import app, memory_store
 
 
class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app, raise_server_exceptions=False)
 
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
 
    def test_chat_returns_200(self, client):
        resp = client.post("/chat", json={"message": "I feel stressed"})
        assert resp.status_code == 200
 
    def test_chat_response_has_required_fields(self, client):
        resp = client.post("/chat", json={"message": "I'm worried"})
        body = resp.json()
        assert "response" in body
        assert "emotion" in body
        assert "session_id" in body
 
    def test_chat_empty_message(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["response"], str)
 
    def test_chat_null_message(self, client):
        resp = client.post("/chat", json={"message": None})
        assert resp.status_code == 200
 
    def test_chat_missing_message_field(self, client):
        resp = client.post("/chat", json={})
        # FastAPI will 422, but should not 500
        assert resp.status_code in (200, 422)
 
    def test_chat_very_long_message(self, client):
        resp = client.post("/chat", json={"message": "x " * 3000})
        assert resp.status_code == 200
 
    def test_chat_special_chars(self, client):
        resp = client.post("/chat", json={"message": "<script>alert(1)</script> help me"})
        assert resp.status_code == 200
 
    def test_chat_session_continuity(self, client):
        r1 = client.post("/chat", json={"message": "I'm anxious", "session_id": "test-abc"})
        r2 = client.post("/chat", json={"message": "It's about work", "session_id": "test-abc"})
        assert r1.json()["session_id"] == "test-abc"
        assert r2.json()["session_id"] == "test-abc"
 
    def test_history_endpoint(self, client):
        client.post("/chat", json={"message": "hello", "session_id": "hist-test"})
        resp = client.get("/session/hist-test/history")
        assert resp.status_code == 200
        assert "turns" in resp.json()
 
    def test_history_unknown_session(self, client):
        resp = client.get("/session/does-not-exist/history")
        assert resp.status_code == 200
        assert resp.json()["turns"] == []
 