"""
response_generator.py
 
Therapist-grade AI response engine.
Produces high-fidelity, attuned, non-diagnostic responses to complex user input.
"""
 
import re
import random
from difflib import SequenceMatcher
 
 
# ---------------------------------------------------------------------------
# SAFETY RESPONSES
# ---------------------------------------------------------------------------
 
SAFETY_RESPONSES = [
    (
        "What you're carrying sounds serious, and I'm glad you're still here. "
        "Please reach out to a crisis line or someone you trust—you don't have to hold this alone. "
        "iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 (24/7)."
    ),
    (
        "I hear something heavy in what you've shared, and I want you to know that matters. "
        "Talking to a real person right now could make a difference—please consider calling a helpline "
        "or reaching out to someone close to you. iCall: 9152987821."
    ),
]
 
 
# ---------------------------------------------------------------------------
# PHRASE EXTRACTION
# ---------------------------------------------------------------------------
 
# Salient emotional / sensory / identity keywords to surface
_SALIENT_PATTERNS = [
    r"\b(invisible|hollow|numb|lost|empty|broken|tired|exhausted|fading|dim|small)\b",
    r"\b(alone|lonely|isolated|excluded|left out|outside|apart|disconnected)\b",
    r"\b(spark|light|energy|drive|joy|hope|warmth|colour|fire)\b",
    r"\b(shame|guilt|worthless|failure|burden|wrong|mistake|stupid)\b",
    r"\b(anxious|scared|terrified|panicking|dread|fear|worried)\b",
    r"\b(angry|rage|furious|resentful|bitter|frustrated)\b",
    r"\b(confused|lost|uncertain|don't know|unsure|unclear)\b",
    r"\b(disappeared|vanished|faded|gone|slipping|dissolving)\b",
    r"\b(watching|saw|noticed|felt|heard|sensed)\b",
    r"\b(together|laughing|connecting|close|bonded)\b",
]
 
 
def phrase_extraction(user_input: str) -> list[str]:
    """
    Pull 2–4 salient phrases from user input.
    Returns matched words/phrases; falls back to first meaningful noun phrases.
    """
    text = user_input.lower()
    found = []
    for pattern in _SALIENT_PATTERNS:
        matches = re.findall(pattern, text)
        found.extend(matches)
        if len(found) >= 4:
            break
 
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
 
    return unique[:4] if unique else ["what you described"]
 
 
# ---------------------------------------------------------------------------
# SCENE DETECTOR
# ---------------------------------------------------------------------------
 
def scene_detector(user_input: str) -> str:
    """
    Classify input as: 'short_statement' | 'narrative' | 'poetic'
    """
    word_count = len(user_input.split())
    has_metaphor = bool(re.search(
        r"\b(spark|light|dark|storm|ocean|weight|shadow|colour|fire|fog|wall|bridge|river|mirror)\b",
        user_input.lower()
    ))
    has_scene = bool(re.search(
        r"\b(saw|watched|went|felt|walked|stood|sat|was there|came back|came home)\b",
        user_input.lower()
    ))
 
    if word_count < 20 and not has_scene:
        return "short_statement"
    if has_metaphor and word_count < 60:
        return "poetic"
    return "narrative"
 
 
# ---------------------------------------------------------------------------
# TEMPLATE BANKS
# Grouped by emotion × scene type.
# Each entry is a dict with slots: {phrases}, {trigger}, {pattern}
# ---------------------------------------------------------------------------
 
# -- A: Grounded Acknowledgment --
ACK_TEMPLATES = {
    "loneliness": [
        "That kind of aloneness—surrounded by people and still feeling apart—is a particular kind of pain.",
        "Being on the outside of connection when you're longing for it carries a specific ache.",
        "What you're describing sounds genuinely hard—wanting to belong and feeling the gap instead.",
        "There's something especially isolating about feeling invisible in the middle of a crowd.",
    ],
    "anxiety": [
        "What you're carrying sounds heavy, and the restlessness underneath it comes through clearly.",
        "That sense of things feeling precarious or out of reach is real, and it makes sense to feel unsettled.",
        "Living with that undercurrent of worry takes more out of a person than most people realise.",
        "I can hear how tiring it is to hold all of that vigilance.",
    ],
    "sadness": [
        "There's a quiet weight in what you're sharing, and I don't want to rush past it.",
        "What you're describing sounds genuinely sorrowful—not in a dramatic way, but in a real, lived one.",
        "I'm sitting with how much heaviness comes through in what you've written.",
        "Sadness of this kind doesn't always announce itself loudly—it just settles.",
    ],
    "anger": [
        "There's something urgent and frustrated in what you're describing, and that makes sense.",
        "That anger sounds like it's been building—and that it's trying to protect something important.",
        "I hear real frustration, maybe even a sense of injustice, in what you're sharing.",
        "It sounds like something crossed a line for you, and the anger is pointing at something real.",
    ],
    "confusion": [
        "There's a real sense of being between things in what you're describing—not knowing which way is forward.",
        "That uncertainty sounds disorienting, especially when you're trying to make sense of it.",
        "I hear a kind of searching in what you're sharing, like you're trying to locate yourself.",
        "Being unsure—even about what you feel—is its own kind of difficult.",
    ],
}
 
# -- B: Specific Reflection --
REFLECT_TEMPLATES = {
    "loneliness_narrative": [
        "Seeing others connect so easily can make your own distance feel sharper—like being on the other side of glass.",
        "There's something about witnessing closeness you want and don't have that can intensify the sense of not belonging.",
        "When everyone else seems to fit together naturally, it can make the space around you feel wider than it is.",
    ],
    "loneliness_poetic": [
        "It sounds like something that once gave you warmth—a sense of belonging, or aliveness—feels harder to reach right now.",
        "The image you're using suggests a kind of dimming: not a sudden loss, but a gradual fading of what connected you to others.",
    ],
    "loneliness_short_statement": [
        "Even a single moment of feeling unseen can leave a long shadow.",
        "It sounds like that feeling of separation arrived quickly and stayed.",
    ],
    "anxiety_narrative": [
        "That cycle of anticipation and then bracing for something to go wrong can be exhausting to live inside.",
        "It sounds like your mind is working hard to stay ahead of things—trying to feel safe by staying alert.",
    ],
    "anxiety_poetic": [
        "The way you're describing it, it's like the ground beneath you feels less certain—solid enough, but not quite trustworthy.",
        "There's something in your words about holding on while also expecting things to slip.",
    ],
    "anxiety_short_statement": [
        "Even a small thing can trip the wire when the background tension is already high.",
        "That vigilance can make ordinary moments feel heavier than they should.",
    ],
    "sadness_narrative": [
        "It sounds like something meaningful has shifted—and part of you is still catching up to what's been lost.",
        "There's a kind of grief in what you're describing, even if it's not attached to a single event.",
    ],
    "sadness_poetic": [
        "The imagery you're using suggests a world that's lost some colour—not all at once, but in ways that accumulate.",
        "It sounds like something is fading that once felt reliable or sustaining.",
    ],
    "sadness_short_statement": [
        "That flatness or low feeling can sometimes sneak up gradually, until it's just the background tone.",
        "Even quiet sadness asks something of you.",
    ],
    "anger_narrative": [
        "It sounds like something happened that felt unfair or dismissive—and part of you is still holding it.",
        "There's a sense that a line was crossed, and the anger is trying to name what shouldn't have happened.",
    ],
    "anger_poetic": [
        "The way you're expressing it, the anger sounds less like explosion and more like pressure—something compressed and looking for space.",
        "It sounds like something rightful in you was blocked or ignored.",
    ],
    "anger_short_statement": [
        "Even a small injustice can carry a disproportionate sting when it touches something you already care about.",
        "That irritation sometimes points at something older underneath.",
    ],
    "confusion_narrative": [
        "It sounds like you're holding several things at once that don't quite resolve—and that tension is disorienting.",
        "When the story you've been telling about yourself starts to feel uncertain, it can be hard to know where to stand.",
    ],
    "confusion_poetic": [
        "The way you're describing it, it sounds like the map no longer matches the terrain—and you're trying to locate yourself again.",
        "There's a searching quality to what you're sharing, like something familiar has become unfamiliar.",
    ],
    "confusion_short_statement": [
        "Sometimes not knowing what we feel is its own form of overwhelm.",
        "That blurriness can be its own kind of hard.",
    ],
}
 
# -- C: Gentle Insight (pattern-linking without diagnosing) --
INSIGHT_TEMPLATES = {
    "loneliness": [
        "I'm noticing a thread of comparison running through this—measuring the distance between where you are and where others seem to be.",
        "There's something about how this connects to self-worth—as if not belonging becomes evidence of something lacking in you, rather than just a moment.",
        "It seems like this isn't just about one event, but touches something longer-standing about where you fit.",
    ],
    "anxiety": [
        "I'm hearing a pattern of trying to stay one step ahead—as if relaxing might mean being caught off guard.",
        "There's something here about the cost of constant preparation: it keeps you safe but also keeps you tense.",
        "It sounds like the worry is trying to do something useful—protect you—but it's running on overtime.",
    ],
    "sadness": [
        "I'm noticing this sadness seems quieter than dramatic—which often means it's been there longer than it announces.",
        "There's something about how we can carry grief for a long time before we name it as that.",
        "It sounds like part of you is mourning something—a version of yourself, a connection, or a possibility.",
    ],
    "anger": [
        "I'm hearing that the anger might be standing in for something else underneath—hurt, perhaps, or a sense of being undervalued.",
        "Sometimes anger is the clearest signal we have that something important to us wasn't respected.",
        "There's a kind of self-protective energy in what you're describing—anger as a boundary that didn't get listened to.",
    ],
    "confusion": [
        "I'm noticing that some of the confusion might come from holding contradictory things that both feel true at once.",
        "Sometimes not knowing is less about the absence of answers and more about being in a genuine in-between.",
        "It sounds like you're in a kind of transition—where the old frame doesn't fit, and the new one isn't clear yet.",
    ],
}
 
# -- D: Open-Ended Questions --
QUESTION_TEMPLATES = {
    "loneliness": [
        "When you think back to that moment, what felt most sharp—the comparison, or the sense of not belonging?",
        "What does belonging feel like for you when it's present—is there a moment you can call to mind?",
        "Is there a part of you that knows where this feeling of distance started, or does it feel older than any single event?",
        "What would it mean to feel less invisible, even in small ways—what would that look like for you?",
    ],
    "anxiety": [
        "When the worry is loudest, what is it most often telling you to watch out for?",
        "Is there a version of yourself that exists beneath the tension—and what do you imagine that feels like?",
        "What would it look like to give yourself permission to rest from the vigilance, even briefly?",
        "When you feel most anxious, what do you find yourself needing most from the people around you?",
    ],
    "sadness": [
        "What do you think you're most grieving right now, if you had to name it?",
        "Is there something you've been carrying quietly that hasn't had space to be said yet?",
        "When you imagine the sadness lifting even a little, what does the day feel like?",
        "What does the sadness seem to be asking of you, or asking for?",
    ],
    "anger": [
        "Beneath the anger, what do you think is the hurt or need it's trying to protect?",
        "When you imagine being heard fully about this—what would you most want the other person to understand?",
        "Is there something you've been holding back from saying, and what's made it hard to say?",
        "What would feel like a fair resolution to what happened—even if it's not possible right now?",
    ],
    "confusion": [
        "If you had to choose the one thing you feel most uncertain about right now, what would it be?",
        "Is there a part of you that does know what it needs, even if the rest of you isn't sure?",
        "When you imagine things being clearer, what would that clarity feel like in your body or your day?",
        "What would it be okay to not know yet, if you gave yourself permission?",
    ],
}
 
 
# ---------------------------------------------------------------------------
# ANTI-REPETITION
# ---------------------------------------------------------------------------
 
def _similarity(a: str, b: str) -> float:
    """Returns similarity ratio between two strings (0–1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
 
 
def _get_recent_assistant_text(history: list, n: int = 3) -> list[str]:
    """Extract last N assistant messages from history."""
    texts = []
    for turn in reversed(history):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            texts.append(turn.get("content", ""))
            if len(texts) >= n:
                break
    return texts
 
 
def _pick_fresh(candidates: list[str], recent: list[str], threshold: float = 0.45) -> str:
    """
    Pick a candidate that is sufficiently dissimilar from recent responses.
    Falls back to least-similar if all are too close.
    """
    random.shuffle(candidates)
    for candidate in candidates:
        if all(_similarity(candidate, r) < threshold for r in recent):
            return candidate
    # Fallback: return the one least similar to the most recent
    if recent:
        return min(candidates, key=lambda c: _similarity(c, recent[0]))
    return candidates[0]
 
 
# ---------------------------------------------------------------------------
# HISTORY REFERENCE
# ---------------------------------------------------------------------------
 
def _build_history_bridge(history: list, emotion: str) -> str:
    """
    If the same or related emotion appeared earlier, return a soft callback phrase.
    Otherwise returns empty string.
    """
    if not history:
        return ""
    recent = _get_recent_assistant_text(history, n=5)
    emotion_synonyms = {
        "loneliness": ["alone", "lonely", "isolated", "invisible", "excluded"],
        "anxiety": ["anxious", "worried", "scared", "nervous", "tense"],
        "sadness": ["sad", "grief", "loss", "low", "heavy", "sorrowful"],
        "anger": ["angry", "frustrated", "resentful", "furious"],
        "confusion": ["confused", "uncertain", "lost", "unclear"],
    }
    keywords = emotion_synonyms.get(emotion, [])
    for msg in recent:
        if any(kw in msg.lower() for kw in keywords):
            return "You've mentioned something like this before—"
    return ""
 
 
# ---------------------------------------------------------------------------
# FALLBACK
# ---------------------------------------------------------------------------
 
FALLBACK_RESPONSE = (
    "I want to make sure I'm understanding you well. "
    "What you're describing sounds meaningful, and I don't want to rush past it. "
    "Would you be willing to share a little more about what's feeling most present for you right now?"
)
 
 
# ---------------------------------------------------------------------------
# PHILOSOPHY INTEGRATION
# ---------------------------------------------------------------------------
 
PHILOSOPHY_INTEGRATIONS = {
    "loneliness": "There's something in many traditions about how belonging starts in relation, not in performance—",
    "anxiety": "Some thinkers have noticed that anxiety often lives in the gap between now and an imagined future—",
    "sadness": "Grief, in many understandings, is the measure of what mattered—",
    "anger": "Anger, at its root, is often pointing toward something that should not have been—",
    "confusion": "Not-knowing, in some traditions, is considered a starting point rather than a failure—",
}
 
 
# ---------------------------------------------------------------------------
# CORE: COMPOSER
# ---------------------------------------------------------------------------
 
def _compose_response(
    emotion: str,
    scene: str,
    phrases: list[str],
    history: list,
    philosophy: dict | None,
) -> str:
    """
    Assembles the 4-move response: Acknowledgment → Reflection → Insight → Question.
    """
    recent_texts = _get_recent_assistant_text(history)
 
    # -- A: Acknowledgment --
    ack_pool = ACK_TEMPLATES.get(emotion, ACK_TEMPLATES["confusion"])
    ack = _pick_fresh(ack_pool, recent_texts)
 
    # -- B: Reflection --
    reflect_key = f"{emotion}_{scene}"
    reflect_fallback_key = f"{emotion}_narrative"
    reflect_pool = REFLECT_TEMPLATES.get(
        reflect_key, REFLECT_TEMPLATES.get(reflect_fallback_key, [
            "What you're describing sounds specific and real, even if it's hard to put into words."
        ])
    )
    reflection = _pick_fresh(reflect_pool, recent_texts)
 
    # Optionally embed a salient phrase from the user's input
    if phrases and random.random() > 0.4:
        phrase = phrases[0]
        if phrase not in reflection.lower():
            reflection = reflection.rstrip(".") + f"—especially that sense of feeling '{phrase}'."
 
    # -- C: Insight --
    insight_pool = INSIGHT_TEMPLATES.get(emotion, INSIGHT_TEMPLATES["confusion"])
    insight = _pick_fresh(insight_pool, recent_texts)
 
    # Optional philosophy integration (subtle, one phrase)
    if philosophy and random.random() > 0.5:
        bridge = PHILOSOPHY_INTEGRATIONS.get(emotion, "")
        if bridge:
            insight = bridge + insight[0].lower() + insight[1:]
 
    # Optional history callback
    history_bridge = _build_history_bridge(history, emotion)
    if history_bridge and random.random() > 0.5:
        insight = history_bridge + insight[0].lower() + insight[1:]
 
    # -- D: Question --
    question_pool = QUESTION_TEMPLATES.get(emotion, QUESTION_TEMPLATES["confusion"])
    question = _pick_fresh(question_pool, recent_texts)
 
    # Assemble
    parts = [ack, reflection, insight, question]
    return " ".join(parts)
 
 
# ---------------------------------------------------------------------------
# MAIN: generate_response
# ---------------------------------------------------------------------------
 
def generate_response(
    user_input: str,
    emotion: str,
    history: list,
    philosophy: dict | None,
    safety_status: str,
) -> str:
    """
    Generate a therapist-grade response to complex user input.
 
    Args:
        user_input:     The user's raw message (narrative, poetic, or short).
        emotion:        Detected primary emotion label (e.g. 'loneliness', 'anxiety').
        history:        List of prior conversation turns [{'role': ..., 'content': ...}].
        philosophy:     Optional dict with philosophical context (e.g. {'tradition': 'stoic'}).
        safety_status:  'safe' | 'low_risk' | 'crisis'
 
    Returns:
        A 2–5 sentence attuned response string.
    """
    try:
        # ── SAFETY OVERRIDE ─────────────────────────────────────────────────
        if safety_status and safety_status.lower() in ("crisis", "high_risk"):
            return random.choice(SAFETY_RESPONSES)
 
        # ── NORMALISE EMOTION ────────────────────────────────────────────────
        valid_emotions = {"loneliness", "anxiety", "sadness", "anger", "confusion"}
        emotion_clean = emotion.lower().strip() if emotion else "confusion"
        if emotion_clean not in valid_emotions:
            # Attempt fuzzy match
            for ve in valid_emotions:
                if ve in emotion_clean or emotion_clean in ve:
                    emotion_clean = ve
                    break
            else:
                emotion_clean = "confusion"
 
        # ── EXTRACT SCENE & PHRASES ──────────────────────────────────────────
        scene = scene_detector(user_input)
        phrases = phrase_extraction(user_input)
 
        # ── COMPOSE ──────────────────────────────────────────────────────────
        response = _compose_response(
            emotion=emotion_clean,
            scene=scene,
            phrases=phrases,
            history=history,
            philosophy=philosophy,
        )
 
        # ── ANTI-REPETITION GUARD ────────────────────────────────────────────
        # If the composed response is too similar to the last assistant turn, recompose once
        recent = _get_recent_assistant_text(history, n=1)
        if recent and _similarity(response, recent[0]) > 0.55:
            response = _compose_response(
                emotion=emotion_clean,
                scene=scene,
                phrases=phrases,
                history=history,
                philosophy=philosophy,
            )
 
        return response.strip() if response.strip() else FALLBACK_RESPONSE
 
    except Exception:
        # ── FAIL-SAFE ────────────────────────────────────────────────────────
        return FALLBACK_RESPONSE
 
 
# ---------------------------------------------------------------------------
# QUICK SMOKE TEST (run directly: python response_generator.py)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    test_cases = [
        {
            "user_input": (
                "I went out tonight feeling excited to see everyone, but when I got there "
                "I just watched groups laughing and couldn't break in. I felt completely invisible. "
                "My spark is fading—I used to be the one who brought people together."
            ),
            "emotion": "loneliness",
            "history": [],
            "philosophy": {"tradition": "existential"},
            "safety_status": "safe",
        },
        {
            "user_input": "Everything feels like it's about to fall apart and I don't know why.",
            "emotion": "anxiety",
            "history": [],
            "philosophy": None,
            "safety_status": "safe",
        },
        {
            "user_input": "I'm so tired of pretending everything is fine.",
            "emotion": "sadness",
            "history": [
                {"role": "user", "content": "I've been feeling low for weeks."},
                {"role": "assistant", "content": "That kind of sustained heaviness sounds exhausting to carry."},
            ],
            "philosophy": None,
            "safety_status": "safe",
        },
        {
            "user_input": "I can't take it anymore.",
            "emotion": "sadness",
            "history": [],
            "philosophy": None,
            "safety_status": "crisis",
        },
    ]
 
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─'*60}")
        print(f"TEST {i} | emotion={tc['emotion']} | safety={tc['safety_status']}")
        print(f"INPUT: {tc['user_input'][:80]}...")
        print(f"\nRESPONSE:\n{generate_response(**tc)}")
    print(f"\n{'─'*60}")
 