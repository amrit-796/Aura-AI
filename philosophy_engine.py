"""
philosophy_engine.py — Retrieves contextually relevant wisdom snippets.

The wisdom database is organised by emotion/theme. Each entry contains:
  - text   : The insight or quote (paraphrased or lightly attributed).
  - source : Book / film / philosophy / tradition.
  - theme  : Primary emotion tag.

The retrieval function:
  1. Filters by matching emotion tag.
  2. Optionally scores entries by keyword overlap with user input.
  3. Returns one snippet (or None to skip it entirely).

Design intention
----------------
Wisdom should feel like a thoughtful friend mentioning something
they once read — not like a motivational-poster bot.
The response_generator decides *how* to weave the snippet into
the final reply; this module only surfaces a candidate.
"""

import random
import re
from typing import Optional

from emotion_detection import (
    EMOTION_ANXIETY, EMOTION_SADNESS, EMOTION_ANGER,
    EMOTION_LONELINESS, EMOTION_CONFUSION, EMOTION_STRESS,
    EMOTION_HOPELESS, EMOTION_GUILT, EMOTION_FEAR, EMOTION_NEUTRAL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Wisdom database
# Each dict: {text, source, theme, keywords}
# keywords help score relevance against the user's actual words.
# ─────────────────────────────────────────────────────────────────────────────

WISDOM_DB: list[dict] = [

    # ── ANXIETY ──────────────────────────────────────────────────────────────
    {
        "text": "You suffer more in imagination than in reality. Most of what we dread never arrives — and when it does, we find we are stronger than we thought.",
        "source": "Seneca (Stoic philosophy)",
        "theme": EMOTION_ANXIETY,
        "keywords": ["imagine", "dread", "what if", "future", "fear"],
    },
    {
        "text": "The present moment is the only place where life actually happens. Worry is a story about a future that hasn't been written yet.",
        "source": "Mindfulness tradition",
        "theme": EMOTION_ANXIETY,
        "keywords": ["present", "moment", "future", "worry", "now"],
    },
    {
        "text": "In 'Your Lie in April', Kousei learns that the music doesn't stop when we're afraid — we just have to keep playing through the fear.",
        "source": "Your Lie in April (anime)",
        "theme": EMOTION_ANXIETY,
        "keywords": ["perform", "afraid", "music", "continue", "keep going"],
    },
    {
        "text": "Anxiety is often the gap between where we are and where we think we should be. Narrowing that gap starts with accepting where we are right now.",
        "source": "General life wisdom",
        "theme": EMOTION_ANXIETY,
        "keywords": ["should", "expect", "gap", "pressure", "standard"],
    },

    # ── SADNESS ──────────────────────────────────────────────────────────────
    {
        "text": "Grief is just love with nowhere to go. It is not weakness — it is proof that something mattered deeply to you.",
        "source": "Jamie Anderson",
        "theme": EMOTION_SADNESS,
        "keywords": ["grief", "loss", "love", "miss", "gone"],
    },
    {
        "text": "In 'Inside Out', Joy finally understands that Sadness is not the enemy. She is the one who helps us process what we've been through and reach out to others.",
        "source": "Inside Out (Pixar film)",
        "theme": EMOTION_SADNESS,
        "keywords": ["sad", "cry", "feelings", "understand", "process"],
    },
    {
        "text": "Even the darkest night will end and the sun will rise.",
        "source": "Victor Hugo, Les Misérables",
        "theme": EMOTION_SADNESS,
        "keywords": ["dark", "night", "end", "hope", "morning"],
    },
    {
        "text": "The Bhagavad Gita reminds us: 'You have the right to perform your actions, but not to the fruits of your actions.' Sometimes healing is simply the act of continuing — not the immediate result.",
        "source": "Bhagavad Gita (Chapter 2)",
        "theme": EMOTION_SADNESS,
        "keywords": ["healing", "continue", "result", "effort", "outcome"],
    },

    # ── ANGER ─────────────────────────────────────────────────────────────────
    {
        "text": "Marcus Aurelius wrote: 'How much more harmful are the consequences of anger than the circumstances that aroused it.' Anger is valid — but it often costs us more than the thing that caused it.",
        "source": "Marcus Aurelius, Meditations",
        "theme": EMOTION_ANGER,
        "keywords": ["angry", "rage", "cost", "consequences", "reaction"],
    },
    {
        "text": "Anger is often a secondary emotion — underneath it is usually hurt, fear, or a sense of injustice. Listening to what it's protecting can be surprisingly revealing.",
        "source": "Psychology insight",
        "theme": EMOTION_ANGER,
        "keywords": ["hurt", "injustice", "underneath", "protect", "secondary"],
    },
    {
        "text": "In 'Fullmetal Alchemist: Brotherhood', Roy Mustang's anger at injustice was real and valid — but Ed reminds him that burning everything down would make him the very thing he hated.",
        "source": "Fullmetal Alchemist: Brotherhood (anime)",
        "theme": EMOTION_ANGER,
        "keywords": ["justice", "burn", "enemy", "revenge", "hatred"],
    },

    # ── LONELINESS ────────────────────────────────────────────────────────────
    {
        "text": "The loneliness of feeling unseen is one of the most painful human experiences — and yet it is also one of the most universally shared ones. You are not alone in feeling alone.",
        "source": "General life wisdom",
        "theme": EMOTION_LONELINESS,
        "keywords": ["unseen", "alone", "nobody", "invisible", "isolated"],
    },
    {
        "text": "In 'A Silent Voice', Shoya spends years in isolation before realising that connection was possible all along — he just needed to open his eyes to those already beside him.",
        "source": "A Silent Voice (anime film)",
        "theme": EMOTION_LONELINESS,
        "keywords": ["connection", "isolated", "friend", "beside", "open"],
    },
    {
        "text": "Rumi wrote: 'Out beyond ideas of wrongdoing and rightdoing, there is a field. I'll meet you there.' Sometimes loneliness dissolves not through adding people, but through letting walls come down.",
        "source": "Rumi",
        "theme": EMOTION_LONELINESS,
        "keywords": ["walls", "connection", "judge", "open", "field"],
    },

    # ── CONFUSION ─────────────────────────────────────────────────────────────
    {
        "text": "Not knowing is not the same as being lost. The Zen tradition honours 'beginner's mind' — the openness of someone who hasn't yet decided they know the answer.",
        "source": "Zen Buddhism",
        "theme": EMOTION_CONFUSION,
        "keywords": ["don't know", "lost", "direction", "answer", "decide"],
    },
    {
        "text": "In 'Good Will Hunting', it's not the brilliant answers that help Will — it's someone sitting with him in the confusion and saying 'It's not your fault.'",
        "source": "Good Will Hunting (film)",
        "theme": EMOTION_CONFUSION,
        "keywords": ["fault", "understand", "brilliant", "answer", "help"],
    },
    {
        "text": "The Bhagavad Gita addresses Arjuna's confusion not by telling him what to do immediately, but by helping him understand who he is first. Clarity often follows self-knowledge.",
        "source": "Bhagavad Gita (Chapter 1–2)",
        "theme": EMOTION_CONFUSION,
        "keywords": ["who am i", "identity", "clarity", "decision", "lost"],
    },

    # ── STRESS ───────────────────────────────────────────────────────────────
    {
        "text": "You cannot pour from an empty cup. Rest is not a reward for finishing your work — it is part of the work itself.",
        "source": "General life wisdom",
        "theme": EMOTION_STRESS,
        "keywords": ["tired", "exhausted", "rest", "work", "cup", "drain"],
    },
    {
        "text": "Epictetus said: 'Make the best use of what is in your power, and take the rest as it happens.' We can only control our next small action — not the whole mountain.",
        "source": "Epictetus (Stoic philosophy)",
        "theme": EMOTION_STRESS,
        "keywords": ["control", "pressure", "power", "mountain", "overwhelm"],
    },
    {
        "text": "In 'Haikyuu!!', Hinata learns that overwhelm shrinks the moment you focus on one set, one rally, one point — not the entire match.",
        "source": "Haikyuu!! (anime)",
        "theme": EMOTION_STRESS,
        "keywords": ["overwhelm", "focus", "one step", "too much", "pressure"],
    },

    # ── HOPELESSNESS ─────────────────────────────────────────────────────────
    {
        "text": "Hope is not the belief that things will definitely get better. It is the willingness to act as though they might — and that willingness is itself a form of courage.",
        "source": "General life wisdom",
        "theme": EMOTION_HOPELESS,
        "keywords": ["hope", "better", "courage", "give up", "point"],
    },
    {
        "text": "Viktor Frankl, who survived the Holocaust, wrote: 'When we are no longer able to change a situation, we are challenged to change ourselves.' Something inside us can remain free even when everything else feels locked.",
        "source": "Viktor Frankl, Man's Search for Meaning",
        "theme": EMOTION_HOPELESS,
        "keywords": ["change", "situation", "survive", "free", "meaning"],
    },
    {
        "text": "In 'Demon Slayer', Tanjiro keeps going not because victory is certain — but because giving up would mean the people he loves never mattered. Love becomes the reason to continue.",
        "source": "Demon Slayer (anime)",
        "theme": EMOTION_HOPELESS,
        "keywords": ["keep going", "give up", "love", "reason", "continue"],
    },

    # ── GUILT ────────────────────────────────────────────────────────────────
    {
        "text": "There is a difference between guilt (I did something wrong) and shame (I am something wrong). One can be corrected; the other is a lie.",
        "source": "Brené Brown, Daring Greatly",
        "theme": EMOTION_GUILT,
        "keywords": ["shame", "wrong", "failure", "fault", "bad person"],
    },
    {
        "text": "Self-compassion is not self-indulgence. Kristin Neff's research shows that people who forgive themselves for mistakes are actually more likely to improve than those who self-punish.",
        "source": "Kristin Neff, Self-Compassion",
        "theme": EMOTION_GUILT,
        "keywords": ["forgive", "punish", "mistake", "improve", "compassion"],
    },
    {
        "text": "The Bhagavad Gita says: 'Do your duty and be detached from results.' You acted with the knowledge and capacity you had at the time. That is all any of us can do.",
        "source": "Bhagavad Gita",
        "theme": EMOTION_GUILT,
        "keywords": ["duty", "result", "past", "did my best", "knowledge"],
    },

    # ── FEAR ─────────────────────────────────────────────────────────────────
    {
        "text": "Courage is not the absence of fear — it is deciding that something matters more than the fear. You don't have to feel brave to act bravely.",
        "source": "General life wisdom",
        "theme": EMOTION_FEAR,
        "keywords": ["brave", "courage", "scared", "afraid", "act"],
    },
    {
        "text": "In 'Spirited Away', Chihiro is terrified — but she keeps moving forward one small step at a time. The spirit world doesn't get less scary; she just keeps going anyway.",
        "source": "Spirited Away (Studio Ghibli film)",
        "theme": EMOTION_FEAR,
        "keywords": ["terrified", "keep going", "scary", "step", "forward"],
    },

    # ── NEUTRAL / GENERAL ─────────────────────────────────────────────────────
    {
        "text": "You are allowed to be both a masterpiece and a work in progress at the same time.",
        "source": "Sophia Bush",
        "theme": EMOTION_NEUTRAL,
        "keywords": ["progress", "imperfect", "growth", "work in progress"],
    },
    {
        "text": "The purpose of life is not to be happy. It is to be useful, to be honourable, to be compassionate, to have it make some difference that you have lived at all.",
        "source": "Ralph Waldo Emerson",
        "theme": EMOTION_NEUTRAL,
        "keywords": ["purpose", "meaning", "life", "matter", "difference"],
    },
]


def _score_entry(entry: dict, user_text: str) -> int:
    """
    Count how many of the entry's keywords appear in the user's message.
    Used to rank candidates by relevance.
    """
    lower = user_text.lower()
    return sum(1 for kw in entry.get("keywords", []) if kw in lower)


def get_wisdom(emotion: str, user_text: str) -> Optional[dict]:
    """
    Retrieve a relevant wisdom snippet for the current turn.

    Strategy
    --------
    1. Filter candidates whose theme matches the detected emotion.
    2. Also include NEUTRAL entries as fallback candidates.
    3. Score by keyword overlap with user_text.
    4. Return the highest-scoring entry (random tiebreak).
    5. Return None if no candidates at all (shouldn't happen in practice).

    Parameters
    ----------
    emotion   : Detected emotion label (e.g. EMOTION_ANXIETY).
    user_text : Raw user message, used for keyword scoring.

    Returns
    -------
    A wisdom dict {text, source, theme} or None.
    """
    candidates = [
        e for e in WISDOM_DB
        if e["theme"] in (emotion, EMOTION_NEUTRAL)
    ]

    if not candidates:
        return None

    # Score and sort
    scored = [(entry, _score_entry(entry, user_text)) for entry in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Take the top scorers (up to 3) and pick randomly among them for variety
    top_score = scored[0][1]
    top_pool = [e for e, s in scored if s >= top_score]
    return random.choice(top_pool)
