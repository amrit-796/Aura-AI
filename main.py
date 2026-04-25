"""
main.py — Entry point for the Emotional AI Companion.

Starts the CLI chat loop and ties together all modules:
emotion detection, safety, philosophy, memory, and response generation.
"""

import os
import sys
from datetime import datetime

from emotion_detection import detect_emotion
from safety_layer import check_safety, SAFE, CRISIS, CONCERNING
from philosophy_engine import get_wisdom
from memory_module import ConversationMemory
from response_generator import generate_response
from utils.display import print_banner, print_assistant, print_user_prompt, print_divider
from utils.config import load_config


def chat_loop(config: dict):
    """
    Main conversation loop.

    Runs indefinitely until the user types 'quit', 'exit', or 'bye'.
    Each turn:
      1. Reads user input
      2. Checks safety
      3. Detects emotion
      4. Retrieves a wisdom snippet (optional)
      5. Generates a response
      6. Stores the exchange in memory
    """
    memory = ConversationMemory(max_turns=config.get("max_memory_turns", 10))
    session_start = datetime.now().strftime("%Y-%m-%d %H:%M")

    print_banner()
    print_assistant(
        "Hello. I'm really glad you're here. This is a safe space — "
        "feel free to share whatever's on your mind. "
        "I'm here to listen, not to judge.\n"
        f"(Session started: {session_start})"
    )
    print_divider()

    while True:
        user_input = print_user_prompt()

        # ── Exit commands ──────────────────────────────────────────────────
        if user_input.strip().lower() in {"quit", "exit", "bye", "goodbye"}:
            print_assistant(
                "Take good care of yourself. Remember — reaching out, "
                "even to an AI, takes courage. You've got this. 💙"
            )
            break

        if not user_input.strip():
            continue

        # ── 1. Safety check ───────────────────────────────────────────────
        safety_status, safety_response = check_safety(user_input)

        if safety_status == CRISIS:
            print_assistant(safety_response)
            memory.add_turn(user_input, safety_response, emotion="crisis")
            print_divider()
            continue  # Skip normal pipeline for crisis messages

        # ── 2. Emotion detection ──────────────────────────────────────────
        emotion = detect_emotion(user_input)

        # ── 3. Optional wisdom snippet ────────────────────────────────────
        wisdom = get_wisdom(emotion, user_input)

        # ── 4. Build context for the LLM ─────────────────────────────────
        history = memory.get_history()

        # ── 5. Generate response ──────────────────────────────────────────
        response = generate_response(
            user_input=user_input,
            emotion=emotion,
            wisdom=wisdom,
            history=history,
            safety_status=safety_status,
            config=config,
        )

        # ── 6. Display & store ────────────────────────────────────────────
        print_assistant(response)
        memory.add_turn(user_input, response, emotion=emotion)
        print_divider()


def main():
    config = load_config()
    try:
        chat_loop(config)
    except KeyboardInterrupt:
        print("\n\nTake care. 💙")
        sys.exit(0)


if __name__ == "__main__":
    main()
