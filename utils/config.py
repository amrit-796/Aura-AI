"""
utils/config.py — Configuration loader.

Priority order (highest to lowest):
  1. Environment variables
  2. config.json in the project root
  3. Built-in defaults

This makes the system easy to configure without touching code.
"""

import json
import os


_DEFAULTS = {
    "provider":         "openai",   # "openai" | "anthropic" | "local"
    "model":            "gpt-4o-mini",
    "temperature":      0.8,
    "max_tokens":       500,
    "max_memory_turns": 10,
    "persist_memory":   False,
    "memory_path":      "data/conversation_history.json",
    # base_url: leave None for default OpenAI endpoint;
    #           set to e.g. "http://localhost:11434/v1" for Ollama
    "base_url":         None,
    "api_key":          None,       # Falls back to env vars if None
}


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from file + environment variables.

    Environment variables override file values, which override defaults.
    Recognised env vars:
      AURA_PROVIDER, AURA_MODEL, AURA_API_KEY, AURA_BASE_URL,
      OPENAI_API_KEY, ANTHROPIC_API_KEY
    """
    config = dict(_DEFAULTS)

    # ── Load from file ─────────────────────────────────────────────────────
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                file_config = json.load(f)
                config.update(file_config)
            except json.JSONDecodeError:
                print(f"[warning] Could not parse {config_path}; using defaults.")

    # ── Environment variable overrides ─────────────────────────────────────
    env_map = {
        "AURA_PROVIDER":  "provider",
        "AURA_MODEL":     "model",
        "AURA_API_KEY":   "api_key",
        "AURA_BASE_URL":  "base_url",
        # Convenience aliases
        "OPENAI_API_KEY":    "api_key",
        "ANTHROPIC_API_KEY": "api_key",
    }
    for env_var, config_key in env_map.items():
        value = os.getenv(env_var)
        if value:
            config[config_key] = value

    # Detect provider from key env vars if not explicitly set
    if not os.getenv("AURA_PROVIDER"):
        if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            config["provider"] = "anthropic"
            if config["model"] == _DEFAULTS["model"]:
                config["model"] = "claude-3-haiku-20240307"

    return config
