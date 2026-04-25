# Aura — Emotional AI Companion

A compassionate, modular Python AI assistant that supports users through
anxiety, stress, sadness, confusion, loneliness, and life challenges.

---

## Folder Structure

```
emotional_ai/
│
├── main.py                   # Entry point — start here
├── emotion_detection.py      # Detects emotion from user input
├── response_generator.py     # Builds prompts and calls the LLM
├── philosophy_engine.py      # Retrieves contextual wisdom snippets
├── safety_layer.py           # Crisis detection and override responses
├── memory_module.py          # Rolling conversation memory
│
├── utils/
│   ├── __init__.py
│   ├── config.py             # Configuration loader
│   └── display.py            # Terminal UI (colours, wrapping, prompts)
│
├── data/                     # Created automatically if persist_memory=true
│   └── conversation_history.json
│
├── config.example.json       # Copy to config.json and fill in values
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Python dependencies

```bash
cd emotional_ai
pip install -r requirements.txt
```

### 2. Set your API key

**Option A — environment variable (recommended)**
```bash
export OPENAI_API_KEY="sk-..."
```
Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY = "sk-..."
```

**Option B — config file**
```bash
cp config.example.json config.json
# Then open config.json and set "api_key": "sk-..."
```

### 3. Run Aura

```bash
python main.py
```

---

## Configuration Reference

| Key               | Default          | Description                                              |
|-------------------|------------------|----------------------------------------------------------|
| `provider`        | `"openai"`       | `"openai"` \| `"anthropic"` \| `"local"`               |
| `model`           | `"gpt-4o-mini"`  | Model name for the chosen provider                       |
| `temperature`     | `0.8`            | Response creativity (0.0–1.0)                            |
| `max_tokens`      | `500`            | Maximum response length                                  |
| `max_memory_turns`| `10`             | How many past turns to keep in context                   |
| `persist_memory`  | `false`          | Save conversation to disk between sessions               |
| `memory_path`     | `data/...json`   | File path for persisted memory                           |
| `api_key`         | `null`           | API key (defaults to env var)                            |
| `base_url`        | `null`           | Custom endpoint for local models (e.g. Ollama)           |

---

## Using a Local Model (Ollama / LM Studio)

1. Install [Ollama](https://ollama.ai) and pull a model:
   ```bash
   ollama pull llama3
   ```

2. Update `config.json`:
   ```json
   {
     "provider":  "openai",
     "model":     "llama3",
     "base_url":  "http://localhost:11434/v1",
     "api_key":   "ollama"
   }
   ```

3. Run normally — no internet required.

---

## Using Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or in `config.json`:
```json
{
  "provider": "anthropic",
  "model":    "claude-3-haiku-20240307",
  "api_key":  "sk-ant-..."
}
```

---

## Enabling Advanced Emotion Detection

The default emotion detection is rule-based (fast, zero dependencies).
For more accurate results, install the transformer model:

```bash
pip install transformers torch
```

Then in your code, pass `use_transformer=True` to `detect_emotion()`.
The model (`j-hartmann/emotion-english-distilroberta-base`) downloads
automatically on first use (~250 MB).

---

## How It Works (Step-by-Step)

```
User types a message
        │
        ▼
┌───────────────────┐
│   safety_layer    │  ← Checks for crisis/self-harm keywords FIRST
└───────────────────┘
        │
   CRISIS? ──yes──► Print hotline info, skip LLM pipeline
        │
        no
        ▼
┌────────────────────┐
│  emotion_detection │  ← Detects dominant emotion (rule-based or NLP)
└────────────────────┘
        │
        ▼
┌─────────────────────┐
│  philosophy_engine  │  ← Retrieves a contextually scored wisdom snippet
└─────────────────────┘
        │
        ▼
┌──────────────────────┐
│  memory_module       │  ← Provides conversation history for context
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  response_generator  │  ← Builds system prompt + calls LLM API
└──────────────────────┘
        │
        ▼
   Print response
        │
        ▼
   memory.add_turn()   ← Stores the exchange for future context
```

---

## Module Summary

| Module                  | Responsibility                                                  |
|-------------------------|-----------------------------------------------------------------|
| `main.py`               | Orchestrates the loop; ties all modules together                |
| `emotion_detection.py`  | Keyword regex (Tier 1) or transformer model (Tier 2)            |
| `safety_layer.py`       | Crisis keyword detection; hotline responses; pipeline override  |
| `philosophy_engine.py`  | Scored wisdom retrieval by emotion + keyword overlap            |
| `memory_module.py`      | Rolling turn buffer; optional JSON persistence                  |
| `response_generator.py` | System prompt builder; multi-provider LLM client; fallback      |
| `utils/config.py`       | Config file + env var loader with sensible defaults             |
| `utils/display.py`      | ANSI-coloured terminal output; text wrapping                    |

---

## Future Improvements

1. **Voice interface** — Add `speech_recognition` + `pyttsx3` for spoken conversations.
2. **Web UI** — Wrap with FastAPI + a simple React frontend.
3. **Mood tracking** — Plot emotion trends over multiple sessions using matplotlib.
4. **Personalisation memory** — Store user preferences (name, recurring themes) separately from conversation turns.
5. **Multilingual support** — Add language detection and respond in the user's language.
6. **RAG (Retrieval-Augmented Generation)** — Store a larger wisdom corpus in a vector database (e.g. ChromaDB) for richer, more personalised insights.
7. **Richer safety escalation** — Integrate with real crisis APIs (e.g. Crisis Text Line) where permitted.
8. **Session summary** — At the end of each session, generate a brief reflection summary for the user.

---

## Important Notes

- Aura is a **supportive companion**, not a licensed therapist or medical professional.
- It will never attempt to fully handle a mental health crisis — it always directs users to real humans.
- No conversation data leaves your machine unless you explicitly use a cloud LLM API.
