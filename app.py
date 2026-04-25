"""
app.py — FastAPI entry point for Aura Emotional AI Backend.
 
Exposes:
  POST /chat       → main conversation endpoint
  GET  /health     → liveness probe
  GET  /session/{session_id}/history  → retrieve memory for a session
 
Startup:  uvicorn app:app --reload --port 8000
"""

# ─── Serve frontend ─────────────────────────────────────────────
 
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional
 
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
 
from pipeline import run_pipeline
from utils.logger import setup_logger
from utils.config import load_config
from memory_module import MemoryStore
 
# ─── Logging setup ────────────────────────────────────────────────────────────
logger = setup_logger("aura.app")
 
# ─── Application config ───────────────────────────────────────────────────────
config = load_config()
 
# ─── Global memory store (in-process; swap for Redis in production) ───────────
memory_store = MemoryStore(max_turns=config.get("max_memory_turns", 8))
 
 
# ─── Lifespan handler (startup / shutdown) ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Aura backend starting up…")
    yield
    logger.info("Aura backend shutting down…")
 
 
# ─── App factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Aura — Emotional AI Backend",
    description="A compassionate, production-ready AI assistant API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ─── Request / Response models ────────────────────────────────────────────────
 
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=0,
        max_length=4000,
        description="User's raw message (any string accepted)",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity",
    )
 
    @field_validator("message", mode="before")
    @classmethod
    def coerce_message(cls, v):
        """Accept None, integers, or anything — coerce to string safely."""
        if v is None:
            return ""
        return str(v).strip()
 
 
class ChatResponse(BaseModel):
    response: str
    emotion: str
    session_id: str
    safety_flag: bool = False
 
 
class HealthResponse(BaseModel):
    status: str
    version: str
 
 
class HistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]
 
 
# ─── Global exception handler — nothing ever leaks a 500 stack trace ─────────
 
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log error properly
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)

    # Print for quick debugging
    print("🔥 GLOBAL ERROR:", exc)

    return JSONResponse(
        status_code=200,
        content={
            "response": f"DEBUG ERROR: {str(exc)}",
            "emotion": "error",
            "session_id": "debug",
            "safety_flag": False,
        },
    )
 
# ─── Endpoints ────────────────────────────────────────────────────────────────
 
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "version": "1.0.0"}
 
 
@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Main conversation endpoint.
 
    Accepts any string message and returns an empathetic AI response.
    Never returns a 5xx error — all failures degrade to a safe fallback.
    """
    # Assign or reuse session ID
    session_id = request.session_id or str(uuid.uuid4())
 
    logger.info(
        "chat request | session=%s | msg_len=%d",
        session_id,
        len(request.message),
    )
 
    try:
        result = run_pipeline(
            raw_input=request.message,
            session_id=session_id,
            memory_store=memory_store,
            config=config,
        )
        return ChatResponse(
            response=result["response"],
            emotion=result["emotion"],
            session_id=session_id,
            safety_flag=result.get("safety_flag", False),
        )
 
    except Exception as exc:
        # Belt-and-suspenders: pipeline already handles its own errors,
        # but if something slips through we still return gracefully.
        logger.error("Pipeline top-level failure: %s", exc, exc_info=True)
        return ChatResponse(
            response=(
                "I'm here with you. Something didn't process correctly, "
                "but we can continue. How are you feeling right now?"
            ),
            emotion="neutral",
            session_id=session_id,
            safety_flag=False,
        )
 
 
@app.get("/session/{session_id}/history", response_model=HistoryResponse, tags=["memory"])
async def get_history(session_id: str):
    """
    Retrieve conversation history for a session.
    Returns empty list if session not found.
    """
    try:
        turns = memory_store.get_history(session_id)
        return HistoryResponse(session_id=session_id, turns=turns)
    except Exception as exc:
        logger.error("History retrieval error: %s", exc)
        return HistoryResponse(session_id=session_id, turns=[])
 
@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

@app.delete("/session/{session_id}", tags=["memory"])
async def clear_session(session_id: str):
    """Clear all memory for a session."""
    try:
        memory_store.clear(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as exc:
        logger.error("Session clear error: %s", exc)
        return {"status": "error", "session_id": session_id}
 