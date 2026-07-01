"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- LLM (Google Gemini) ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Retrieval ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "30"))
TOP_K_RECOMMEND: int = int(os.getenv("TOP_K_RECOMMEND", "10"))
FAISS_INDEX_PATH: str = os.getenv(
    "FAISS_INDEX_PATH",
    str(Path(__file__).resolve().parent.parent / "catalog" / "processed" / "faiss.index"),
)

# --- Catalog ---
CATALOG_PATH: str = os.getenv(
    "CATALOG_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "raw" / "shl_product_catalog.json"),
)

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# --- Conversation ---
MAX_TURNS: int = int(os.getenv("MAX_TURNS", "8"))
