"""FastAPI application for the SHL Assessment Recommender."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import SHLAgent
from app.catalog import load_catalog
from app.config import HOST, PORT
from app.models import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Starting SHL Assessment Recommender...")
    start = time.time()

    assessments = load_catalog()
    logger.info("Loaded %d assessments from catalog", len(assessments))

    agent = SHLAgent(assessments)
    logger.info("Agent ready in %.1fs", time.time() - start)

    yield

    logger.info("Shutting down SHL Assessment Recommender")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "SHL Assessment Recommender",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "chat": "/chat (POST)"},
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    return await agent.process_chat(messages)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
