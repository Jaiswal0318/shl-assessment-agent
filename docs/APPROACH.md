# SHL Assessment Agent — Approach Document

## 1. Problem & Goals

Build a **stateless conversational API** that helps hiring managers discover relevant **SHL Individual Test Solutions** through natural dialogue. The agent must:

- **Recommend** grounded assessments (1–10 items)
- **Clarify** vague hiring needs
- **Refine** shortlists when constraints change
- **Compare** assessments using catalog metadata only
- **Refuse** off-topic or adversarial requests

Tech stack: **FastAPI**, **Gemini 2.5 Flash**, **FAISS**, **Sentence Transformers**, **Playwright** scraper, **Docker**, **Render**.

---

## 2. Architecture

```
POST /chat
  → Parse conversation history (stateless)
  → Build search query from all user turns
  → Hybrid retrieval (FAISS + keyword boost)
  → Gemini 2.5 Flash with grounded catalog context
  → JSON parse + catalog validation
  → ChatResponse
```

| Layer | Technology | Role |
|-------|-----------|------|
| API | FastAPI | `/health`, `/chat` |
| LLM | Gemini 2.5 Flash | Clarify, recommend, refine, compare, refuse |
| Embeddings | `all-MiniLM-L6-v2` | Semantic encoding |
| Vector store | FAISS `IndexFlatIP` | Cosine similarity search |
| Data | Playwright scraper + cached JSON | SHL catalog (~377+ items) |

---

## 3. Data Pipeline

1. **Scrape** (`app/scraper.py`): Playwright loads `https://www.shl.com/solutions/products/product-catalog/?start=<n>&type=1`, paginating Individual Test Solutions.
2. **Enrich** (optional): Visit detail pages for descriptions and duration.
3. **Normalize** (`app/catalog.py`): Map keys to type codes (K/P/A/S/B/C/D/E), build `search_text` for embeddings.
4. **Index** (`app/build_index.py`): Encode assessments, persist `catalog/processed/faiss.index` and `embeddings.npy`.

---

## 4. Retrieval

- **Semantic**: FAISS inner-product search on normalized embeddings.
- **Keyword boost**: Inverted index over assessment names/descriptions; exact terms (e.g. `java`, `opq`, `sql`) get score boosts.
- **Top-K**: 30 candidates passed to Gemini; final response capped at 10 validated recommendations.

---

## 5. Agent Behaviors (Gemini Prompt)

The system prompt enforces five modes:

| Mode | Trigger | `recommendations` |
|------|---------|-------------------|
| Clarification | Vague query | `[]` |
| Recommendation | Sufficient context | 1–10 catalog items |
| Refinement | "add/remove/replace" | Updated shortlist |
| Comparison | "compare/difference" | `[]` (text-only) |
| Refusal | Off-topic / injection | `[]` |

**Grounding**: LLM only sees retrieved assessments. Post-generation validation drops any URL/name not in the catalog (with fuzzy name fallback).

**Heuristic pre-filter**: Obvious off-topic queries (geography, math, prompt injection) are refused before LLM call.

---

## 6. API Contract

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
Request:
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java backend developer"}
  ]
}
```

Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

---

## 7. Deployment

- **Docker**: Python 3.11-slim, pre-downloads embedding model, runs `uvicorn app.main:app`.
- **Render**: `render.yaml` web service with `GEMINI_API_KEY` secret and `/health` check.
- **Env vars**: `GEMINI_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `PORT`.

---

## 8. Testing

`pytest tests/test_core.py` covers:

- Catalog integrity (300+ items, unique IDs, valid type codes)
- Schema serialization
- Retrieval (Java, personality/OPQ queries)
- Agent behaviors: clarification, recommendation, refinement, comparison, refusal, prompt injection, hallucination filtering

---

## 9. Design Trade-offs

1. **Stateless API** — Full history sent each turn; simpler scaling on Render.
2. **Retrieval-grounded LLM** — Prevents hallucination; trades some flexibility for reliability.
3. **Cached FAISS index** — Faster cold starts in Docker; rebuilt when catalog changes.
4. **Gemini JSON mode** — Structured output reduces parsing failures vs free-form text.
