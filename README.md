# SHL Assessment Recommender

Conversational AI agent that helps hiring managers discover relevant **SHL Individual Test Solutions** through natural dialogue. Built for the SHL AI Intern take-home assignment.

## Features

- **FastAPI** backend with `GET /health` and `POST /chat`
- **Stateless** multi-turn conversations (full history in each request)
- **Gemini 2.5 Flash** for clarification, recommendations, refinement, comparison, and refusal
- **FAISS + Sentence Transformers** hybrid retrieval
- **Playwright** catalog scraper
- **Docker** and **Render** deployment ready
- Comprehensive **unit tests**

## Architecture

```
POST /chat → Hybrid Retrieval (FAISS + keywords) → Gemini 2.5 Flash → Catalog Validation → JSON Response
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Google AI Studio API key](https://aistudio.google.com/apikey) for Gemini

### Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### Build FAISS Index (optional — built automatically on first run)

```bash
python -m app.build_index
```

### Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Scrape Catalog (Playwright)

```bash
python -m app.scraper --skip-details
```

## API

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"I am hiring a Java backend developer with 4 years experience\"}]}"
```

Sample response:
```json
{
  "reply": "Here are SHL assessments suited for a mid-level Java backend developer...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

### Multi-turn Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [
    {\"role\": \"user\", \"content\": \"I need assessments\"},
    {\"role\": \"assistant\", \"content\": \"What role are you hiring for?\"},
    {\"role\": \"user\", \"content\": \"Senior data analyst with SQL and stakeholder skills\"}
  ]}"
```

## Tests

```bash
pytest tests/test_core.py -v
```

Covers: catalog loading, schema validation, retrieval, clarification, recommendation, refinement, comparison, off-topic refusal, prompt injection, and hallucination filtering.

## Docker

```bash
docker build -t shl-assessment-agent .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-assessment-agent
```

## Render Deployment

1. Push this repo to GitHub.
2. Create a **Web Service** on [Render](https://render.com) and connect the repo.
3. Render detects `render.yaml` automatically, or configure manually:
   - **Runtime**: Docker
   - **Health check path**: `/health`
4. Add environment variable:
   - `GEMINI_API_KEY` — your Google AI API key
5. Deploy. The service exposes `/health` and `/chat`.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Gemini model name |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed ONNX model |
| `TOP_K_RETRIEVAL` | No | `30` | Candidates retrieved per query |
| `TOP_K_RECOMMEND` | No | `10` | Max recommendations returned |
| `MAX_TURNS` | No | `8` | Conversation turn budget |
| `PORT` | No | `8000` | Server port |

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI app
│   ├── agent.py          # Gemini orchestration + validation
│   ├── retrieval.py      # FAISS hybrid search
│   ├── catalog.py        # Catalog loader
│   ├── scraper.py        # Playwright scraper
│   ├── build_index.py    # FAISS index builder
│   ├── prompts.py        # System prompts
│   ├── models.py         # Pydantic schemas
│   └── config.py         # Configuration
├── data/raw/             # SHL catalog JSON
├── catalog/processed/    # FAISS index cache
├── tests/                # Unit tests
├── docs/                 # Approach document
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## Documentation

See [docs/APPROACH.md](docs/APPROACH.md) for architecture, design decisions, and evaluation notes.

## License

MIT
