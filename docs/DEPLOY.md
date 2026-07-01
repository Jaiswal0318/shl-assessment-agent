# Deploy to Render

## 1. Push code to GitHub

```bash
git init
git add .
git commit -m "SHL Assessment Agent - production ready"
```

Create a new repository on GitHub named `shl-assessment-agent`, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/shl-assessment-agent.git
git branch -M main
git push -u origin main
```

> If your default branch is `master`, update `branch: master` in `render.yaml` or rename to `main`.

## 2. Create Render Blueprint

1. Go to [https://dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account and select the `shl-assessment-agent` repository.
4. Render detects `render.yaml` automatically.
5. When prompted for **GEMINI_API_KEY**, paste your Google AI Studio key.
6. Click **Apply** / **Deploy Blueprint**.

First deploy takes **10–20 minutes** (Docker build downloads the embedding model and builds the FAISS index).

## 3. Verify deployment

Once live, your service URL will look like:

```
https://shl-assessment-agent.onrender.com
```

Test endpoints:

```bash
curl https://shl-assessment-agent.onrender.com/health

curl -X POST https://shl-assessment-agent.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"I am hiring a Java developer\"}]}"
```

## Environment variables (Render dashboard)

| Variable | Required | Value |
|----------|----------|-------|
| `GEMINI_API_KEY` | Yes | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` |

Render sets `PORT` automatically — do not override it.

## Troubleshooting

- **Build timeout**: Free tier may time out on first build. Retry deploy or upgrade plan.
- **503 on /chat**: Service still starting (cold start ~30s after idle).
- **Gemini errors**: Confirm `GEMINI_API_KEY` is set in Render → Environment.
