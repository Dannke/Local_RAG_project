# Deployment Guide

## Overview

This project is designed to run on a VPS with **no public ports exposed**. Access is provided via **Tailscale** or **Cloudflare Tunnel** for secure remote access.

## Prerequisites

- Docker Engine and Docker Compose v2+
- A VPS with at least 2 GB RAM (4 GB+ recommended for embedding model)
- Domain name (for Cloudflare Tunnel) or Tailscale account

## Quick Start

### 1. Clone and Configure

```bash
git clone <your-repo>
cd Local_RAG_project
cp .env.example .env
# Edit .env with your API keys and settings
```

Required environment variables in `.env`:
- `OPENROUTER_API_KEY` — your OpenRouter API key
- `HF_TOKEN` — Hugging Face token (for model downloads)
- `RAG_STRUCTURED_LOGGING=true` — enable JSON logs for production

### 2. Build and Start

```bash
docker compose up -d --build
```

This creates:
- **rag-app** container (non-root user `raguser`)
- Named volumes: `rag_data` (indexes, chats, logs), `hf_cache` (model cache)
- Health check endpoint (used by Docker)

### 3. Access the UI

**Option A: Tailscale (recommended, simplest)**

```bash
# On VPS
tailscale up
tailscale serve --bg --http 8501 /rag

# On your machine
tailscale up
# Access via: http://<vps-tailscale-ip>:8501 or the Tailscale MagicDNS name
```

**Option B: Cloudflare Tunnel**

```bash
# On VPS
cloudflared tunnel --no-autoupdate run --token <YOUR_TUNNEL_TOKEN> \
  --url http://localhost:8501
```

Or use a `config.yml` for Cloudflare:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /etc/cloudflared/<TUNNEL_ID>.json
ingress:
  - hostname: rag.yourdomain.com
    service: http://localhost:8501
  - service: http_status:404
```

### 4. Verify Health

```bash
docker compose exec rag-app python scripts/health_check.py --no-llm
```

Expected output:
```
Health check results:
  [PASS] Disk space: Disk free: X.XX GB (min 1.0 GB)
  [PASS] Index: Index OK: N vectors at /data/vector_store
  [PASS] Retrieval: Retrieval OK: N results
  [PASS] LLM: LLM responded: ...
```

## Directory Structure (on VPS)

```
/data/
├── vector_store/       # FAISS index (shared default)
├── raw/                # Source documents
├── chats/              # Per-chat isolated indexes & messages
├── logs/               # Rotating JSON logs (app.log, app.log.1, ...)
└── .cache/huggingface/ # Model cache (HF_TOKEN required)
```

## Configuration

Key settings (via `.env` or `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `OPENROUTER_MODEL` | `openrouter/auto` | LLM model (auto-selects free) |
| `RAG_CHUNK_SIZE` | `800` | Chunk size in **tokens** |
| `RAG_CHUNK_OVERLAP` | `120` | Chunk overlap in **tokens** |
| `TOP_K` | `5` | Default retrieval top-k |
| `USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker model |
| `RATE_LIMIT_MAX_REQUESTS` | `0` | LLM rate limit (0 = disabled) |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window |
| `RAG_LOG_DIR` | `/data/logs` | Log directory |
| `RAG_LOG_MAX_BYTES` | `10000000` | Log rotation size (10 MB) |
| `RAG_LOG_BACKUP_COUNT` | `5` | Number of rotated logs |

## Updating

```bash
git pull
docker compose up -d --build
```

Volumes persist across updates. The index is incremental (only changed files re-indexed).

## Logs

Structured JSON logs in `/data/logs/app.log` (rotated at 10 MB, 5 backups). Each line includes:
- `ts`: epoch milliseconds
- `level`: INFO/WARNING/ERROR
- `logger`: module name
- `message`: human-readable
- `request_id`: correlation ID per query
- Custom fields: `latency_ms`, `prompt_tokens`, `completion_tokens`, `retrieval_ms`, `top_scores`, etc.

Example log line:
```json
{"ts":1699999999999,"level":"INFO","logger":"rag_project.llm.llm_client","message":"llm_completed","request_id":"abc123","model":"openrouter/auto","latency_ms":1234,"success":true,"prompt_tokens":456,"completion_tokens":78,"total_tokens":534}
```

## Backup

```bash
# Backup data volume
docker run --rm -v rag_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/rag_backup_$(date +%F).tar.gz /data
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `HF_TOKEN` invalid | Regenerate at huggingface.co/settings/tokens |
| `OPENROUTER_API_KEY` not working | Verify key at openrouter.ai/keys |
| Out of memory | Increase VPS RAM or reduce `RAG_CHUNK_SIZE` |
| Index not found | Run `docker compose exec rag-app python scripts/ingest.py` |
| Logs not appearing | Check `RAG_STRUCTURED_LOGGING=true` and volume mount |

## Security Notes

- Container runs as non-root user `raguser` (UID 1000)
- No ports exposed in `docker-compose.yml` — only accessible via Tailscale/Tunnel
- API keys only in `.env` (not in image)
- Rate limiting available per session (`RATE_LIMIT_MAX_REQUESTS`)

## Evaluation Baseline

See `eval/baseline_results.json` for retrieval and LLM judge metrics.