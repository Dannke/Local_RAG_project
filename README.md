# Local RAG Project

Minimal Python RAG project over DOCX/PDF/TXT/MD documents.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Create `.env` from the example:

```powershell
Copy-Item .env.example .env
```

Set your OpenRouter values in `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_MODEL=openrouter/auto
TOP_K=5
MAX_CONTEXT_CHARS=12000
TEMPERATURE=0.2
```

## Web UI

Start the Streamlit app:

```powershell
streamlit run app.py
```

Typical workflow:

1. Open the web UI.
2. Upload one or more `PDF` or `DOCX` files.
3. Click `Проиндексировать документы`.
   - 📊 **New:** Progress bar shows indexing stages and chunk count in real-time
4. Ask a question, for example: `Какие основные выводы есть в документах?`
5. Read the answer and inspect the retrieved chunks below it.

Uploaded documents are shown as a list in the UI. You can delete a single file
with `Удалить`; the saved FAISS index is reset automatically so search cannot
return chunks from a removed document.

To remove uploaded files from `data/raw`, use `Очистить загруженные документы`.
The UI can also remove the saved FAISS index at the same time.

Indexing is incremental for newly added files. If a source file is changed or
deleted, the app safely rebuilds the index because the current FAISS index type
does not support reliable in-place vector deletion.

Chunking is paragraph-aware and heading-aware. The pipeline prefers paragraph
boundaries and Markdown-style headings, then falls back to fixed-size windows
for very long text blocks.

Answers in the Web UI are streamed: after you submit a question, the app first
shows `печатает...`, then renders the model response progressively.

Retrieved context shows source file, chunk number, score, and page metadata when
available. PDF files are indexed page-by-page. DOCX files do not contain stable
rendered page numbers, so they are marked as page `1` unless richer metadata is
added later. Rebuild the index after this update to write page metadata into
`data/vector_store/documents.json`.

The UI is a thin layer over the existing backend:

- document loading and chunking use the existing ingest pipeline
- embeddings use sentence-transformers
- retrieval uses the existing FAISS vector store
- chat uses the existing retrieval plus OpenRouter LLM pipeline

## CLI

Index documents:

```powershell
.\.venv\Scripts\python.exe -m rag_project.cli ingest --data data/raw --index data/vector_store
```

Search:

```powershell
.\.venv\Scripts\python.exe -m rag_project.cli search "О чем эти документы?" --index data/vector_store
```

Chat:

```powershell
.\.venv\Scripts\python.exe -m rag_project.cli chat --index data/vector_store
```

Helper scripts:

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py
.\.venv\Scripts\python.exe scripts\query.py "О чем эти документы?"
.\.venv\Scripts\python.exe scripts\chat.py
```

## RAG Prompt

System prompt:

```text
Ты AI assistant для RAG-системы.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если информации недостаточно — так и скажи.
Не придумывай факты.
```
