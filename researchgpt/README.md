# Research Assistant

A local, source-grounded research assistant built with FastAPI, Streamlit, LangGraph, Gemini, ChromaDB, and SQLite. Upload research material, index web or video content, and ask questions against the indexed sources.

## Features

- Ingest PDF, DOCX, TXT, CSV, and PPTX files
- Extract and index public webpage text
- Index available YouTube video transcripts
- Retrieve relevant content through semantic vector search
- Run an explicit RAG workflow: query rewrite → retrieval → reranking → answer generation → verification
- Return source-labelled answers
- Persist chat history locally with SQLite
- Explore backend endpoints through FastAPI's built-in OpenAPI documentation

## Architecture

```text
                        ┌───────────────────────────┐
                        │      Source ingestion      │
                        │ Files | Web | YouTube      │
                        └─────────────┬─────────────┘
                                      │
                        Text extraction and chunking
                                      │
                Gemini embeddings + source metadata
                                      │
                        ┌─────────────▼─────────────┐
                        │         ChromaDB           │
                        │     Semantic vector store  │
                        └─────────────┬─────────────┘
                                      │
 User question → LangGraph: rewrite → retrieve → rerank → answer → verify
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
          FastAPI REST API                         SQLite history
                  │
          Streamlit chat interface
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application language and AI ecosystem |
| FastAPI | REST API for ingestion, question answering, and history |
| Streamlit | Lightweight user interface for document upload and chat |
| LangChain | Integrations for documents, Gemini, embeddings, and ChromaDB |
| LangGraph | Structured orchestration of the multi-step RAG workflow |
| Google Gemini | Query rewriting, embeddings, answer generation, and verification |
| Groq + Qwen 3.8 27B (optional) | Alternative LLM provider for query rewriting, answers, and verification |
| ChromaDB | Local vector database for semantic document retrieval |
| SQLite | Lightweight, persistent per-session chat history |

## RAG Workflow

1. **Ingest** — Extract plain text from the uploaded file, webpage, or transcript.
2. **Chunk** — Split text into 900-character chunks with 150-character overlap to preserve context at boundaries.
3. **Embed and index** — Generate Gemini embeddings and store the chunks and source metadata in ChromaDB.
4. **Rewrite** — Convert the user's question into a concise, standalone retrieval query.
5. **Retrieve** — Use semantic similarity to find the eight most relevant chunks.
6. **Rerank** — Apply a transparent keyword-overlap score and keep the strongest four chunks.
7. **Generate** — Ask Gemini to answer only from the selected context and cite the source labels.
8. **Verify** — Run a final grounding check to determine whether the answer is supported by the retrieved text.
9. **Persist** — Save the conversation to SQLite using its session ID.

## Getting Started

### Prerequisites

- Python 3.14
- A Google Gemini API key

### Run locally

From the project directory, create the virtual environment and install the dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add `GOOGLE_API_KEY`. Then start the API and UI in separate terminals, keeping both terminals in the project directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8003
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the second terminal. The UI defaults to `http://127.0.0.1:8003` for the API; set `API_URL` before starting Streamlit if the API uses another address.

### Switching answer models

Use the **Answer model** selector in the Streamlit sidebar to choose Gemini 3.8 Flash or Qwen 3.8 27B via Groq for each question. Both options use the same Gemini-powered ChromaDB index, so switching models does not require re-uploading documents. To use Qwen, add `GROQ_API_KEY` to `.env`; the backend reads the key when Qwen is first selected.

The default embedding model is `gemini-embedding-001`. If you change the embedding model, remove `data/chroma/` before indexing documents again so Chroma does not mix vectors with different dimensions.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Confirms that the API is running and Gemini is configured |
| `POST` | `/ingest/file` | Upload and index a supported document |
| `POST` | `/ingest/website` | Extract and index a webpage |
| `POST` | `/ingest/youtube` | Fetch and index a YouTube transcript |
| `POST` | `/ask` | Ask a question against indexed sources |
| `GET` | `/history/{session_id}` | Retrieve saved messages for a session |

### Ask a question

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "What are the main findings?",
  "session_id": "demo-user"
}
```

### Index a website

```http
POST /ingest/website
Content-Type: application/json
```

```json
{
  "url": "https://example.com/article"
}
```

For YouTube ingestion, send the value after `v=` in a typical YouTube URL as `video_id`.

## Project Structure

```text
researchgpt/
├── backend/
│   ├── database.py       # SQLite chat-history functions
│   ├── loaders.py        # Extractors for supported sources
│   ├── main.py           # FastAPI routes and application lifecycle
│   └── rag.py            # ChromaDB setup and LangGraph RAG pipeline
├── streamlit_app.py      # Streamlit upload-and-chat interface
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── README.md
```
