# Research Assistant

A local, source-grounded research assistant built with FastAPI, Streamlit, LangGraph, Gemini, ChromaDB, and SQLite. Upload research material, index web or video content, and ask questions against the indexed sources.

> This project is intentionally designed to demonstrate a clear, explainable Retrieval-Augmented Generation (RAG) architecture rather than act as a production-ready SaaS application.

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
| Groq + Qwen 3.6 27B (optional) | Alternative LLM provider for query rewriting, answers, and verification |
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

### Installation

```powershell
git clone <your-repository-url>
cd researchgpt

py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set your API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2

# Optional: required only to select Qwen through Groq in the interface.
GROQ_API_KEY=your_groq_api_key_here
GROQ_QWEN_MODEL=qwen/qwen3.6-27b
```

### Run the application

Start the API in one terminal:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8003 --reload
```

Start the frontend in another terminal:

```powershell
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

Open the application at `http://127.0.0.1:8503`. API documentation is available at `http://127.0.0.1:8003/docs`.

### Switching answer models

Use the **Answer model** selector in the Streamlit sidebar to choose Gemini 3.6 Flash or Qwen 3.6 27B via Groq for each question. Both options use the same Gemini-powered ChromaDB index, so switching models does not require re-uploading documents. To use Qwen, add `GROQ_API_KEY` to `.env`; the backend reads the key when Qwen is first selected.

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

## Interview Talking Points

**What problem does it solve?**  
It gives users a simple way to ask questions across their research sources without manually searching each file or webpage.

**Why use RAG?**  
RAG grounds Gemini's response in retrieved user content. This improves relevance and makes it possible to show which sources informed the answer.

**Why ChromaDB?**  
ChromaDB supports local semantic search without requiring a separate managed database. It is a practical fit for a prototype and simple local deployment.

**Why LangGraph?**  
LangGraph makes the pipeline's stages explicit and independently testable. It is easier to explain and extend than a single, large prompt-based function.

**Why SQLite?**  
SQLite is serverless and persists chat history in a single local database file. A production multi-user version would typically use PostgreSQL.

**Production improvements**  
Add authentication, document-level authorization, page-level citations, asynchronous indexing, a cross-encoder reranker, evaluation datasets, observability, rate limiting, and a managed database/vector store.

## Limitations

- The application requires a valid Gemini API key and network access.
- Website extraction works best for publicly accessible, server-rendered pages.
- YouTube transcript indexing only works when a transcript is available.
- Local ChromaDB and SQLite are suitable for a demo or single-user workflow, not high-scale production workloads.

## License

This project is provided for learning and portfolio use.
