import os
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from backend.database import get_history, save_message
from backend.loaders import read_upload, read_website, read_youtube
from backend.rag import ResearchAssistant

load_dotenv()
assistant_service: ResearchAssistant | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global assistant_service
    if os.getenv("GOOGLE_API_KEY"):
        assistant_service = ResearchAssistant()
    yield


app = FastAPI(title="Simple Research Assistant", lifespan=lifespan)


def service() -> ResearchAssistant:
    if not assistant_service:
        raise HTTPException(503, "Set GOOGLE_API_KEY in .env, then restart the API.")
    return assistant_service


class Question(BaseModel):
    question: str
    session_id: str = "default"
    provider: Literal["gemini", "qwen"] = "gemini"


class Website(BaseModel):
    url: HttpUrl


class Youtube(BaseModel):
    video_id: str


@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": assistant_service is not None}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    try:
        text = read_upload(file.filename or "upload", await file.read())
        chunks = service().add_text(text, file.filename or "upload")
        return {"message": "Document indexed", "chunks": chunks, "source": file.filename}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/ingest/website")
def ingest_website(item: Website):
    try:
        chunks = service().add_text(read_website(str(item.url)), str(item.url))
        return {"message": "Website indexed", "chunks": chunks}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/ingest/youtube")
def ingest_youtube(item: Youtube):
    try:
        chunks = service().add_text(read_youtube(item.video_id), f"YouTube: {item.video_id}")
        return {"message": "Transcript indexed", "chunks": chunks}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/ask")
def ask(item: Question):
    try:
        answer = service().ask(item.question, item.session_id, item.provider)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Research pipeline failed: {exc}") from exc
    save_message(item.session_id, "user", item.question)
    save_message(item.session_id, "assistant", answer["answer"])
    return answer


@app.get("/history/{session_id}")
def history(session_id: str):
    return get_history(session_id)
