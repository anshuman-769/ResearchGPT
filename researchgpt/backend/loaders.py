"""Small, dependency-light text extractors for common research inputs."""
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from pptx import Presentation
from youtube_transcript_api import YouTubeTranscriptApi


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".pptx"}


def read_upload(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    stream = BytesIO(content)
    if extension == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(stream).pages)
    if extension == ".docx":
        return "\n".join(p.text for p in Document(stream).paragraphs)
    if extension == ".txt":
        return content.decode("utf-8", errors="ignore")
    if extension == ".csv":
        return pd.read_csv(stream).to_csv(index=False)
    if extension == ".pptx":
        presentation = Presentation(stream)
        return "\n".join(
            shape.text
            for slide in presentation.slides
            for shape in slide.shapes
            if hasattr(shape, "text")
        )
    raise ValueError(f"Unsupported file type: {extension}")


def read_website(url: str) -> str:
    response = requests.get(url, timeout=15, headers={"User-Agent": "ResearchAssistant/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
    return soup.get_text(" ", strip=True)


def read_youtube(video_id: str) -> str:
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return " ".join(snippet.text for snippet in transcript)
