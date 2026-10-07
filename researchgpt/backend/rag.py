import os
import re
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_chroma import Chroma

from backend.database import get_history


CHROMA_PATH = Path("data/chroma")


class ResearchState(TypedDict, total=False):
    question: str
    provider: Literal["gemini", "qwen"]
    rewritten_question: str
    documents: list[Document]
    answer: str
    verified: bool
    sources: list[str]


class ResearchAssistant:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.gemini_llm = ChatGoogleGenerativeAI(model=model)
        self.qwen_llm: ChatOpenAI | None = None
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        self.store = Chroma(
            collection_name="research_documents_gemini_embedding_001",
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_PATH),
        )
        self.graph = self._build_graph()

    def _llm_for(self, provider: str):
        if provider == "gemini":
            return self.gemini_llm
        if provider == "qwen":
            if self.qwen_llm is None:
                # Read a newly added Groq key without requiring an API restart.
                load_dotenv(override=True)
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("Qwen is not configured. Add GROQ_API_KEY to .env, then try again.")
                self.qwen_llm = ChatOpenAI(
                    model=os.getenv("GROQ_QWEN_MODEL", "qwen/qwen3.8-27b"),
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.1,
                    reasoning_effort="none",
                )
            return self.qwen_llm

    @staticmethod
    def _response_text(response) -> str:
        """Return text from both legacy string and Gemini 3 structured responses."""
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)

    @staticmethod
    def _chunks(text: str, source: str, document_id: str) -> list[Document]:
        # Overlap prevents a fact split across two chunks from being lost.
        size, overlap = 900, 150
        return [
            Document(page_content=text[start : start + size], metadata={"source": source, "document_id": document_id})
            for start in range(0, len(text), size - overlap)
            if text[start : start + size].strip()
        ]

    def add_text(self, text: str, source: str) -> int:
        document_id = str(uuid4())
        chunks = self._chunks(text, source, document_id)
        if not chunks:
            raise ValueError("No readable text was found in this source.")
        self.store.add_documents(chunks)
        return len(chunks)

    def _build_graph(self):
        graph = StateGraph(ResearchState)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("answer", self._answer)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "answer")
        graph.add_edge("answer", "verify")
        graph.add_edge("verify", END)
        return graph.compile()

    def _rewrite(self, state: ResearchState) -> dict:
        # A short rewrite makes follow-up questions more searchable without hiding the step.
        prompt = "Rewrite this as a concise standalone search query. Return only the query.\n" + state["question"]
        query = self._response_text(self._llm_for(state.get("provider", "gemini")).invoke(prompt)).strip()
        return {"rewritten_question": query}

    def _retrieve(self, state: ResearchState) -> dict:
        return {"documents": self.store.similarity_search(state["rewritten_question"], k=8)}

    def _rerank(self, state: ResearchState) -> dict:
        # Transparent lexical reranking: easy to explain, no extra model/API needed.
        words = set(re.findall(r"\w+", state["rewritten_question"].lower()))
        ranked = sorted(
            state["documents"],
            key=lambda doc: len(words & set(re.findall(r"\w+", doc.page_content.lower()))),
            reverse=True,
        )[:4]
        return {"documents": ranked}

    def _answer(self, state: ResearchState) -> dict:
        context = "\n\n".join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in state["documents"])
        prompt = (
            "Answer only from the supplied context. If the answer is absent, say so clearly. "
            "Be concise and cite sources using [Source: filename].\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}"
        )
        return {"answer": self._response_text(self._llm_for(state.get("provider", "gemini")).invoke(prompt))}

    def _verify(self, state: ResearchState) -> dict:
        sources = list(dict.fromkeys(d.metadata["source"] for d in state["documents"]))
        verification = self._response_text(self._llm_for(state.get("provider", "gemini")).invoke(
            "Does this answer stay grounded in the context? Reply only YES or NO.\n"
            f"Context: {' '.join(d.page_content for d in state['documents'])[:6000]}\n"
            f"Answer: {state['answer']}"
        )).strip().upper()
        return {"verified": verification.startswith("YES"), "sources": sources}

    def ask(self, question: str, session_id: str, provider: Literal["gemini", "qwen"] = "gemini") -> dict:
        if self.store._collection.count() == 0:
            raise ValueError("No sources are indexed yet. Upload a file and select Index file before asking a question.")
        history = get_history(session_id)
        # History is intentionally kept in SQLite; the graph uses the latest question for retrieval.
        result = self.graph.invoke({"question": question, "provider": provider})
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "verified": result["verified"],
            "provider": provider,
            "history_count": len(history),
        }
