import os
import uuid

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8003")
st.set_page_config(page_title="Simple Research Assistant", page_icon="🔎")
st.title("🔎 Simple Research Assistant")
st.caption("Upload research material, then ask grounded questions about it.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Add knowledge")
    provider = st.selectbox(
        "Answer model",
        options=["gemini", "qwen"],
        format_func=lambda option: "Gemini 3.6 Flash" if option == "gemini" else "Qwen 3.6 27B via Groq",
        help="Choose the model used for query rewriting, answer generation, and verification.",
    )
    upload = st.file_uploader("PDF, DOCX, TXT, CSV, or PPTX", type=["pdf", "docx", "txt", "csv", "pptx"])
    if upload and st.button("Index file"):
        response = requests.post(f"{API}/ingest/file", files={"file": (upload.name, upload.getvalue())})
        st.success(response.json().get("message", response.text)) if response.ok else st.error(response.text)
    url = st.text_input("Website URL")
    if url and st.button("Index website"):
        response = requests.post(f"{API}/ingest/website", json={"url": url})
        st.success(response.json().get("message", response.text)) if response.ok else st.error(response.text)
    video_id = st.text_input("YouTube video ID")
    if video_id and st.button("Index transcript"):
        response = requests.post(f"{API}/ingest/youtube", json={"video_id": video_id})
        st.success(response.json().get("message", response.text)) if response.ok else st.error(response.text)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask a question about your sources"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your sources..."):
            response = requests.post(
                f"{API}/ask",
                json={"question": question, "session_id": st.session_state.session_id, "provider": provider},
            )
            if response.ok:
                data = response.json()
                text = data["answer"] + "\n\nSources: " + ", ".join(data["sources"])
                if not data["verified"]:
                    text += "\n\n⚠️ Verification was inconclusive."
                st.write(text)
                st.session_state.messages.append({"role": "assistant", "content": text})
            else:
                st.error(response.text)
