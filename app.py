from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from ingest import INDEX_DIR, MODEL_NAME, PDF_DIR, ingest_pdfs

load_dotenv()
ROOT = Path(__file__).resolve().parent
INDEX_PATH = INDEX_DIR / "study.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

st.set_page_config(
    page_title="Syllabus Studio | AI Study Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17233b; --muted:#718096; --brand:#5b6cff; --line:#e8edf5; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7f9ff 0%,#f7f8fc 55%,#fff 100%); color:var(--ink); }
.block-container { max-width:1250px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:#17233b; }
.hero { padding:1.7rem 1.9rem; border:1px solid #e8ecf8; border-radius:24px;
        background:linear-gradient(120deg,#fff 5%,#f0f2ff 100%); margin-bottom:1.2rem; }
.eyebrow { color:#5b6cff; font-size:.76rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { font-size:2.35rem; margin:.3rem 0; }
.hero p { color:#66738a; margin:0; font-size:1.02rem; }
.panel { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; }
.small-muted { color:#77839a; font-size:.88rem; }
div[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e9edf5; }
.stButton>button { border-radius:11px; font-weight:650; border:1px solid #dfe4f2; }
.stButton>button[kind="primary"] { background:#5b6cff; border-color:#5b6cff; }
div[data-testid="stChatMessage"] { background:#fff; border:1px solid #edf0f6; border-radius:16px; padding:.9rem; }
div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #cbd3e5; border-radius:14px; padding:.6rem; }
hr { border-color:#e9edf5; }
</style>
""", unsafe_allow_html=True)


# ---------- Cached resources ----------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_faiss():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []
    idx = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("Add GROQ_API_KEY to your .env file or Streamlit secrets.")
    return Groq(api_key=key)


def retrieve(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    index, metadata = load_faiss()
    if index is None or not metadata:
        return []
    embedder = load_embedder()
    vector = embedder.encode([query], convert_to_numpy=True,
                             normalize_embeddings=True).astype("float32")
    scores, ids = index.search(vector, min(top_k, index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(metadata[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def make_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['source']} | Page: {c['page']} | Similarity: {c['score']:.3f}]\\n{c['text']}"
        for c in chunks
    )


def ask_tutor(question: str, mode: str = "Explain simply", top_k: int = 6) -> tuple[str, list[dict]]:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return ("I couldn't find a relevant passage in the indexed PDFs. Upload your book PDFs and click **Build / refresh knowledge base** first.", [])
    context = make_context(chunks)
    system_prompt = """You are Syllabus Studio, a patient, clear, human-sounding study tutor.
Use the supplied PDF excerpts as the primary source. Be accurate and teach for understanding.

Rules:
- Answer the student's actual question directly. Do not pad the answer or repeat the same idea.
- Explain difficult ideas in plain language, then add a concise technical definition where useful.
- Organize with short headings, steps, examples, and simple analogies when they improve understanding.
- Do not invent facts, page references, formulas, or syllabus requirements.
- If the excerpts do not contain the answer, say what is missing and distinguish general background from PDF-supported facts.
- If sources disagree, point out the disagreement instead of silently choosing.
- Cite factual claims with [filename, p. X] using the source labels in the excerpts.
- Keep the response focused and appropriate for a student."""
    user_prompt = f"""Study mode: {mode}

PDF excerpts:
{context}

Student request:
{question}

Teach clearly and concisely. Avoid unnecessary repetition."""
    client = get_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.35,
        max_tokens=1800,
    )
    return completion.choices[0].message.content or "No response returned.", chunks


def all_source_text() -> str:
    _, metadata = load_faiss()
    # Keep syllabus planning grounded while avoiding sending the whole book.
    seen, parts = set(), []
    for item in metadata:
        key = (item["source"], item["page"])
        if key not in seen:
            seen.add(key)
            parts.append(f"{item['source']} p.{item['page']}: {item['text'][:500]}")
        if len(parts) >= 45:
            break
    return "\n".join(parts)


def generate_random_questions(count: int = 5) -> str:
    prompt = f"""Create {count} varied self-test questions based only on these study excerpts.
Mix recall, understanding, and application. Do not include answers immediately.
Number the questions and keep them unambiguous.
Excerpts:
{all_source_text()}"""
    result = get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You create high-quality study questions grounded in provided textbook excerpts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_tokens=1200,
    )
    return result.choices[0].message.content or "Could not create questions."


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📚 Syllabus Studio")
    st.caption("Your personal PDF-powered study space")
    st.markdown("---")
    st.markdown("### 1 · Add your books")
    uploads = st.file_uploader(
        "Upload textbook PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned image PDFs need OCR, which is not included in this starter version.",
    )
    if uploads:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for uploaded in uploads:
            target = PDF_DIR / Path(uploaded.name).name
            target.write_bytes(uploaded.getbuffer())
            saved.append(target.name)
        st.success(f"Saved {len(saved)} PDF(s) to data/pdfs/.")
    st.markdown("### 2 · Prepare your knowledge base")
    if st.button("Build / refresh knowledge base", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting pages, creating chunks and embeddings..."):
                count, location = ingest_pdfs()
            load_faiss.clear()
            st.success(f"Ready: {count} chunks indexed.")
        except Exception as exc:
            st.error(str(exc))
    st.markdown("---")
    top_k = st.slider("Context passages per answer", 3, 10, 6)
    st.caption("More passages can add context, but may also add noise.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Your books · Your learning plan</div>
  <h1>Learn deeply. Study smarter.</h1>
  <p>A focused AI tutor that explains your PDFs, helps plan your syllabus, and prepares you for tests.</p>
</div>
""", unsafe_allow_html=True)

index, metadata = load_faiss()
if index is None:
    st.info("Start by uploading one or more textbook PDFs in the left sidebar, then select **Build / refresh knowledge base**.")
else:
    st.caption(f"Knowledge base ready · {len(metadata)} indexed passages")

tab_tutor, tab_plan, tab_test, tab_notes = st.tabs(
    ["💬 Tutor", "🗓️ Syllabus planner", "🧠 Grand test", "📝 Notes"]
)

with tab_tutor:
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Ask your study tutor")
        mode = st.selectbox("How should I explain it?", [
            "Explain simply", "Step-by-step explanation", "Give an example",
            "Compare concepts", "Exam-focused explanation"
        ])
        question = st.text_area(
            "What are you studying?",
            placeholder="Example: Explain the working of the human respiratory system in simple steps...",
            height=105,
        )
        if st.button("Explain this topic", type="primary", disabled=not question.strip()):
            try:
                with st.spinner("Finding relevant passages and preparing your explanation..."):
                    answer, sources = ask_tutor(question, mode, top_k)
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = sources
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
        if st.session_state.get("last_answer"):
            st.markdown(st.session_state["last_answer"])
            with st.expander("View textbook passages used"):
                for source in st.session_state.get("last_sources", []):
                    st.markdown(f"**{source['source']} · Page {source['page']} · Score {source['score']:.3f}**")
                    st.write(source["text"])
    with right:
        st.markdown('<div class="panel"><b>How this tutor works</b><p class="small-muted">Your question is matched against the indexed textbook passages. The most relevant passages are sent to the language model so its explanation stays connected to your books.</p></div>', unsafe_allow_html=True)
        st.markdown("#### Quick prompts")
        prompts = [
            "Summarize the key ideas in this chapter.",
            "Explain the most difficult concept in this topic.",
            "What should I memorize and what should I understand?",
            "Give me 5 application-based questions.",
        ]
        for p in prompts:
            if st.button(p, use_container_width=True, key=f"quick_{p}"):
                st.session_state["quick_question"] = p
                st.rerun()
        if st.session_state.get("quick_question"):
            st.caption("Prompt ready — copy it into the question box.")

with tab_plan:
    st.subheader("Turn your syllabus into a realistic plan")
    st.write("Ask for a study roadmap based on your uploaded books. Add your exam date and available study time.")
    c1, c2, c3 = st.columns(3)
    with c1:
        exam_date = st.date_input("Target exam date")
    with c2:
        hours_per_day = st.number_input("Study hours per day", min_value=1, max_value=12, value=3)
    with c3:
        days_off = st.multiselect("Days unavailable", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    priorities = st.text_area("Your syllabus / weak areas (optional)", placeholder="Paste your official syllabus or list the chapters you find difficult.")
    if st.button("Create my study strategy", type="primary"):
        try:
            request = f"""Create a practical syllabus coverage plan from the PDF material.
Exam date: {exam_date}. Available study time: {hours_per_day} hours/day.
Days unavailable: {days_off}. Student's syllabus and weak areas: {priorities or 'Not provided'}.
First identify the topics/chapters you can verify from the excerpts. Then make a phased plan with daily targets, revision, active recall, and buffer time. Do not claim the PDF is the official syllabus. Flag any missing syllabus information."""
            with st.spinner("Building your study strategy..."):
                answer, sources = ask_tutor(request, "Syllabus planning", top_k=10)
            st.markdown(answer)
            with st.expander("Planning sources"):
                for s in sources:
                    st.caption(f"{s['source']} · page {s['page']}")
        except Exception as exc:
            st.error(str(exc))

with tab_test:
    st.subheader("Test what you know")
    st.write("Generate a practice test from the material in your PDF library.")
    q_count = st.select_slider("Number of questions", options=[5, 10, 15, 20], value=10)
    if st.button("Generate practice test", type="primary"):
        try:
            with st.spinner("Writing questions from your books..."):
                st.session_state["practice_test"] = generate_random_questions(q_count)
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get("practice_test"):
        st.markdown(st.session_state["practice_test"])
        if st.button("Reveal answer key"):
            prompt = f"""Provide a concise answer key with short explanations for these questions:
{st.session_state['practice_test']}
Use only this study material:
{all_source_text()}"""
            try:
                response = get_client().chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.2,
                    max_tokens=1500,
                )
                st.markdown(response.choices[0].message.content)
            except Exception as exc:
                st.error(str(exc))

with tab_notes:
    st.subheader("Clean, exam-ready notes")
    topic = st.text_input("Topic or chapter for notes", placeholder="e.g. Cell division")
    note_style = st.selectbox("Notes format", ["Concise revision notes", "Detailed study notes", "Definitions and key terms", "Last-minute exam sheet"])
    if st.button("Create notes", type="primary", disabled=not topic.strip()):
        try:
            request = f"""Create {note_style.lower()} for: {topic}.
Use clear headings and bullet points. Include key definitions, processes, important distinctions, and one simple example if supported.
Remove duplicated ideas. Do not add unrelated information. End with 3 quick recall questions."""
            with st.spinner("Preparing your notes..."):
                answer, sources = ask_tutor(request, "Note-making", top_k=8)
            st.markdown(answer)
            with st.expander("Sources used"):
                for s in sources:
                    st.caption(f"{s['source']} · page {s['page']}")
        except Exception as exc:
            st.error(str(exc))

st.markdown("---")
st.caption("Syllabus Studio · PDF-grounded learning. Always cross-check official syllabus requirements with your institution.")
