# Syllabus Studio — PDF-powered AI Study Tutor

A Streamlit study tutor using local PDF ingestion, Sentence Transformers embeddings, FAISS retrieval, and Groq's `openai/gpt-oss-120b` chat model.

## Features
- Upload textbook PDFs and index them into FAISS
- Page-aware chunk metadata (PDF filename, page, chunk number)
- Retrieval-grounded explanations with source references
- Syllabus planning, study strategy, practice questions, and revision notes
- Clean Streamlit interface

## Requirements
- Python 3.10–3.12 recommended
- A Groq API key
- Internet connection for the first embedding-model download and Groq requests

## Setup in VS Code (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your Groq API key.

Put your textbook PDFs in `data/pdfs/` (or upload them through the app sidebar), then run:

```powershell
streamlit run app.py
```

In the app, click **Build / refresh knowledge base** after adding PDFs.

## Project layout

```text
study_tutor_rag/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── pdfs/              # add PDFs here; do not commit copyrighted books
└── FAISS_index/           # generated index + metadata; do not commit
```

## GitHub safety
- Never upload `.env`, API keys, private PDFs, or the generated FAISS index.
- `.gitignore` excludes `.env`, `data/pdfs/`, and `FAISS_index/` should be added before first commit if you want to keep books/indexes private. Add these two lines to `.gitignore` if needed:
  ```
  data/pdfs/
  FAISS_index/
  ```
- On Streamlit Community Cloud, add `GROQ_API_KEY` under app Settings → Secrets. Install packages from `requirements.txt`. The app's local FAISS index will not automatically exist in a fresh cloud deployment; include a permitted source corpus or build the index during deployment/runtime.

## Important limitations
- This starter handles text-based PDFs. Scanned/image-only PDFs require OCR.
- The embedding model downloads on first run and runs locally; it does not use your Groq API key.
- The app's first version uses exact vector similarity with normalized embeddings. For production-grade RAG, consider hybrid BM25 + vector retrieval, reranking, query rewriting, and a persistent document-management layer.
- Planning is grounded in the indexed books, not necessarily your institution's official syllabus. Upload/paste the official syllabus to get a more reliable schedule.
