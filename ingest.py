"""
PDF ingestion for the Study Tutor RAG app.

Reads PDFs from data/pdfs/, extracts text, chunks it, embeds each chunk with
a local SentenceTransformer model, and saves a FAISS index plus metadata.
The FAISS_index/ directory is generated locally and intentionally gitignored.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
INDEX_DIR = ROOT / "FAISS_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000       # approximate characters
CHUNK_OVERLAP = 180     # overlap prevents cutting context at boundaries


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append({
                "text": text,
                "source": pdf_path.name,
                "page": page_number,
            })
    return pages


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split near paragraph/sentence boundaries, with a small overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer a nearby paragraph or sentence boundary.
            boundary = max(text.rfind("\n", start, end),
                           text.rfind(". ", start, end),
                           text.rfind("? ", start, end),
                           text.rfind("! ", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest_pdfs(pdf_paths: Iterable[Path] | None = None) -> tuple[int, str]:
    paths = list(pdf_paths) if pdf_paths is not None else sorted(PDF_DIR.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(
            f"No PDFs found. Add PDF files to: {PDF_DIR}"
        )

    records = []
    for path in paths:
        for page in extract_pdf_pages(path):
            for chunk_number, chunk in enumerate(split_into_chunks(page["text"]), start=1):
                records.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_number,
                })

    if not records:
        raise ValueError("No extractable text found. The PDFs may be scanned images; OCR is needed.")

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [r["text"] for r in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity on normalized vectors
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "study.index"))
    with (INDEX_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with (INDEX_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"embedding_model": MODEL_NAME, "dimension": int(vectors.shape[1])}, f, indent=2)

    return len(records), str(INDEX_DIR)


if __name__ == "__main__":
    count, location = ingest_pdfs()
    print(f"Indexed {count} chunks. Saved index to {location}")
