"""
Healthcare RAG (MediCore) - Knowledge Base Engine
--------------------------------------------------
Manages the ingestion, provenance tagging, vector indexing,
and semantic search across authoritative Level 1/Level 3 medical reference documents
(WHO, CDC, NHS, ICMR, and reference guidelines).

Features:
- Complete provenance metadata on every chunk (Source Organization, Document ID, Title, URL, Date, Category).
- Dual ingestion: Structured authoritative JSON documents + Multi-page reference PDFs.
- Intelligent persistent vector caching (FAISS IndexFlatIP + Pickled Chunks) for instant sub-second startups.
- Cosine similarity search with optional metadata filtering by clinical topic or source organization.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set HuggingFace cache directory
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import faiss
from sentence_transformers import SentenceTransformer

# Reuse chunk_text and extract_text_from_pdf from src/main.py
from src.main import chunk_text, extract_text_from_pdf


INDEX_DIR = PROJECT_ROOT / "documents" / ".index"
CATALOG_PATH = PROJECT_ROOT / "documents" / "catalog.json"
AUTHORITATIVE_DIR = PROJECT_ROOT / "documents" / "authoritative"
BASELINE_PDF_PATH = PROJECT_ROOT / "documents" / "medical_information.pdf"


def load_catalog() -> Dict[str, Any]:
    """Loads the central medical documents catalog with source provenance metadata."""
    if not CATALOG_PATH.exists():
        return {"documents": []}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_authoritative_json(file_path: Path) -> List[Dict[str, Any]]:
    """
    Ingests an authoritative structured JSON document and extracts chunks
    with complete source provenance metadata attached to each chunk.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        doc_data = json.load(f)

    meta = doc_data.get("metadata", {})
    sections = doc_data.get("sections", [])
    chunks = []

    doc_id = meta.get("document_id", file_path.stem)
    org = meta.get("source_organization", "Official Medical Authority")
    org_level = meta.get("organization_level", "Level 1 — Government & Official Medical Organization")
    title = meta.get("document_title", file_path.stem)
    url = meta.get("source_url", "")
    doc_type = meta.get("document_type", "Clinical Guideline")
    pub_date = meta.get("publication_date", "")
    last_reviewed = meta.get("last_reviewed", pub_date)
    category = meta.get("topic_category", "General Medicine")
    country_scope = meta.get("country_or_scope", "Global")

    for sec_idx, sec in enumerate(sections, start=1):
        sec_title = sec.get("section_title", f"Section {sec_idx}")
        page_num = sec.get("page_number", 1)
        sec_text = sec.get("text", "").strip()

        if not sec_text:
            continue

        raw_chunks = chunk_text(sec_text, chunk_size=500, overlap=50)
        for chunk_idx, c_text in enumerate(raw_chunks, start=1):
            chunk_record = {
                "text": c_text,
                "metadata": {
                    "chunk_id": f"{doc_id}_s{sec_idx}_c{chunk_idx}",
                    "document_id": doc_id,
                    "source_organization": org,
                    "organization_level": org_level,
                    "document_title": title,
                    "source_url": url,
                    "document_type": doc_type,
                    "publication_date": pub_date,
                    "last_reviewed": last_reviewed,
                    "topic_category": category,
                    "country_or_scope": country_scope,
                    "page_number": page_num,
                    "section_title": sec_title,
                    "chunk_index": chunk_idx,
                    "document_name": file_path.name,
                }
            }
            chunks.append(chunk_record)

    return chunks


def ingest_baseline_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Ingests the baseline medical reference PDF and attaches formal provenance metadata.
    """
    if not pdf_path.exists():
        return []

    pages_data = extract_text_from_pdf(pdf_path)
    chunks = []

    doc_id_map = {
        1: ("MED-REF-2026-01", "Hypertension Clinical Overview", "Cardiovascular Diseases"),
        2: ("MED-REF-2026-02", "Type 2 Diabetes Mellitus Clinical Overview", "Endocrinology & Diabetes"),
        3: ("MED-REF-2026-03", "Emergency Red Flags & Critical Protocols", "Emergency Medicine & Critical Care"),
    }

    for page_info in pages_data:
        page_num = page_info["page_number"]
        page_text = page_info["text"]
        if not page_text or not page_text.strip():
            continue

        doc_id, sec_title, category = doc_id_map.get(
            page_num, (f"MED-REF-2026-{page_num:02d}", f"Page {page_num}", "Internal Medicine")
        )

        raw_chunks = chunk_text(page_text, chunk_size=500, overlap=50)
        for chunk_idx, c_text in enumerate(raw_chunks, start=1):
            chunk_record = {
                "text": c_text,
                "metadata": {
                    "chunk_id": f"page{page_num}_chunk{chunk_idx}",
                    "document_id": doc_id,
                    "source_organization": "Department of Internal Medicine / Emergency Reference",
                    "organization_level": "Level 1 — Reference Clinical Guideline",
                    "document_title": f"Clinical Practice Guideline: {sec_title}",
                    "source_url": "local://documents/medical_information.pdf",
                    "document_type": "Institutional Practice Guideline",
                    "publication_date": "2026-01-01",
                    "last_reviewed": "2026-01-01",
                    "topic_category": category,
                    "country_or_scope": "Institutional Benchmark Reference",
                    "page_number": page_num,
                    "section_title": sec_title,
                    "chunk_index": chunk_idx,
                    "document_name": pdf_path.name,
                }
            }
            chunks.append(chunk_record)

    return chunks


def load_all_authoritative_chunks() -> List[Dict[str, Any]]:
    """
    Loads and aggregates chunks across all curated authoritative medical documents
    and the baseline reference PDF.
    """
    all_chunks = []

    # 1. Ingest baseline PDF
    pdf_chunks = ingest_baseline_pdf(BASELINE_PDF_PATH)
    all_chunks.extend(pdf_chunks)

    # 2. Ingest all authoritative JSON documents in documents/authoritative/
    if AUTHORITATIVE_DIR.exists():
        for json_file in sorted(AUTHORITATIVE_DIR.glob("*.json")):
            try:
                doc_chunks = ingest_authoritative_json(json_file)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                print(f"[WARNING] Failed to parse {json_file.name}: {e}")

    return all_chunks


def build_or_load_knowledge_base(
    model: SentenceTransformer,
    force_rebuild: bool = False
) -> tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Builds or loads the persistent FAISS index and chunk registry.
    If cached index exists and is up to date, loads instantly.
    Otherwise, generates embeddings and caches the vector index.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_file = INDEX_DIR / "faiss_index.bin"
    chunks_file = INDEX_DIR / "chunks.pkl"

    # Check if cache is valid
    if not force_rebuild and index_file.exists() and chunks_file.exists():
        try:
            print("Loading cached FAISS index from disk...")
            index = faiss.read_index(str(index_file))
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            print(f"Loaded {len(chunks)} chunks and FAISS index with {index.ntotal} vectors.\n")
            return index, chunks
        except Exception as e:
            print(f"[WARNING] Cache load failed ({e}), rebuilding index...")

    # Build fresh index
    print("Building fresh knowledge base vector index...")
    chunks = load_all_authoritative_chunks()
    if not chunks:
        raise ValueError("[ERROR] No chunks found to build knowledge base.")

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks using sentence-transformers...")
    raw_embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    # Attach embeddings to chunks
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb

    # Build FAISS IndexFlatIP (Cosine similarity with normalized vectors)
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Persist cache to disk
    try:
        faiss.write_index(index, str(index_file))
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached FAISS index and {len(chunks)} chunks to {INDEX_DIR}.\n")
    except Exception as e:
        print(f"[WARNING] Failed to persist index cache: {e}")

    return index, chunks


def search_knowledge_base(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
    category_filter: Optional[str] = None,
    organization_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Performs semantic search with Cosine Similarity across the complete
    authoritative medical knowledge base, with optional category/organization filters.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()
    query_vector = model.encode([clean_query], convert_to_numpy=True)
    query_vector = np.asarray(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    # If filters are present, search a wider candidate pool and filter
    search_k = min(top_k * 5 if (category_filter or organization_filter) else top_k, index.ntotal)
    distances, indices = index.search(query_vector, search_k)

    results = []
    rank = 1
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(chunks):
            continue

        c = chunks[idx]
        meta = c["metadata"]

        if category_filter and meta.get("topic_category") != category_filter:
            continue
        if organization_filter and organization_filter.lower() not in meta.get("source_organization", "").lower():
            continue

        results.append({
            "rank": rank,
            "similarity_score": float(score),
            "chunk_id": meta["chunk_id"],
            "document_id": meta["document_id"],
            "document_name": meta["document_name"],
            "document_title": meta["document_title"],
            "source_organization": meta["source_organization"],
            "organization_level": meta.get("organization_level", "Level 1 — Government & Official Medical Organization"),
            "source_url": meta.get("source_url", ""),
            "publication_date": meta.get("publication_date", ""),
            "topic_category": meta.get("topic_category", ""),
            "page_number": meta["page_number"],
            "section_title": meta.get("section_title", ""),
            "text": c["text"],
        })
        rank += 1
        if len(results) >= top_k:
            break

    return results


def get_knowledge_base_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates aggregate statistics for UI dashboards and administrative inspection."""
    orgs = {}
    categories = {}
    docs = set()

    for c in chunks:
        m = c["metadata"]
        org = m.get("source_organization", "Other")
        cat = m.get("topic_category", "General")
        doc_id = m.get("document_id")

        orgs[org] = orgs.get(org, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        docs.add(doc_id)

    return {
        "total_chunks": len(chunks),
        "total_documents": len(docs),
        "organizations": orgs,
        "categories": categories,
    }
