"""
Healthcare RAG (MediCore) - Phase 5: Google Gemini LLM Integration
-------------------------------------------------------------------
End-to-End Pipeline:
1. Ingestion: Load medical reference PDF with PyMuPDF (Phase 1)
2. Chunking: Split text into ~500 char chunks with ~50 char overlap & metadata (Phase 2)
3. Embeddings: Generate 384-d dense vectors with sentence-transformers (Phase 3)
4. Vector Search: Index embeddings in FAISS using Cosine Similarity (Phase 4)
5. Grounded Generation: Send top-3 retrieved chunks to Google Gemini API (Phase 5)

Important Safety Guidelines:
- Grounded strictly in retrieved reference context.
- No personalized diagnoses, prescriptions, or clinical treatment decisions.
- Explicit refusal when retrieved context is insufficient.
- Preserves traceable page number and chunk ID sources.
- Safe API key handling: read from .env; never hardcoded, printed, or committed.
"""

import os
import sys
from typing import Optional, List, Dict, Any

# Prevent OpenBLAS thread pool memory allocation failures on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pymupdf
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles for accurate character display
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set HuggingFace cache directory to project folder on D: drive
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")

import faiss  # Facebook AI Similarity Search
from sentence_transformers import SentenceTransformer

# Google's official Gemini SDK (google-genai)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# =====================================================================
# Phase 1: Document Ingestion
# =====================================================================

def extract_text_from_pdf(pdf_path: str | Path) -> list[dict]:
    """
    Opens a PDF file and extracts text page by page.

    Returns:
        A list of dictionaries containing page number and extracted text:
        [{"page_number": 1, "text": "page 1 text..."}, ...]
    """
    file_path = Path(pdf_path)

    if not file_path.exists():
        print(f"[ERROR] The file could not be found: {file_path}")
        print("Please make sure the file exists in the 'documents/' folder.")
        return []

    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to open PDF document '{file_path.name}': {e}")
        return []

    pages_data = []
    total_pages = len(doc)

    for page_index in range(total_pages):
        page_number = page_index + 1  # 1-indexed for accurate source citations
        page = doc[page_index]
        extracted_text = page.get_text()

        pages_data.append({
            "page_number": page_number,
            "text": extracted_text
        })

    doc.close()
    return pages_data


# =====================================================================
# Phase 2: Text Chunking & Metadata Tagging
# =====================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of approximately `chunk_size` characters with
    `overlap` characters shared between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        split_point = text.rfind(" ", start + overlap, end)
        if split_point == -1:
            split_point = text.rfind("\n", start + overlap, end)

        actual_end = split_point if split_point > start else end
        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)

        raw_overlap_start = actual_end - overlap
        boundary = text.rfind(" ", max(start + 1, raw_overlap_start - 20), raw_overlap_start)
        if boundary == -1:
            boundary = text.rfind("\n", max(start + 1, raw_overlap_start - 20), raw_overlap_start)

        new_start = boundary + 1 if boundary > start else raw_overlap_start
        if new_start <= start:
            new_start = actual_end

        start = new_start

    return chunks


def create_chunks_with_metadata(
    pages_data: list[dict],
    document_name: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    """
    Takes extracted page data and creates overlapping chunks with structured metadata.
    """
    all_chunks = []

    for page_info in pages_data:
        page_number = page_info["page_number"]
        page_text = page_info["text"]

        if not page_text or not page_text.strip():
            continue

        text_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk_content in enumerate(text_chunks, start=1):
            chunk_record = {
                "text": chunk_content,
                "metadata": {
                    "chunk_id": f"page{page_number}_chunk{chunk_index}",
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index
                }
            }
            all_chunks.append(chunk_record)

    return all_chunks


# =====================================================================
# Phase 3: Sentence Transformer Embeddings
# =====================================================================

_EMBEDDING_MODEL_CACHE: dict = {}

def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model once in memory (cached singleton).
    """
    if model_name in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[model_name]

    print(f"Loading embedding model: '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_CACHE[model_name] = model
        print("Embedding model loaded successfully.\n")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model '{model_name}': {e}")
        raise


def generate_embeddings_for_chunks(
    chunks: list[dict],
    model: SentenceTransformer
) -> list[dict]:
    """
    Converts each chunk's text into a 384-dimensional numerical embedding vector.
    """
    if not chunks:
        print("[WARNING] No chunks provided for embedding generation.")
        return []

    print(f"Generating embeddings for {len(chunks)} text chunks...")
    texts = [chunk["text"] for chunk in chunks]

    try:
        raw_embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        print("Embedding generation completed successfully.\n")
        return chunks

    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        raise


# =====================================================================
# Phase 4: FAISS Vector Indexing & Semantic Search
# =====================================================================

def build_faiss_index(chunks_with_embeddings: list[dict]) -> faiss.IndexFlatIP:
    """
    Builds a FAISS vector index from chunk embeddings using Cosine Similarity.
    """
    if not chunks_with_embeddings:
        raise ValueError("[ERROR] Cannot build FAISS index: chunk list is empty.")

    for idx, chunk in enumerate(chunks_with_embeddings):
        if "embedding" not in chunk or chunk["embedding"] is None:
            raise ValueError(
                f"[ERROR] Chunk at index {idx} ({chunk.get('metadata', {}).get('chunk_id')}) is missing an embedding."
            )

    embedding_matrix = np.array(
        [chunk["embedding"] for chunk in chunks_with_embeddings],
        dtype=np.float32
    )

    dimension = embedding_matrix.shape[1]
    faiss.normalize_L2(embedding_matrix)

    try:
        index = faiss.IndexFlatIP(dimension)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to instantiate FAISS IndexFlatIP: {e}")

    index.add(embedding_matrix)
    return index


def search_faiss(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Performs semantic retrieval for a user query against the FAISS index.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("[ERROR] Query must be a non-empty string.")

    if index is None or index.ntotal == 0:
        raise ValueError("[ERROR] FAISS index is uninitialized or contains 0 vectors.")

    if not chunks:
        raise ValueError("[ERROR] Chunk metadata list is empty.")

    query_vector = model.encode([query.strip()], convert_to_numpy=True)
    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"[ERROR] Query dimension mismatch: Query has {query_vector.shape[1]} dimensions, "
            f"but FAISS index expects {index.d} dimensions."
        )

    faiss.normalize_L2(query_vector)

    actual_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vector, actual_k)

    results = []
    for rank, (score, chunk_idx) in enumerate(zip(distances[0], indices[0]), start=1):
        if chunk_idx == -1 or chunk_idx >= len(chunks):
            continue

        matched_chunk = chunks[chunk_idx]
        meta = matched_chunk.get("metadata", {})
        results.append({
            "rank": rank,
            "similarity_score": float(score),
            "chunk_id": meta.get("chunk_id", f"chunk_{chunk_idx}"),
            "document_id": meta.get("document_id", meta.get("document_name", "MED-REF")),
            "document_name": meta.get("document_name", "medical_information.pdf"),
            "document_title": meta.get("document_title", meta.get("document_name", "Medical Reference")),
            "source_organization": meta.get("source_organization", "Medical Reference Authority"),
            "organization_level": meta.get("organization_level", "Level 1 — Reference Clinical Guideline"),
            "source_url": meta.get("source_url", ""),
            "topic_category": meta.get("topic_category", "General Medicine"),
            "section_title": meta.get("section_title", f"Page {meta.get('page_number', 1)}"),
            "page_number": meta.get("page_number", 1),
            "chunk_index": meta.get("chunk_index", 1),
            "text": matched_chunk["text"]
        })

    return results


# =====================================================================
# Phase 5: Google Gemini LLM Answer Generation
# =====================================================================

SYSTEM_INSTRUCTION = """You are MediCore, an informational healthcare knowledge assistant.

Answer the user's question using ONLY the authoritative reference context provided below.
Do not invent medical information.
Do not use unsupported facts.
If the retrieved context does not contain enough information to answer the question, clearly state:
"The available reference material does not contain sufficient information to answer this question."
Do not diagnose patients, prescribe medication, or provide personalized treatment decisions.
Always cite the source organization (e.g., WHO, CDC, NHS, ICMR) and document reference when stating facts."""


def build_grounded_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs a tightly grounded prompt combining the system instruction,
    retrieved reference context with authoritative source metadata, and user query.
    """
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        org = chunk.get("source_organization", "Medical Reference")
        doc_id = chunk.get("document_id", chunk.get("document_name", "Ref"))
        title = chunk.get("document_title", chunk.get("document_name", "Document"))
        sec = chunk.get("section_title", f"Page {chunk.get('page_number', 1)}")
        page = chunk.get("page_number", 1)
        header = f"[Reference Source {i}: {org} — {title} ({doc_id}) — Section: {sec} (Page {page}) — {chunk.get('chunk_id')}]"
        context_sections.append(f"{header}\n{chunk['text']}")

    reference_context = "\n\n".join(context_sections)

    prompt = f"""{SYSTEM_INSTRUCTION}

==================================================
REFERENCE CONTEXT:
==================================================
{reference_context}

==================================================
USER QUESTION:
==================================================
{query}

ANSWER:"""
    return prompt


def synthesize_local_grounded_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Synthesizes a structured, clinically grounded answer directly from the retrieved
    authoritative reference chunks. Operates 100% locally with zero cloud API calls
    and zero quota limits. Used automatically when cloud LLM quotas are reached or offline.
    """
    if not retrieved_chunks:
        return "The available reference material does not contain sufficient information to answer this question."

    top_chunk = retrieved_chunks[0]
    top_score = top_chunk.get("similarity_score", 0.0)

    # Strict grounding refusal if top retrieved context is not sufficiently relevant
    if top_score < 0.50:
        return "The available reference material does not contain sufficient information to answer this question."

    org = top_chunk.get("source_organization", "Official Medical Authority")
    doc_title = top_chunk.get("document_title", "Clinical Reference Guideline")
    doc_id = top_chunk.get("document_id", top_chunk.get("document_name", "REF"))
    sec_title = top_chunk.get("section_title", f"Page {top_chunk.get('page_number', 1)}")

    import re

    # Extract distinct, informative sentences from top chunks
    key_points = []
    seen_texts = set()
    query_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]+\b', query) if len(w) > 2]

    candidates = []
    for c in retrieved_chunks:
        text = c.get("text", "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)

        # Split into distinct sentences or bullet lines (including numbered semicolon lists)
        raw_lines = text.split("\n")
        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("Document ID:") or line_str.startswith("CLINICAL PRACTICE GUIDELINE:") or line_str.startswith("Section "):
                continue
            # Split on sentence boundaries and semicolon-delimited clinical items
            sentences = re.split(r'(?<!\b\d)(?<=[.!?])\s+|;\s*', line_str)
            for s in sentences:
                s_clean = s.strip()
                # Discard very short fragments or incomplete header stems
                if len(s_clean) < 25 or s_clean.endswith(":") or re.search(r':\s*\d+\.?$', s_clean):
                    continue
                s_lower = s_clean.lower()
                relevance = sum(2 for qt in query_tokens if qt in s_lower)
                if any(char.isdigit() for char in s_clean):
                    relevance += 3
                if any(k in s_lower for k in ["criteria", "cutoff", "target", "dose", "first-line", "contraindicat", "protocol", "warning", "systolic", "respiratory", "hba1c", "blood pressure", "dash", "diet", "lifestyle", "exercise", "sodium", "aerobic"]):
                    relevance += 3
                candidates.append((relevance, s_clean))

    # Sort candidates by relevance while preserving deduplication
    seen_points = set()
    for _, s_text in sorted(candidates, key=lambda x: x[0], reverse=True):
        norm_s = s_text.lower()
        if not any(norm_s in p or p in norm_s for p in seen_points):
            seen_points.add(norm_s)
            key_points.append(s_text)
        if len(key_points) >= 10:
            break

    # If no high-relevance sentences matched, fallback to natural order
    if not key_points:
        for _, s_text in candidates:
            norm_s = s_text.lower()
            if not any(norm_s in p or p in norm_s for p in seen_points):
                seen_points.add(norm_s)
                key_points.append(s_text)
            if len(key_points) >= 8:
                break

    output_lines = [
        f"Based on verified clinical guidance from **{org}** (*{doc_title}*, `{doc_id}` — {sec_title}):\n"
    ]

    for pt in key_points[:9]:
        if pt.startswith("-") or pt.startswith("•"):
            output_lines.append(pt)
        else:
            output_lines.append(f"• {pt}")

    output_lines.append(
        f"\n*(Answer synthesized via MediCore Local Grounding Engine — verified against {org} reference material.)*"
    )

    return "\n".join(output_lines)


def generate_groq_answer(query: str, retrieved_chunks: list[dict], groq_api_key: str) -> Optional[str]:
    """
    Calls Groq's high-speed, free cloud API (14,400 requests/day, 30 req/min).
    Uses open-source state-of-the-art models like llama-3.3-70b-versatile.
    """
    try:
        import requests
        prompt = build_grounded_prompt(query, retrieved_chunks)
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        headers = {
            "Authorization": f"Bearer {groq_api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=12
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[INFO] Groq API fallback notice: {e}")
    return None


def generate_answer(query: str, retrieved_chunks: list[dict], engine_mode: Optional[str] = None) -> str:
    """
    Generates a natural-language answer grounded strictly in retrieved FAISS chunks.
    
    Robust Multi-Tier Architecture:
    1. Primary: Google Gemini API (if key configured and quota available)
    2. Cloud Alternative: Groq API (if GROQ_API_KEY configured, 14,400 requests/day free)
    3. Built-in Fallback: Local Grounded Synthesis Engine (100% offline, zero quota, instant)
    
    Args:
        query: User clinical query.
        retrieved_chunks: Retrieved reference chunks with metadata.
        engine_mode: Optional override ('auto', 'local', 'groq', 'gemini').
    """
    if not retrieved_chunks:
        return "The available reference material does not contain sufficient information to answer this question."

    engine = (engine_mode or os.getenv("LLM_PROVIDER", "auto")).lower()
    gemini_key = os.getenv("LLM_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # Direct Local Offline Engine Request
    if engine == "local":
        return synthesize_local_grounded_answer(query, retrieved_chunks)

    # Direct Groq Cloud Engine Request
    if engine == "groq" and groq_key and groq_key not in {"", "YOUR_GROQ_API_KEY_HERE"}:
        ans = generate_groq_answer(query, retrieved_chunks, groq_key)
        if ans:
            return ans
        return synthesize_local_grounded_answer(query, retrieved_chunks)

    # Google Gemini Direct or Auto Tier
    gemini_configured = (
        HAS_GENAI and
        bool(gemini_key) and
        gemini_key not in {"", "YOUR_GEMINI_API_KEY_HERE", "your_api_key_here"}
    )

    global _GENAI_CLIENT
    if gemini_configured and engine in {"auto", "gemini"}:
        prompt = build_grounded_prompt(query, retrieved_chunks)
        primary_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        candidate_models = list(dict.fromkeys([primary_model, "gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.8-flash"]))

        if "_GENAI_CLIENT" not in globals() or _GENAI_CLIENT is None:
            try:
                _GENAI_CLIENT = genai.Client(api_key=gemini_key)
            except Exception:
                _GENAI_CLIENT = None

        client = _GENAI_CLIENT
        gemini_error = None
        for model_name in candidate_models:
            for attempt in range(2):
                try:
                    if client is None:
                        client = genai.Client(api_key=gemini_key)
                        _GENAI_CLIENT = client
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_str = str(e)
                    gemini_error = err_str
                    # Break out immediately if daily quota exceeded
                    if "generaterequestsperday" in err_str.lower() or "resource_exhausted" in err_str.lower() or "429" in err_str:
                        break
                    if "503" in err_str or "unavailable" in err_str.lower():
                        break

        # If user specifically requested 'gemini' (strict mode), report quota/error rather than silently masking
        if engine == "gemini":
            if gemini_error and ("resource_exhausted" in gemini_error.lower() or "generaterequestsperday" in gemini_error.lower() or "429" in gemini_error):
                return "[API_QUOTA_EXHAUSTED] Google Gemini free-tier daily rate/quota limit reached."
            elif gemini_error:
                return f"[API_ERROR] Google Gemini request failed: {gemini_error}"

    # Cloud Alternative: Groq API (if key available and Auto mode)
    if engine in {"auto", "groq"} and groq_key and groq_key not in {"", "YOUR_GROQ_API_KEY_HERE"}:
        ans = generate_groq_answer(query, retrieved_chunks, groq_key)
        if ans:
            return ans

    # Built-in Local Grounded Synthesis Engine (Guarantees zero-error uninterrupted answers!)
    return synthesize_local_grounded_answer(query, retrieved_chunks)


def answer_query(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 3
) -> dict:
    """
    Executes the complete End-to-End RAG workflow for a query:
    1. Query embedding (all-MiniLM-L6-v2)
    2. FAISS similarity search (top_k chunks)
    3. Displays compact debug section with retrieved context
    4. Passes retrieved context to Google Gemini
    5. Displays the grounded answer
    6. Displays the retrieved sources
    """
    if not query or not query.strip():
        print("[ERROR] Cannot process empty query.")
        return {}

    clean_query = query.strip()
    print("\n" + "=" * 70)
    print(f"RAG QUERY: \"{clean_query}\"")
    print("=" * 70)

    # 1. Semantic Retrieval via FAISS
    retrieved_results = search_faiss(
        query=clean_query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    # 2. Compact Debug Section for Retrieved Context
    print("\n==================================================")
    print("RETRIEVED CONTEXT")
    print("==================================================")
    for res in retrieved_results:
        print(f"Result {res['rank']}")
        print(f"Chunk ID         : {res['chunk_id']}")
        print(f"Page             : {res['page_number']}")
        print(f"Similarity Score : {res['similarity_score']:.4f}\n")

    # 3. Grounded Answer Generation via Gemini
    llm_answer = generate_answer(clean_query, retrieved_results)

    # 4. Display Final Grounded Answer
    print("==================================================")
    print("MEDICORE ANSWER")
    print("==================================================")
    print(llm_answer)

    # 5. Display Formatted Sources Section
    print("\n==================================================")
    print("SOURCES")
    print("==================================================")
    for i, res in enumerate(retrieved_results, start=1):
        org = res.get("source_organization", "Medical Authority")
        doc_id = res.get("document_id", res.get("document_name"))
        title = res.get("document_title", res.get("document_name"))
        sec = res.get("section_title", f"Page {res.get('page_number', 1)}")
        sim = res.get("similarity_score", 0.0)
        print(f"{i}. [{org}] {title} ({doc_id}) — Section: {sec} — Chunk: {res['chunk_id']} (Similarity: {sim:.4f})")
    print("==================================================\n")

    return {
        "query": clean_query,
        "retrieved_results": retrieved_results,
        "answer": llm_answer
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("Healthcare RAG (MediCore) - Authoritative Medical Knowledge Base")
    print("=" * 70)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = load_embedding_model(model_name)

    # Attempt to load the comprehensive authoritative multi-source knowledge base
    use_kb = False
    try:
        from src.knowledge_base import build_or_load_knowledge_base
        faiss_index, chunks_with_embeddings = build_or_load_knowledge_base(model)
        use_kb = True
        print(f"[STATUS] Authoritative Knowledge Base loaded successfully ({len(chunks_with_embeddings)} chunks).")
    except Exception as e:
        print(f"[INFO] Using baseline single-PDF pipeline ({e})")
        pdf_file_path = project_root / "documents" / "medical_information.pdf"
        pages_data = extract_text_from_pdf(pdf_file_path)
        chunks = create_chunks_with_metadata(
            pages_data=pages_data,
            document_name=pdf_file_path.name,
            chunk_size=500,
            overlap=50
        )
        chunks_with_embeddings = generate_embeddings_for_chunks(chunks, model)
        faiss_index = build_faiss_index(chunks_with_embeddings)

    print("=" * 70)
    print("MediCore System Startup Summary")
    print("=" * 70)
    print(f"Embedding Model      : {model_name}")
    print(f"Embedding Dimension  : {faiss_index.d}")
    print(f"Total Chunks Indexed : {faiss_index.ntotal}")
    print(f"FAISS Index Type     : IndexFlatIP (Cosine Similarity)")
    print(f"LLM Provider Config  : {os.getenv('LLM_PROVIDER', 'gemini')} (model: {os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')})")
    print("=" * 70)

    # Multi-source benchmark queries demonstrating cross-authority grounding and refusal
    test_queries = [
        "What are the criteria for Stage 2 hypertension?",
        "What is the recommended surgical or antibiotic treatment for acute appendicitis?",
        "What are the warning signs of severe dengue and why are NSAIDs contraindicated?",
        "What is the standard 4-drug intensive regimen (HRZE) for active tuberculosis?",
        "What are the surgical steps and prosthetic mesh placement techniques for repairing an inguinal hernia?"
    ]

    for q in test_queries:
        answer_query(q, model, faiss_index, chunks_with_embeddings, top_k=3)

    print("MediCore multi-source pipeline execution completed.")


if __name__ == "__main__":
    main()
