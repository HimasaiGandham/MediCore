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

# Set HuggingFace cache directory to project folder on D: drive
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model once in memory.
    """
    print(f"Loading embedding model: '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
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
        results.append({
            "rank": rank,
            "similarity_score": float(score),
            "chunk_id": matched_chunk["metadata"]["chunk_id"],
            "document_name": matched_chunk["metadata"]["document_name"],
            "page_number": matched_chunk["metadata"]["page_number"],
            "chunk_index": matched_chunk["metadata"]["chunk_index"],
            "text": matched_chunk["text"]
        })

    return results


# =====================================================================
# Phase 5: Google Gemini LLM Answer Generation
# =====================================================================

SYSTEM_INSTRUCTION = """You are MediCore, an informational healthcare knowledge assistant.

Answer the user's question using ONLY the reference context provided below.
Do not invent medical information.
Do not use unsupported facts.
If the retrieved context does not contain enough information to answer the question, clearly state:
"The available reference material does not contain sufficient information to answer this question."
Do not diagnose patients, prescribe medication, or provide personalized treatment decisions.
Keep answers informational and cite the relevant document name and page number when stating facts."""


def build_grounded_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Constructs a tightly grounded prompt combining the system instruction,
    retrieved reference context with source metadata, and user query.
    """
    context_sections = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        header = f"[Reference Source {i}: {chunk['document_name']} — Page {chunk['page_number']} — {chunk['chunk_id']}]"
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


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Generates a natural-language answer using Google's Gemini API, grounded
    strictly in the retrieved FAISS chunks.

    Requirements:
    - Reads LLM_API_KEY from .env
    - Never prints or exposes the API key
    - If API key is missing or unconfigured, displays a clear beginner-friendly message
    - Returns informational answers or explicit refusal when context is insufficient
    """
    if not retrieved_chunks:
        return "The available reference material does not contain sufficient information to answer this question."

    # Retrieve API key from environment (.env)
    api_key = os.getenv("LLM_API_KEY")

    # Beginner-friendly error handling if API key is not configured
    if not api_key or api_key.strip() in {"", "YOUR_GEMINI_API_KEY_HERE", "your_api_key_here"}:
        return "Gemini API key not configured. Add LLM_API_KEY to .env."

    if not HAS_GENAI:
        return "[ERROR] google-genai library is not installed. Please run: pip install -r requirements.txt"

    prompt = build_grounded_prompt(query, retrieved_chunks)

    # Try up to 4 attempts with dynamic backoff for transient server spikes and rate limits (429/503)
    for attempt in range(4):
        try:
            # Initialize Google's official Gemini client
            client = genai.Client(api_key=api_key.strip())

            # Generate grounded response using Gemini 3.8 Flash
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )

            if response and response.text:
                return response.text.strip()
            else:
                return "The model returned an empty response."

        except Exception as e:
            err_str = str(e)
            transient_patterns = ["503", "429", "UNAVAILABLE", "RemoteProtocolError", "Connection", "Timeout", "EOF"]
            is_transient = any(pat in err_str or pat in type(e).__name__ for pat in transient_patterns)
            if "generaterequestsperday" in err_str.lower():
                return f"[ERROR] Gemini API request failed: {type(e).__name__} - Daily quota reached (20 requests/day free tier)"

            if is_transient and attempt < 2:
                import time
                import re
                retry_match = re.search(r"Please retry in ([\d\.]+)s", err_str)
                if not retry_match:
                    retry_match = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
                if retry_match:
                    wait_time = float(retry_match.group(1)) + 1.0
                else:
                    wait_time = 2.0 * (attempt + 1)

                if wait_time <= 10.0:
                    print(f"  [API Backoff] Rate limit reached. Waiting {wait_time:.1f}s before attempt {attempt + 2}/3...", flush=True)
                    time.sleep(wait_time)
                    continue
            return f"[ERROR] Gemini API request failed: {type(e).__name__} - {e}"


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
        print(f"{i}. {res['document_name']} — Page {res['page_number']} — {res['chunk_id']}")
    print("==================================================\n")

    return {
        "query": clean_query,
        "retrieved_results": retrieved_results,
        "answer": llm_answer
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    pdf_file_path = project_root / "documents" / "medical_information.pdf"

    print("=" * 70)
    print("Healthcare RAG (MediCore) - Phase 5: Google Gemini LLM Integration")
    print("=" * 70)

    # 1. Phase 1: Ingestion
    pages_data = extract_text_from_pdf(pdf_file_path)
    if not pages_data:
        print("[ERROR] No pages were extracted. Exiting.")
        return

    # 2. Phase 2: Chunking & Metadata
    document_name = pdf_file_path.name
    chunks = create_chunks_with_metadata(
        pages_data=pages_data,
        document_name=document_name,
        chunk_size=500,
        overlap=50
    )
    if not chunks:
        print("[ERROR] No chunks were created. Exiting.")
        return

    # 3. Phase 3: Embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = load_embedding_model(model_name)
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks, model)

    # 4. Phase 4: FAISS Vector Indexing
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

    # 5. Phase 5 Test Suite (3 required verification queries)
    test_queries = [
        "What are the criteria for Stage 2 hypertension?",
        "What are the glycemic targets for Type 2 diabetes?",
        "What is the treatment for appendicitis?"
    ]

    for q in test_queries:
        answer_query(q, model, faiss_index, chunks_with_embeddings, top_k=3)

    print("Phase 5 pipeline execution completed.")


if __name__ == "__main__":
    main()
