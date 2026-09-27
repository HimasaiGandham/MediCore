"""
Healthcare RAG (MediCore) - Phase 7: Streamlit Web Interface
------------------------------------------------------------
A clean, modern healthcare knowledge assistant interface built with Streamlit.
Reuses the existing MediCore RAG pipeline (Phases 1-5) without code duplication.

Features:
- Cached resource initialization (Embedding Model, FAISS Vector Index, Chunks).
- Seamless query retrieval with Cosine Similarity scoring.
- Grounded answer generation via Google Gemini 3.8 Flash.
- Transparent source attribution (Document, Page, Chunk ID, Similarity).
- Graceful handling of quota limits, missing keys, and unsupported questions.
- Strict clinical safety and disclaimer notices.
"""

import os
import sys
from pathlib import Path

# Ensure memory-safety environment variables are active before importing scientific libs
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set HuggingFace cache directory
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Reuse existing MediCore RAG pipeline functions from src/main.py
from src.main import (
    extract_text_from_pdf,
    create_chunks_with_metadata,
    load_embedding_model,
    generate_embeddings_for_chunks,
    build_faiss_index,
    search_faiss,
    generate_answer,
)


# =====================================================================
# Streamlit Page Configuration & Modern Clinical Styling
# =====================================================================

st.set_page_config(
    page_title="MediCore — Healthcare Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional, clean medical-research theme
st.markdown(
    """
    <style>
        /* Base typography & colors */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.1rem;
        }
        .main-subtitle {
            font-size: 1.15rem;
            color: #0284c7;
            font-weight: 500;
            margin-bottom: 0.4rem;
        }
        .main-info {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.4rem;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }
        .answer-box {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #0284c7;
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
            font-size: 1.02rem;
            line-height: 1.6;
            color: #1e293b;
        }
        .source-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .source-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            border-color: #cbd5e1;
        }
        .source-title {
            font-weight: 600;
            color: #0f172a;
            font-size: 0.98rem;
            margin-bottom: 0.3rem;
        }
        .source-meta {
            font-size: 0.88rem;
            color: #475569;
            margin-bottom: 0.4rem;
        }
        .source-score {
            display: inline-block;
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .pipeline-badge {
            background-color: #f1f5f9;
            color: #334155;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 500;
            margin-bottom: 0.4rem;
            border: 1px solid #e2e8f0;
        }
        .disclaimer-card {
            background-color: #fffbeb;
            border: 1px solid #fef3c7;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            color: #92400e;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Resource Initialization with Streamlit Caching
# =====================================================================

@st.cache_resource(show_spinner="Loading MediCore pipeline & indexing reference documents...")
def load_rag_pipeline():
    """
    Initializes and caches expensive pipeline resources:
    1. Loads PDF text
    2. Chunks text with metadata
    3. Loads sentence-transformer embedding model
    4. Computes 384-d chunk embeddings
    5. Builds FAISS FlatIP Cosine Similarity index

    Using @st.cache_resource guarantees the model and vector database
    are initialized once and shared across user queries.
    """
    pdf_path = PROJECT_ROOT / "documents" / "medical_information.pdf"
    if not pdf_path.exists():
        st.error(f"Medical reference PDF not found at: {pdf_path}")
        return None, None, [], "medical_information.pdf", 0

    # Phase 1: Ingestion
    pages_data = extract_text_from_pdf(pdf_path)

    # Phase 2: Chunking & Metadata
    chunks = create_chunks_with_metadata(
        pages_data=pages_data,
        document_name=pdf_path.name,
        chunk_size=500,
        overlap=50,
    )

    # Phase 3: Sentence Transformer Embeddings
    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks, model)

    # Phase 4: FAISS Vector Search Index
    faiss_index = build_faiss_index(chunks_with_embeddings)

    return model, faiss_index, chunks_with_embeddings, pdf_path.name, len(chunks_with_embeddings)


# =====================================================================
# Sidebar Architecture & Project Metadata
# =====================================================================

def render_sidebar(document_name: str, total_chunks: int):
    with st.sidebar:
        st.markdown("## 🩺 MediCore")
        st.caption("AI-Powered Clinical Knowledge Assistant")
        st.divider()

        st.markdown("### 📋 System Pipeline")
        st.markdown(
            """
            <div class="pipeline-badge">1. Document Extraction (PyMuPDF)</div>
            <div class="pipeline-badge">2. Chunking (~500 chars, 50 overlap)</div>
            <div class="pipeline-badge">3. Embeddings (all-MiniLM-L6-v2)</div>
            <div class="pipeline-badge">4. FAISS Retrieval (Cosine Similarity)</div>
            <div class="pipeline-badge">5. Grounded Generation (Gemini 3.8)</div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("### 📑 Reference Document")
        st.write(f"📄 **{document_name}**")
        st.write(f"📦 **{total_chunks} chunks indexed**")
        st.divider()

        st.markdown("### ⚙️ System Configuration")
        st.write("**Model:** `gemini-3.8-flash`")
        st.write("**Embedding Dim:** 384 dimensions")
        st.write("**Search Metric:** Cosine Similarity")

        api_configured = bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_API_KEY").strip() not in {"", "YOUR_GEMINI_API_KEY_HERE"})
        if api_configured:
            st.success("API Status: Configured (.env)")
        else:
            st.warning("API Status: Not Configured")

        st.divider()
        st.markdown("### 💡 Quick Test Questions")
        st.caption("Click to copy into the question box:")
        sample_questions = [
            "What are the criteria for Stage 2 hypertension?",
            "What are the glycemic targets for Type 2 diabetes?",
            "What are the qSOFA criteria for identifying sepsis?",
            "What is the treatment for appendicitis?",
        ]
        for q in sample_questions:
            if st.button(q, key=f"btn_{q[:15]}", use_container_width=True):
                st.session_state["query_input"] = q


# =====================================================================
# Main Application Interface
# =====================================================================

def main():
    # Initialize cached RAG pipeline
    model, faiss_index, chunks, doc_name, chunk_count = load_rag_pipeline()

    # Render Sidebar
    render_sidebar(doc_name, chunk_count)

    # Header section
    st.markdown('<div class="main-header">MediCore</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Healthcare Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-info">Answers are generated from the available reference material.</div>', unsafe_allow_html=True)

    # Initialize session state for input
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    # Question Input Form
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_area(
            label="Ask a medical question...",
            value=st.session_state["query_input"],
            placeholder="e.g., What are the criteria for Stage 2 hypertension?",
            height=100,
            help="Enter a clinical or medical query grounded in the reference document.",
        )
        submit_button = st.form_submit_button("Ask MediCore", type="primary", use_container_width=False)

    # Process Query Submission
    if submit_button:
        clean_query = user_query.strip() if user_query else ""

        # 1. Validation: Empty Query
        if not clean_query:
            st.warning("Please enter a question.")
            return

        # 2. Pipeline Execution with Spinner
        with st.spinner("Searching medical reference material..."):
            try:
                # FAISS Semantic Retrieval
                retrieved_chunks = search_faiss(
                    query=clean_query,
                    model=model,
                    index=faiss_index,
                    chunks=chunks,
                    top_k=3,
                )
            except Exception as e:
                st.error(f"Retrieval error occurred: {e}")
                return

            if not retrieved_chunks:
                st.error("No relevant document sections found in the vector index.")
                return

            # Gemini Grounded Generation
            api_key = os.getenv("LLM_API_KEY")
            if not api_key or api_key.strip() in {"", "YOUR_GEMINI_API_KEY_HERE"}:
                answer_text = "Gemini API key is not configured. Please add LLM_API_KEY to .env."
            else:
                try:
                    answer_text = generate_answer(clean_query, retrieved_chunks)
                except Exception as e:
                    answer_text = f"[ERROR] Unexpected generation failure: {e}"

        # 3. Display ANSWER Section
        st.markdown('<div class="section-header">ANSWER</div>', unsafe_allow_html=True)

        # Handle Quota / Error States Gracefully
        if "Daily quota reached" in answer_text or "RESOURCE_EXHAUSTED" in answer_text:
            st.warning(
                "⚠️ **Gemini API Quota Reached:** The daily free-tier request quota (20 requests/day) has been reached. "
                "The semantic retrieval pipeline is fully operational (see verified source citations below). "
                "Natural-language generation will automatically resume once the quota replenishes."
            )
            st.markdown(
                '<div class="answer-box"><em>[Informational notice: LLM answer temporarily paused due to free-tier quota. Retrieved source references below provide verified factual context for this question.]</em></div>',
                unsafe_allow_html=True,
            )
        elif "Gemini API key not configured" in answer_text or "Gemini API key is not configured" in answer_text:
            st.warning("Gemini API key is not configured.")
            st.markdown(
                f'<div class="answer-box">{answer_text}</div>',
                unsafe_allow_html=True,
            )
        elif answer_text.startswith("[ERROR]"):
            st.error(f"Generation Notice: {answer_text}")
        else:
            # Check for Grounded Refusal Notice
            if "not contain sufficient information" in answer_text.lower():
                st.info("ℹ️ **Reference Boundary Notice:** The available reference material does not contain information on this topic.")

            st.markdown(
                f'<div class="answer-box">{answer_text}</div>',
                unsafe_allow_html=True,
            )

        # 4. Display SOURCES Section
        st.markdown('<div class="section-header">SOURCES</div>', unsafe_allow_html=True)

        cols = st.columns(len(retrieved_chunks))
        for idx, (col, chunk_data) in enumerate(zip(cols, retrieved_chunks), start=1):
            with col:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">Source {idx}: {chunk_data['document_name']}</div>
                        <div class="source-meta">
                            <strong>Page:</strong> {chunk_data['page_number']}<br>
                            <strong>Chunk ID:</strong> <code>{chunk_data['chunk_id']}</code>
                        </div>
                        <div class="source-score">Similarity: {chunk_data['similarity_score']:.4f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"View Chunk Text ({chunk_data['chunk_id']})"):
                    st.caption(chunk_data["text"])

    # Clinical Disclaimer Card
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Research & Informational Disclaimer:</strong>
            MediCore is an educational and research Retrieval-Augmented Generation (RAG) prototype.
            It does not provide clinical diagnoses, medical advice, prescriptions, or personalized treatment plans.
            All responses are derived strictly from local reference documentation.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
