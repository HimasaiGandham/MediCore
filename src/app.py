"""
Healthcare RAG (MediCore) - Phase 8: Streamlit Chatbot UI
---------------------------------------------------------
A clean, modern medical knowledge assistant chat interface built with Streamlit.
Connected directly to the multi-source authoritative medical knowledge base
(Level 1 WHO, CDC, NHS, ICMR guidelines and baseline clinical references).

Features:
- Multi-turn session chat history (st.chat_message & st.chat_input)
- Cached resource initialization (Embedding Model, FAISS Vector Index, Authoritative Chunks)
- Multi-source semantic query retrieval with FAISS and Cosine Similarity scoring
- Grounded answer generation via Google Gemini
- Distinct, structured source display (Source Organization, Document ID, Title, Section, URL, Similarity Score)
- Strict grounded refusal for unsupported medical questions
- Optional clinical domain filter
- Non-blocking handling for API quota limits and missing keys
- Prominent clinical safety and educational disclaimers
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

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

# Import RAG pipeline & knowledge base components
from src.main import (
    load_embedding_model,
    generate_answer,
    extract_text_from_pdf,
    create_chunks_with_metadata,
    generate_embeddings_for_chunks,
    build_faiss_index,
)
from src.knowledge_base import (
    build_or_load_knowledge_base,
    search_knowledge_base,
    get_knowledge_base_statistics,
    load_catalog,
)


# =====================================================================
# Streamlit Page Configuration & Modern Clean Clinical Styling
# =====================================================================

st.set_page_config(
    page_title="MediCore — Healthcare Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, professional medical research theme
st.markdown(
    """
    <style>
        /* Base header styles */
        .medicore-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .medicore-subtitle {
            font-size: 1.05rem;
            color: #0284c7;
            font-weight: 500;
            margin-bottom: 0.3rem;
        }
        .medicore-description {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 1.2rem;
            line-height: 1.5;
        }
        /* Source display styling */
        .sources-container {
            margin-top: 0.9rem;
            padding-top: 0.7rem;
            border-top: 1px dashed #cbd5e1;
        }
        .sources-title {
            font-size: 0.88rem;
            font-weight: 700;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.6rem;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .source-pill-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #0284c7;
            border-radius: 6px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.5rem;
            font-size: 0.86rem;
            height: 100%;
        }
        .source-meta-item {
            color: #1e293b;
            margin-bottom: 0.2rem;
            line-height: 1.35;
        }
        .source-score-badge {
            display: inline-block;
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }
        .org-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .org-badge-nhs { background-color: #005eb8; color: #ffffff; }
        .org-badge-who { background-color: #008dc9; color: #ffffff; }
        .org-badge-cdc { background-color: #0b4f8a; color: #ffffff; }
        .org-badge-icmr { background-color: #d97706; color: #ffffff; }
        .org-badge-ref { background-color: #334155; color: #ffffff; }

        .refusal-badge {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.92rem;
            margin-bottom: 0.6rem;
        }
        .quota-badge {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            color: #92400e;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.90rem;
            margin-bottom: 0.6rem;
        }
        .sidebar-badge {
            background-color: #f1f5f9;
            color: #334155;
            padding: 0.3rem 0.6rem;
            border-radius: 5px;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
            border: 1px solid #e2e8f0;
        }
        .disclaimer-banner {
            background-color: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 0.6rem 0.9rem;
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .url-link {
            display: inline-block;
            font-size: 0.78rem;
            color: #0284c7;
            text-decoration: none;
            margin-top: 0.25rem;
            font-weight: 500;
        }
        .url-link:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Resource Initialization with Streamlit Caching
# =====================================================================

@st.cache_resource(show_spinner="Initializing MediCore Authoritative Knowledge Base & Vector Store...")
def load_rag_pipeline():
    """
    Initializes and caches pipeline resources:
    1. Loads sentence-transformer embedding model
    2. Builds or loads persistent FAISS index & authoritative medical chunks
    3. Loads central catalog & aggregate statistics across official medical authorities
    """
    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    try:
        faiss_index, chunks = build_or_load_knowledge_base(model)
        stats = get_knowledge_base_statistics(chunks)
        catalog = load_catalog()
    except Exception as e:
        # Graceful fallback to baseline PDF if multi-source ingestion fails
        pdf_path = PROJECT_ROOT / "documents" / "medical_information.pdf"
        pages_data = extract_text_from_pdf(pdf_path)
        chunks = create_chunks_with_metadata(pages_data, pdf_path.name, chunk_size=500, overlap=50)
        chunks = generate_embeddings_for_chunks(chunks, model)
        faiss_index = build_faiss_index(chunks)
        stats = {
            "total_chunks": len(chunks),
            "total_documents": 1,
            "organizations": {"Department of Internal Medicine": len(chunks)},
            "categories": {"Internal Medicine & Emergency Care": len(chunks)},
        }
        catalog = {"documents": []}

    return model, faiss_index, chunks, stats, catalog


def get_org_badge_class(org_name: str) -> tuple[str, str]:
    """Returns CSS class and display name for source organization badge."""
    org_upper = org_name.upper()
    if "NHS" in org_upper:
        return "org-badge org-badge-nhs", "🏛️ NHS (UK)"
    elif "WHO" in org_upper or "WORLD HEALTH" in org_upper:
        return "org-badge org-badge-who", "🌐 WHO (Global)"
    elif "CDC" in org_upper:
        return "org-badge org-badge-cdc", "🏛️ CDC (US)"
    elif "ICMR" in org_upper:
        return "org-badge org-badge-icmr", "🏛️ ICMR (India)"
    else:
        return "org-badge org-badge-ref", "🏥 Clinical Reference"


# =====================================================================
# Sidebar Architecture & Project Controls
# =====================================================================

def render_sidebar(stats: dict, catalog: dict):
    with st.sidebar:
        st.markdown("### 🩺 MediCore Assistant")
        st.caption("Authoritative Grounded Healthcare RAG")

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()

        # Knowledge Base Overview
        total_docs = stats.get("total_documents", 10)
        total_chunks = stats.get("total_chunks", 82)
        st.markdown("#### 📑 Authoritative Corpus")
        st.markdown(f"🏛️ **{total_docs} Verified Medical Documents**")
        st.markdown(f"📦 **{total_chunks} Total Chunks Indexed**")

        with st.expander("🏛️ Medical Authorities Represented", expanded=False):
            st.markdown("- **WHO**: Hypertension, Sepsis, Anaphylaxis")
            st.markdown("- **CDC**: Type 2 Diabetes, Tuberculosis, Asthma")
            st.markdown("- **NHS UK**: Appendicitis, ACS, Acute Stroke")
            st.markdown("- **ICMR**: National Dengue Guidelines")
            st.markdown("- **Clinical Guidelines**: Internal Medicine Practice")

        # Category Filter Control
        categories = ["All Clinical Domains"] + sorted(list(stats.get("categories", {}).keys()))
        selected_cat = st.selectbox(
            "Filter by Clinical Domain:",
            options=categories,
            index=0,
            help="Optionally restrict retrieval to a specific medical specialty.",
        )
        st.session_state["active_domain_filter"] = None if selected_cat == "All Clinical Domains" else selected_cat

        st.divider()

        # Pipeline Specifications
        st.markdown("#### ⚙️ Pipeline Specifications")
        st.markdown('<div class="sidebar-badge"><b>Embeddings:</b> all-MiniLM-L6-v2 (384-d)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-badge"><b>Vector Store:</b> FAISS IndexFlatIP (Cosine)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-badge"><b>Provenance:</b> Level 1 & Level 3 Guidelines</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-badge"><b>LLM:</b> {os.getenv("GEMINI_MODEL", "gemini-3.8-flash")}</div>', unsafe_allow_html=True)

        api_configured = bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_API_KEY").strip() not in {"", "YOUR_GEMINI_API_KEY_HERE"})
        if api_configured:
            st.success("API Status: Configured (.env)")
        else:
            st.warning("API Status: Missing Key (.env)")

        st.divider()

        # Sample Benchmark Queries
        st.markdown("#### 💡 Clinical Benchmark Queries")
        st.caption("Click any query to evaluate grounded retrieval:")
        sample_questions = [
            ("What is the recommended treatment for acute appendicitis?", "NHS UK"),
            ("What are the warning signs of severe dengue and why are NSAIDs contraindicated?", "ICMR"),
            ("What is the standard 4-drug intensive regimen (HRZE) for active tuberculosis?", "CDC"),
            ("What are the FAST criteria and thrombolytic time window for acute stroke?", "NHS UK"),
            ("What are the diagnostic criteria for Stage 2 hypertension?", "WHO / Reference"),
            ("What are the contraindications for administering Nitroglycerin in acute coronary syndrome?", "NHS UK"),
            ("What are the surgical steps and prosthetic mesh placement techniques for repairing an inguinal hernia?", "Refusal Test"),
        ]
        for q, tag in sample_questions:
            btn_label = f"[{tag}] {q[:42]}..."
            if st.button(btn_label, key=f"side_btn_{hash(q)}", use_container_width=True):
                st.session_state["pending_query"] = q
                st.rerun()

        st.divider()
        st.caption("MediCore is an educational college project demonstrating Retrieval-Augmented Generation.")


# =====================================================================
# RAG Response Processing
# =====================================================================

def process_user_query(query: str, model, faiss_index, chunks, category_filter: Optional[str] = None):
    """
    Executes the End-to-End RAG workflow for a user chat query:
    1. Multi-source semantic retrieval via FAISS (Top-3 Chunks)
    2. Google Gemini Grounded Generation with full provenance
    3. Source Citation and Metadata Extraction
    """
    clean_query = query.strip()
    if not clean_query:
        return

    # 1. Semantic Retrieval across Authoritative Knowledge Base
    retrieved_chunks = search_knowledge_base(
        query=clean_query,
        model=model,
        index=faiss_index,
        chunks=chunks,
        top_k=3,
        category_filter=category_filter,
    )

    # 2. Extract structured source details with complete provenance
    sources = []
    for c in retrieved_chunks:
        badge_class, badge_name = get_org_badge_class(c.get("source_organization", ""))
        sources.append({
            "doc_id": c.get("document_id", c.get("chunk_id")),
            "document_name": c.get("document_name", ""),
            "document_title": c.get("document_title", c.get("document_name", "Clinical Guideline")),
            "source_organization": c.get("source_organization", "Medical Authority"),
            "organization_level": c.get("organization_level", "Level 1 — Official Medical Authority"),
            "source_url": c.get("source_url", ""),
            "topic_category": c.get("topic_category", "General Medicine"),
            "section_title": c.get("section_title", f"Page {c.get('page_number', 1)}"),
            "page_number": c.get("page_number", 1),
            "chunk_id": c.get("chunk_id", ""),
            "similarity_score": round(c.get("similarity_score", 0.0), 4),
            "badge_class": badge_class,
            "badge_name": badge_name,
            "text": c.get("text", ""),
        })

    # 3. Gemini Answer Generation
    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key.strip() in {"", "YOUR_GEMINI_API_KEY_HERE"}:
        answer_text = "Gemini API key not configured. Add LLM_API_KEY to .env."
    else:
        answer_text = generate_answer(clean_query, retrieved_chunks)

    # 4. Determine state flags
    is_refusal = (
        "not contain sufficient information" in answer_text.lower() or
        "insufficient information" in answer_text.lower() or
        "insufficient reference material" in answer_text.lower()
    )
    is_quota_error = "daily quota reached" in answer_text.lower() or "resource_exhausted" in answer_text.lower()

    # Append to session history
    st.session_state["messages"].append({
        "role": "user",
        "content": clean_query,
    })

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
        "is_refusal": is_refusal,
        "is_quota_error": is_quota_error,
    })


def render_sources(sources: list[dict]):
    """Renders formatted, visually distinct source citations with organization provenance."""
    if not sources:
        return

    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
    st.markdown('<div class="sources-title">📚 Supporting Authoritative Medical Sources</div>', unsafe_allow_html=True)

    cols = st.columns(len(sources))
    for idx, (col, src) in enumerate(zip(cols, sources), start=1):
        with col:
            url_html = ""
            if src.get("source_url") and src["source_url"].startswith("http"):
                url_html = f'<div class="source-meta-item"><a class="url-link" href="{src["source_url"]}" target="_blank">🔗 Official Source Document</a></div>'

            st.markdown(
                f"""
                <div class="source-pill-card">
                    <span class="{src['badge_class']}">{src['badge_name']}</span>
                    <div class="source-meta-item"><b>Guideline:</b> {src['document_title']}</div>
                    <div class="source-meta-item"><b>Doc ID:</b> <code>{src['doc_id']}</code></div>
                    <div class="source-meta-item"><b>Section:</b> {src['section_title']} (P. {src['page_number']})</div>
                    <div class="source-meta-item"><b>Domain:</b> {src['topic_category']}</div>
                    {url_html}
                    <span class="source-score-badge">Cosine Sim: {src['similarity_score']:.4f}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"View Reference Excerpt ({src['chunk_id']})"):
                st.caption(src["text"])

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# Main Application Flow
# =====================================================================

def main():
    # Load cached RAG pipeline
    model, faiss_index, chunks, stats, catalog = load_rag_pipeline()

    # Render Sidebar
    render_sidebar(stats, catalog)

    # Main Header
    st.markdown(
        """
        <div class="medicore-header">
            <span>🩺 MediCore</span>
        </div>
        <div class="medicore-subtitle">
            Authoritative Medical Knowledge Assistant
        </div>
        <div class="medicore-description">
            Ask questions grounded strictly in official clinical reference material from <b>WHO</b>, <b>CDC</b>, <b>NHS UK</b>, <b>ICMR</b>, and established medical guidelines.
        </div>
        <div class="disclaimer-banner">
            <b>⚠️ Clinical & Safety Disclaimer:</b> MediCore is an informational Retrieval-Augmented Generation prototype designed for education and research.
            It does not diagnose patients, prescribe medication, or replace professional clinical judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am **MediCore**, your medical reference assistant. "
                    "I am connected to an authoritative medical knowledge base comprising official guidance from "
                    "the **World Health Organization (WHO)**, **Centers for Disease Control and Prevention (CDC)**, "
                    "**National Health Service (NHS UK)**, and **Indian Council of Medical Research (ICMR)**.\n\n"
                    "Ask me questions regarding conditions such as hypertension, diabetes, acute coronary syndrome, appendicitis, "
                    "dengue fever, sepsis, asthma, or stroke, and I will answer strictly based on our verified clinical sources."
                ),
                "sources": [],
                "is_refusal": False,
                "is_quota_error": False,
            }
        ]

    # Render existing conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤"):
            if msg.get("is_quota_error"):
                st.markdown(
                    """
                    <div class="quota-badge">
                        ⚠️ <b>API Quota Notice:</b> The free-tier request quota (20 requests/day) has been reached.
                        The underlying FAISS semantic retrieval pipeline is fully operational — see verified reference sources below.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif msg.get("is_refusal"):
                st.markdown(
                    """
                    <div class="refusal-badge">
                        🛡️ <b>Reference Boundary Refusal:</b> The available reference material does not contain sufficient information to answer this question.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write(msg["content"])

            if msg.get("sources"):
                render_sources(msg["sources"])

    # Check for pending query from sidebar buttons
    user_input = None
    if "pending_query" in st.session_state and st.session_state["pending_query"]:
        user_input = st.session_state["pending_query"]
        st.session_state["pending_query"] = None

    # Chat Input widget
    chat_prompt = st.chat_input("Ask a health-related question from the medical reference material...")
    if chat_prompt:
        user_input = chat_prompt

    # Process query if provided
    if user_input:
        active_filter = st.session_state.get("active_domain_filter")
        with st.chat_message("user", avatar="👤"):
            st.write(user_input)

        with st.chat_message("assistant", avatar="🧑‍⚕️"):
            with st.spinner("Retrieving verified reference chunks and generating grounded answer..."):
                process_user_query(user_input, model, faiss_index, chunks, category_filter=active_filter)
                st.rerun()


if __name__ == "__main__":
    main()
