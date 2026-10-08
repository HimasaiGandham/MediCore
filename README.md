# 🩺 MediCore – Healthcare Information Assistant

> **Making medical information easier to find.**

MediCore is a healthcare-focused Retrieval-Augmented Generation (RAG) assistant that helps users find reliable, grounded information from **authoritative medical reference documents** without having to manually search through long PDFs.

Instead of reading through hundreds of pages, users can simply ask a question, and MediCore retrieves the verified clinical evidence and provides an answer grounded strictly in official guidelines along with complete **provenance, source organization, and section details**.

---

## 🚀 Project Overview

The core architecture behind MediCore is simple:

**Ask a question → Retrieve verified reference chunks → Generate grounded answer → Show source & provenance**

| User Question | Issuing Authority | MediCore Grounded Response |
|---|---|---|
| What are the criteria for Stage 2 hypertension? | 🌐 WHO / Clinical Reference | Systolic BP ≥ 140 mmHg or Diastolic BP ≥ 90 mmHg |
| What is the treatment for appendicitis? | 🇬🇧 NHS UK | Appendectomy (keyhole or open) and intravenous antibiotics |
| What are the warning signs of severe dengue? | 🇮🇳 ICMR | Severe abdominal pain, persistent vomiting, fluid accumulation, mucosal bleeding; NSAIDs contraindicated |
| What is the 4-drug regimen for tuberculosis? | 🇺🇸 CDC | Intensive 2-month HRZE regimen (Isoniazid, Rifampin, Pyrazinamide, Ethambutol) |
| What are the surgical steps for inguinal hernia repair? | *Unindexed procedure* | 🛡️ **Safe Refusal:** *"The available reference material does not contain sufficient information to answer this question."* |

If the required information is not available in the authoritative reference material, MediCore strictly refuses to answer rather than inventing ungrounded medical facts.

---

## 📁 Project Structure & Respective Folder Functions

```text
MediCore/
├── documents/                      # Medical reference corpus, structured clinical guidance & vector index cache
│   ├── catalog.json                # Central registry of clinical reference documents with provenance metadata
│   ├── medical_information.pdf     # Baseline institutional practice guideline PDF (Internal Medicine & Emergency)
│   ├── .index/                     # Persistent FAISS cache enabling <0.1s instant application startup
│   │   ├── faiss_index.bin         # 384-dimensional FAISS IndexFlatIP binary vector index
│   │   └── chunks.pkl              # Pickled chunk registry containing text and provenance metadata
│   └── authoritative/              # Structured clinical guidelines from Level 1 official health organizations
│       ├── nhs_appendicitis.json   # NHS UK: Appendicitis symptoms, appendectomy surgery, and antibiotic therapy
│       ├── who_hypertension.json   # WHO: Hypertension diagnostic staging cutoffs and first-line drug classes
│       ├── cdc_diabetes.json       # CDC: Type 2 diabetes glycemic targets, metformin dosing, and complications
│       ├── nhs_acute_coronary_syndrome.json # NHS UK: ACS pre-hospital interventions and GTN contraindications
│       ├── who_sepsis_anaphylaxis.json      # WHO: qSOFA critical sepsis scoring and anaphylaxis epinephrine protocol
│       ├── icmr_dengue_guidelines.json      # ICMR: Dengue phases, warning signs, fluid resuscitation, NSAID warnings
│       ├── cdc_tuberculosis.json   # CDC: Latent vs active TB distinctions, intensive 4-drug HRZE regimen
│       ├── nhs_stroke_emergency.json        # NHS UK: FAST protocol, alteplase intravenous thrombolysis window
│       └── cdc_respiratory_asthma.json      # CDC: Reliever (SABA) vs controller (ICS) inhalers, acute red flags
│
├── src/                            # Core application source code & RAG pipeline modules
│   ├── main.py                     # End-to-End RAG pipeline CLI demo with multi-source retrieval & Gemini grounding
│   ├── knowledge_base.py           # Ingestion engine, metadata tagging, FAISS indexing, caching & filtered search
│   ├── app.py                      # Phase 8 Streamlit web application with multi-turn chat & organization badges
│   └── evaluation.py               # Deterministic RAG evaluation suite supporting authoritative & baseline modes
│
├── evaluation_results/             # Evaluation benchmark logs & audit reports
│   ├── results_authoritative.json  # Machine-readable evaluation metrics for authoritative multi-source benchmark
│   ├── evaluation_report_authoritative.txt # Human-readable audit report for authoritative benchmark (100% precision)
│   ├── results.json                # Machine-readable evaluation metrics for baseline benchmark
│   └── evaluation_report.txt       # Human-readable audit report for baseline benchmark
│
├── create_sample_pdf.py            # Generator utility for the baseline sample medical reference PDF
├── requirements.txt                # Python dependencies (PyMuPDF, sentence-transformers, faiss, LLM SDK, streamlit)
├── .env.example                    # Configuration template for API keys and model parameters
├── .gitignore                      # Security rules ensuring virtual environments (.venv) and .env are never committed
└── README.md                       # Comprehensive project documentation, folder functions, and setup guide
```

---

## 🏛️ Medical Source Authority Hierarchy

MediCore strictly avoids unvetted internet articles, user-generated health forums, and AI prior training assumptions. All retrieval and answer generation is grounded exclusively in vetted, authoritative documents:

| Level | Organization Type | Primary Authorities in MediCore | Topic Coverage |
| :--- | :--- | :--- | :--- |
| **Level 1** | **Government & International Health Bodies** | 🌐 **WHO** (World Health Organization)<br>🇺🇸 **CDC** (Centers for Disease Control)<br>🇬🇧 **NHS UK** (National Health Service)<br>🇮🇳 **ICMR** (Indian Council of Medical Research) | Hypertension guidelines, Type 2 diabetes targets, Sepsis qSOFA criteria, Tuberculosis HRZE therapy, Acute stroke FAST protocol, Dengue management |
| **Level 3** | **Clinical Practice Guidelines** | 🏥 **Clinical Emergency & Internal Medicine Reference** | Acute coronary syndrome, Anaphylaxis resuscitation, Appendectomy protocols, Inhaler therapy |

---

## ✨ Features

- 📄 **Multi-Source Clinical Ingestion** – Dual ingestion for structured JSON clinical guidelines and multi-page reference PDFs.
- 🔎 **Semantic FAISS Vector Retrieval** – Exact cosine similarity search matching clinical concepts geometrically (384-dimensional dense vectors).
- ⚡ **Instant Persistent Index Caching** – Pre-computed vector store (`documents/.index/`) loads the complete 82-chunk knowledge base in **< 0.1 seconds**.
- 🧠 **Grounded AI Generation** – Answers generated via Google Gemini strictly derived from retrieved evidence.
- 📑 **Comprehensive Provenance Citations** – Every answer displays source organization badges (NHS UK, WHO, CDC, ICMR), document ID, section name, page number, official URL link, and similarity score.
- 🚫 **Strict Grounded Refusal** – Refuses out-of-domain and unindexed clinical questions to eliminate hallucinations.
- 💬 **Interactive Streamlit Web UI** – Conversational interface with multi-turn chat history, clinical domain filter dropdown, and benchmark query buttons.
- 📊 **Dual Evaluation Benchmark Suite** – Automated evaluation measuring Top-1/Top-3 retrieval precision, source traceability, and safe refusal.

---

## 🛠️ Technologies Used

- **Python** – Core programming language
- **PyMuPDF (`pymupdf`)** – Document ingestion & page text extraction
- **Sentence Transformers (`all-MiniLM-L6-v2`)** – 384-dimensional dense text embeddings
- **FAISS (`IndexFlatIP`)** – High-performance vector similarity search
- **Google Gemini (`google-genai`)** – Grounded clinical answer generation
- **Streamlit** – Clean, modern clinical research web interface

---

## 📊 Evaluation Results

MediCore was benchmarked across multi-source clinical queries and unsupported procedures:

| Metric | Benchmark Result | Notes |
| :--- | :---: | :--- |
| **Top-1 Retrieval Accuracy** | **100.0%** (11/11) | Top retrieved chunk matches target guideline |
| **Top-3 Retrieval Accuracy** | **100.0%** (11/11) | Target guideline present in top-3 candidates |
| **Source Traceability** | **100.0%** (11/11) | Full metadata attached to all retrieved chunks |
| **Safe Refusal on Unsupported Questions** | **100.0%** (2/2) | Accurately refuses unindexed medical procedures |

---

## 🚀 Setup & Running

### 1. Configure Environment
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
In `.env`:
```text
LLM_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.8-flash
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the CLI Pipeline Demo
```bash
python src/main.py
```
Or using the virtual environment:
```powershell
.\.venv\Scripts\python.exe src\main.py
```

### 4. Run the Evaluation Benchmark
```bash
# Authoritative Knowledge Base benchmark (default)
python src/evaluation.py --mode authoritative

# Baseline Single-PDF benchmark
python src/evaluation.py --mode baseline
```
Or with the project's virtual environment:
```powershell
.\.venv\Scripts\python.exe src\evaluation.py --mode authoritative
```

### 5. Launch the Streamlit Web Chatbot
```bash
streamlit run src/app.py
```
Or using the virtual environment:
```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
```
Open your browser at `http://localhost:8501` to use the interactive medical assistant.

---

## 📖 Learning Reference

The basic RAG concept was learned from **LangChain's RAG From Scratch** project.

We used it to understand the basic idea of retrieving relevant information before generating an answer, and then applied the concept to our own healthcare-focused project, **MediCore**.

---

## ⚠️ Disclaimer

MediCore is an **academic project** created for educational and informational purposes.

It is **not** intended to provide medical diagnosis, treatment, or professional medical advice.

For medical concerns, users should consult a qualified healthcare professional.

---

## ⭐ MediCore

**Ask your question. Find the information. Know the source.**
