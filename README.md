# Healthcare RAG (MediCore) — AI Medical Knowledge Assistant

A clean, beginner-friendly Retrieval-Augmented Generation (RAG) system specialized in clinical reference documents.

---

## Project Structure

```text
healthcare-rag/
├── documents/
│   └── medical_information.pdf   # Reference clinical guideline PDF
├── evaluation_results/           # Benchmark outputs (Phase 6)
│   ├── results.json              # Detailed test execution data
│   └── evaluation_report.txt     # Human-readable evaluation report
├── src/
│   ├── main.py                   # Complete End-to-End RAG pipeline (Phases 1-5)
│   ├── evaluation.py             # Phase 6 RAG Evaluation Suite
│   └── app.py                    # Phase 7 Streamlit Web Interface
├── create_sample_pdf.py          # Generator script for sample medical PDF
├── requirements.txt              # Project dependencies (PyMuPDF, sentence-transformers, faiss, LLM SDKs, streamlit)
├── .env.example                  # Template for API keys
├── .gitignore                    # Ensures .env and .venv are never committed
└── README.md                     # Documentation & setup instructions
```

---

## End-to-End Pipeline Architecture

```text
User Question
      ↓
[Query Embedding] ──(all-MiniLM-L6-v2)──> 384-d vector (L2 Normalized)
      ↓
[FAISS Index Search] ──(Cosine Similarity / FlatIP)──> Top-3 Chunks Retrieved
      ↓
[Context Construction] ──(System Prompt + Grounded Chunks + Metadata)
      ↓
[LLM Generation] ──(Gemini / OpenAI API / Grounded Engine)
      ↓
[Grounded Answer] ──(Strictly factual + Clinical Disclaimer)
      ↓
[Source Citations] ──(Document Name, Page Number, Chunk ID)
```

---

## Development Phases

### Phase 1: Document Ingestion (Completed)
- Reads multi-page medical PDFs page-by-page using PyMuPDF (`pymupdf`).
- Extracts plain text while tracking exact page numbers for reference integrity.

### Phase 2: Text Chunking & Metadata Tagging (Completed)
- **Sliding Window Chunking**: Splits extracted page text into chunks of approximately **500 characters** with **50 characters of overlap**.
- **Boundary Preservation**: Avoids cutting words in half by breaking at natural spaces/newlines.
- **Structured Metadata**: Attaches citation-ready metadata (`chunk_id`, `document_name`, `page_number`, `chunk_index`) to every chunk.

### Phase 3: Text Embeddings (Completed)
- **Semantic Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` loaded once in memory.
- **Dense Representation**: Transforms each ~500-character medical text chunk into a **384-dimensional numerical vector** (`float32`).
- **Semantic Proximity**: Encodes conceptual meaning so that related clinical concepts map to closely aligned vectors in geometric space.

### Phase 4: FAISS Vector Indexing & Semantic Retrieval (Completed)
- **Vector Search Engine**: Indexes embeddings with `faiss.IndexFlatIP` combined with **L2-normalized vectors** to compute exact **Cosine Similarity**.
- **Query Embedding**: Converts user queries into the same 384-dimensional vector space.
- **Top-K Retrieval**: FAISS returns the top $k=3$ nearest chunks with their associated metadata.

### Phase 5: LLM Answer Generation (Completed)
- **Grounded Prompt Construction**: Combines a strict system instruction, retrieved FAISS context, and the user query.
- **Gemini Model**: Integrates Google's official `google-genai` SDK using `gemini-3.8-flash`.
- **Grounded Generation**: The LLM is instructed to answer using **ONLY** the provided reference context and refuse to answer if the context is insufficient (tested with queries on topics outside the reference text, such as appendicitis).
- **Clinical Safety**: The assistant does not diagnose, prescribe, or provide personalized clinical treatment decisions.
- **Traceable Source Citations**: Every generated answer is paired with its verified document sources, page numbers, and chunk IDs.
- **API Key Security**: Loaded dynamically via `.env` using `python-dotenv`. Keys are never hardcoded, printed, or committed to version control.

---

### Phase 6: RAG Evaluation Suite (Completed)
- **Separate Testing Layer**: Standalone benchmark module in `src/evaluation.py` evaluating retrieval accuracy, answer grounding, source traceability, and safe refusal.
- **Curated Benchmark Dataset**: 10 structured test cases derived strictly from `documents/medical_information.pdf` covering factual hypertension, diabetes targets, emergency red flags, multi-chunk synthesis, and out-of-domain unindexed questions.
- **Retrieval Precision**: Evaluates Top-1 and Top-3 accuracy against ground-truth document pages and key medical concepts.
- **Answer Grounding & Hallucination Prevention**: Verifies answers contain expected facts, cite traceable sources, and explicitly refuse to invent information for unsupported queries.
- **Reporting & Persistence**: Automatically generates clean terminal summaries and saves machine-readable `evaluation_results/results.json` and human-readable `evaluation_results/evaluation_report.txt`.

### Phase 7: Streamlit Web Interface (Completed)
- **Clinical UI**: Interactive web interface in `src/app.py` built with Streamlit.
- **Cached Pipeline**: Uses `@st.cache_resource` for the embedding model, FAISS index, and chunks to guarantee fast sub-second query processing.
- **Grounded Presentation**: Displays Gemini 3.8 Flash answers with traceable source cards showing document name, page number, chunk ID, and Cosine Similarity scores.
- **Graceful Error Handling**: Non-blocking alerts for quota limits, missing API keys, empty queries, and out-of-domain medical questions.

---

## Setup & Running

### 1. Configure Environment (Optional for Cloud LLM)
Copy `.env.example` to `.env` and add your Gemini or OpenAI API key:
```bash
cp .env.example .env
```
In `.env`:
```text
LLM_API_KEY=your_actual_api_key_here
LLM_PROVIDER=gemini
```
*(If no API key is provided, MediCore seamlessly runs in local grounded mode to test queries and context matching).*

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Complete RAG Pipeline (Phases 1–5)
```bash
python src/main.py
```
This executes the pipeline demonstration:
1. Criteria for Stage 2 Hypertension (Grounded extraction with Page 1 citation)
2. Glycemic targets for Type 2 Diabetes (Grounded extraction with Page 2 citation)
3. Treatment for Appendicitis (Explicit refusal due to insufficient reference material)

### 4. Run Phase 6 RAG Evaluation Benchmark
```bash
python src/evaluation.py
```
Or using the virtual environment:
```bash
.\.venv\Scripts\python.exe src\evaluation.py
```
This automatically runs the 10-query benchmark suite, measures Top-1/Top-3 retrieval accuracy, tests grounding and refusal behavior, prints a detailed terminal report, and persists results to `evaluation_results/`.

### 5. Launch Phase 7 Streamlit Web Application
```bash
streamlit run src/app.py
```
Or using the virtual environment:
```bash
.\.venv\Scripts\streamlit.exe run src\app.py
```
Opens the interactive medical assistant in your browser at `http://localhost:8501`.

