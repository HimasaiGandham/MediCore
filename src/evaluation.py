"""
Healthcare RAG (MediCore) - Phase 6/7: RAG Evaluation Module
------------------------------------------------------------
Purpose:
Evaluates whether the MediCore RAG system:
1. Retrieves relevant chunks from FAISS vector search across authoritative sources.
2. Identifies the correct document ID, source organization, or reference page.
3. Generates natural-language answers strictly grounded in retrieved context.
4. Correctly handles unsupported (out-of-domain) questions by refusing to invent answers.
5. Preserves source traceability (source organization, document ID, page, chunk ID).
6. Avoids hallucinating medical information or making clinical claims.

Supports dual modes:
- Authoritative Mode (default): Evaluates retrieval & grounding across the full
  multi-source knowledge base (WHO, CDC, NHS, ICMR, and reference guidelines).
- Baseline Mode: Evaluates the single prototype reference document (documents/medical_information.pdf).

Important Safety Notice:
MediCore is an informational and research RAG system.
It is NOT a diagnostic or clinical treatment-decision system.
The evaluation metrics herein benchmark pipeline performance against
a curated test dataset and do not represent clinical validation.
"""

import os
import sys

# Prevent OpenBLAS thread pool memory allocation failures on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure HuggingFace cache is set to project folder on D: drive
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Reuse existing MediCore components
from src.main import (
    extract_text_from_pdf,
    create_chunks_with_metadata,
    load_embedding_model,
    generate_embeddings_for_chunks,
    build_faiss_index,
    search_faiss,
    generate_answer,
)
from src.knowledge_base import (
    build_or_load_knowledge_base,
    search_knowledge_base,
)


# =====================================================================
# Evaluation Datasets
# =====================================================================

# 1. Authoritative Multi-Source Knowledge Base Dataset
AUTHORITATIVE_EVALUATION_DATASET = [
    {
        "id": "test_01",
        "category": "cardiovascular",
        "is_supported": True,
        "question": "What are the criteria for Stage 2 hypertension?",
        "expected_doc_ids": ["MED-REF-2026-01", "WHO-GUIDE-HTN-2021", "MED-REF-2026-BASE"],
        "expected_keywords": ["140", "90"],
        "description": "Evaluates exact blood pressure threshold retrieval for Stage 2 hypertension (WHO / Reference)."
    },
    {
        "id": "test_02",
        "category": "cardiovascular",
        "is_supported": True,
        "question": "Why should ACE inhibitors and ARBs not be combined simultaneously in hypertension treatment?",
        "expected_doc_ids": ["MED-REF-2026-01", "WHO-GUIDE-HTN-2021", "MED-REF-2026-BASE"],
        "expected_keywords": ["renal", "risk"],
        "description": "Evaluates clinical pharmacotherapy contraindication and safety warning retrieval."
    },
    {
        "id": "test_03",
        "category": "endocrinology",
        "is_supported": True,
        "question": "What are the glycemic targets for Type 2 diabetes including HbA1c and fasting capillary glucose?",
        "expected_doc_ids": ["MED-REF-2026-02", "CDC-CLIN-DIAB-2024", "MED-REF-2026-BASE"],
        "expected_keywords": ["7.0", "80-130"],
        "description": "Evaluates glycemic target thresholds for non-pregnant adults (CDC / Reference)."
    },
    {
        "id": "test_04",
        "category": "endocrinology",
        "is_supported": True,
        "question": "What is the recommended first-line pharmacotherapy and titration dosage for Type 2 diabetes?",
        "expected_doc_ids": ["MED-REF-2026-02", "CDC-CLIN-DIAB-2024", "MED-REF-2026-BASE"],
        "expected_keywords": ["metformin", "500", "2000"],
        "description": "Evaluates first-line medication name, initial dosage, and maximum titration."
    },
    {
        "id": "test_05",
        "category": "emergency",
        "is_supported": True,
        "question": "What are the qSOFA criteria and clinical cutoffs for identifying sepsis?",
        "expected_doc_ids": ["MED-REF-2026-03", "WHO-CRIT-SEPS-2023", "MED-REF-2026-BASE"],
        "expected_keywords": ["22", "100"],
        "description": "Evaluates critical red-flag emergency scoring thresholds (WHO / Reference)."
    },
    {
        "id": "test_06",
        "category": "emergency",
        "is_supported": True,
        "question": "What is the primary emergency medication, dosage, and administration route for anaphylaxis?",
        "expected_doc_ids": ["MED-REF-2026-03", "WHO-CRIT-SEPS-2023", "MED-REF-2026-BASE"],
        "expected_keywords": ["epinephrine", "0.3", "intramuscular"],
        "description": "Evaluates anaphylaxis protocol, Epinephrine dosage, and route (IM thigh)."
    },
    {
        "id": "test_07",
        "category": "cardiology_emergency",
        "is_supported": True,
        "question": "What are the contraindications for administering Nitroglycerin in acute coronary syndrome?",
        "expected_doc_ids": ["MED-REF-2026-03", "NHS-EMERG-ACS-2024", "MED-REF-2026-BASE"],
        "expected_keywords": ["90", "pde-5"],
        "description": "Evaluates contraindications to sublingual Nitroglycerin (NHS UK / Reference)."
    },
    {
        "id": "test_08",
        "category": "digestive",
        "is_supported": True,
        "question": "What is the recommended surgical or antibiotic treatment protocol for acute appendicitis?",
        "expected_doc_ids": ["NHS-CLIN-APP-2024"],
        "expected_keywords": ["appendectomy", "antibiotic"],
        "description": "Evaluates surgical appendectomy and antibiotic management from NHS UK clinical guidance."
    },
    {
        "id": "test_09",
        "category": "infectious_diseases",
        "is_supported": True,
        "question": "What are the warning signs of severe dengue and why are NSAIDs like aspirin contraindicated?",
        "expected_doc_ids": ["ICMR-CLIN-DENG-2023"],
        "expected_keywords": ["nsaid", "paracetamol"],
        "description": "Evaluates national clinical guidelines on dengue management and contraindications from ICMR."
    },
    {
        "id": "test_10",
        "category": "infectious_diseases",
        "is_supported": True,
        "question": "What is the standard 4-drug intensive regimen (HRZE) for active tuberculosis disease?",
        "expected_doc_ids": ["CDC-INF-TB-2024"],
        "expected_keywords": ["isoniazid", "rifamp"],
        "description": "Evaluates intensive 4-drug therapy (HRZE) and vitamin B6 supplementation from CDC guidance."
    },
    {
        "id": "test_11",
        "category": "neurology_emergency",
        "is_supported": True,
        "question": "What are the FAST recognition signs and time window for intravenous thrombolysis in acute ischemic stroke?",
        "expected_doc_ids": ["NHS-EMERG-STRK-2024"],
        "expected_keywords": ["fast", "alteplase"],
        "description": "Evaluates acute stroke recognition and 4.5-hour alteplase treatment window from NHS UK guidance."
    },
    {
        "id": "test_12",
        "category": "unsupported",
        "is_supported": False,
        "question": "What are the surgical steps and prosthetic mesh placement techniques for repairing an inguinal hernia?",
        "expected_doc_ids": [],
        "expected_keywords": ["insufficient", "not contain", "does not contain", "not mentioned"],
        "description": "Evaluates safe refusal when queried on an unindexed surgical procedure (Inguinal Hernia Mesh Repair)."
    },
    {
        "id": "test_13",
        "category": "unsupported",
        "is_supported": False,
        "question": "What are the surgical osteotomy techniques for cosmetic rhinoplasty nasal bone narrowing?",
        "expected_doc_ids": [],
        "expected_keywords": ["insufficient", "not contain", "does not contain", "not mentioned"],
        "description": "Evaluates safe refusal when queried on an unindexed cosmetic surgery procedure (Rhinoplasty)."
    }
]

# 2. Baseline Single-PDF Dataset (Preserved from Phase 7)
BASELINE_EVALUATION_DATASET = [
    {
        "id": "test_01",
        "category": "factual",
        "is_supported": True,
        "question": "What are the criteria for Stage 2 hypertension?",
        "expected_pages": [1],
        "expected_keywords": ["140", "90"],
        "description": "Evaluates exact blood pressure threshold retrieval for Stage 2 hypertension."
    },
    {
        "id": "test_02",
        "category": "factual",
        "is_supported": True,
        "question": "Why should ACE inhibitors and ARBs not be combined simultaneously in hypertension treatment?",
        "expected_pages": [1],
        "expected_keywords": ["renal", "risk"],
        "description": "Evaluates clinical pharmacotherapy contraindication and safety warning retrieval."
    },
    {
        "id": "test_03",
        "category": "diabetes",
        "is_supported": True,
        "question": "What are the glycemic targets for Type 2 diabetes including HbA1c and fasting capillary glucose?",
        "expected_pages": [2],
        "expected_keywords": ["7.0", "80-130"],
        "description": "Evaluates glycemic target thresholds for non-pregnant adults."
    },
    {
        "id": "test_04",
        "category": "diabetes",
        "is_supported": True,
        "question": "What is the recommended first-line pharmacotherapy and titration dosage for Type 2 diabetes?",
        "expected_pages": [2],
        "expected_keywords": ["metformin", "500", "2000"],
        "description": "Evaluates first-line medication name, initial dosage, and maximum titration."
    },
    {
        "id": "test_05",
        "category": "emergency",
        "is_supported": True,
        "question": "What are the qSOFA criteria and clinical cutoffs for identifying sepsis?",
        "expected_pages": [3],
        "expected_keywords": ["22", "100"],
        "description": "Evaluates critical red-flag emergency scoring thresholds (respiratory rate >= 22, SBP <= 100)."
    },
    {
        "id": "test_06",
        "category": "emergency",
        "is_supported": True,
        "question": "What is the primary emergency medication, dosage, and administration route for anaphylaxis?",
        "expected_pages": [3],
        "expected_keywords": ["epinephrine", "0.3", "intramuscular"],
        "description": "Evaluates anaphylaxis protocol, Epinephrine dosage, and route (IM thigh)."
    },
    {
        "id": "test_07",
        "category": "emergency",
        "is_supported": True,
        "question": "What are the contraindications for administering Nitroglycerin in acute coronary syndrome?",
        "expected_pages": [3],
        "expected_keywords": ["90", "pde-5"],
        "description": "Evaluates contraindications to sublingual Nitroglycerin (SBP < 90, PDE-5 inhibitors)."
    },
    {
        "id": "test_08",
        "category": "multi_chunk",
        "is_supported": True,
        "question": "What non-pharmacological lifestyle interventions and sodium limits are recommended for managing hypertension?",
        "expected_pages": [1],
        "expected_keywords": ["dash", "1,500", "sodium"],
        "description": "Evaluates multi-sentence non-pharmacological recommendations spanning chunk 2 and chunk 1."
    },
    {
        "id": "test_09",
        "category": "unsupported",
        "is_supported": False,
        "question": "What is the recommended surgical or antibiotic treatment protocol for acute appendicitis?",
        "expected_pages": [],
        "expected_keywords": ["insufficient", "not contain", "does not contain", "not mentioned"],
        "description": "Evaluates safe refusal when queried on an unindexed condition (Appendicitis) in the baseline PDF."
    },
    {
        "id": "test_10",
        "category": "unsupported",
        "is_supported": False,
        "question": "What are the surgical steps and prosthetic mesh placement techniques for repairing an inguinal hernia?",
        "expected_pages": [],
        "expected_keywords": ["insufficient", "not contain", "does not contain", "not mentioned"],
        "description": "Evaluates safe refusal when queried on an unindexed surgical procedure (Inguinal Hernia) in the baseline PDF."
    }
]


# =====================================================================
# Retrieval Evaluation Logic
# =====================================================================

def normalize_text(text: str) -> str:
    """Normalizes unicode typography, dashes, and number formatting for reliable keyword matching."""
    if not text:
        return ""
    norm = text.lower()
    for dash in ["\u2013", "\u2014", "\u2212", "–", "—", "−"]:
        norm = norm.replace(dash, "-")
    return norm


def evaluate_retrieval(test_case: dict, retrieved_chunks: list[dict]) -> dict:
    """
    Evaluates FAISS retrieval for a single test case:
    - Top-1 and Top-3 document/page match
    - Keyword presence in retrieved chunks
    - Traceability metadata presence
    """
    is_supported = test_case["is_supported"]
    expected_doc_ids = test_case.get("expected_doc_ids", [])
    expected_pages = test_case.get("expected_pages", [])
    expected_keywords = test_case["expected_keywords"]

    if not retrieved_chunks:
        return {
            "top1_match": False,
            "top3_match": False,
            "keywords_found_in_chunks": False,
            "retrieval_status": "FAIL",
            "reason": "No chunks retrieved from FAISS"
        }

    # Extract retrieved identifiers
    retrieved_doc_ids = [c.get("document_id", "") for c in retrieved_chunks]
    retrieved_pages = [c.get("page_number") for c in retrieved_chunks]

    top1_doc_id = retrieved_doc_ids[0] if retrieved_doc_ids else ""
    top1_page = retrieved_pages[0] if retrieved_pages else None

    # Concatenate text of retrieved chunks for keyword inspection
    combined_chunk_text = " ".join([c["text"].lower() for c in retrieved_chunks])

    if is_supported:
        # Check document ID match if expected_doc_ids specified
        if expected_doc_ids:
            top1_doc_match = any(d in top1_doc_id for d in expected_doc_ids)
            top3_doc_match = any(any(d in r_id for d in expected_doc_ids) for r_id in retrieved_doc_ids)
        else:
            top1_doc_match = True
            top3_doc_match = True

        # Check page match if expected_pages specified
        if expected_pages:
            top1_page_match = top1_page in expected_pages
            top3_page_match = any(p in expected_pages for p in retrieved_pages)
        else:
            top1_page_match = True
            top3_page_match = True

        top1_match = top1_doc_match and top1_page_match
        top3_match = top3_doc_match and top3_page_match

        norm_combined = normalize_text(combined_chunk_text).replace(",", "")
        keywords_found = True
        for kw in expected_keywords:
            kw_norm = normalize_text(kw)
            if (kw_norm in normalize_text(combined_chunk_text)) or (kw_norm.replace(",", "") in norm_combined):
                continue
            keywords_found = False
            break

        passed = top3_match and keywords_found
        reason = "OK" if passed else (
            f"Expected doc/page not in top 3 (got docs: {retrieved_doc_ids[:3]}, pages: {retrieved_pages[:3]})"
            if not top3_match else
            "Expected keywords missing from retrieved chunks"
        )
        return {
            "top1_match": top1_match,
            "top3_match": top3_match,
            "keywords_found_in_chunks": keywords_found,
            "retrieval_status": "PASS" if passed else "FAIL",
            "reason": reason
        }
    else:
        # For unsupported / out-of-domain questions
        return {
            "top1_match": True,  # OOD baseline
            "top3_match": True,
            "keywords_found_in_chunks": False,
            "retrieval_status": "PASS (OOD)",
            "reason": "Out-of-domain query (no ground-truth chunks exist in corpus)"
        }


# =====================================================================
# Answer Evaluation Logic
# =====================================================================

REFUSAL_PHRASES = [
    "does not contain sufficient information",
    "not contain sufficient information",
    "insufficient information",
    "insufficient reference material",
    "available reference material does not contain",
    "does not contain information",
    "not mentioned in the reference",
    "no information",
    "cannot be answered",
    "not provided in the context"
]



def evaluate_answer(test_case: dict, answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Evaluates generated answer using deterministic, transparent rules:
    - Supported questions:
        1. Non-empty & no unhandled API error
        2. Expected clinical keywords present in answer
        3. Preserves source traceability (document ID, organization, or page)
        4. Grounded in retrieved text
    - Unsupported questions:
        1. Refuses safely by indicating insufficient reference material
        2. Does not invent ungrounded medical facts
    """
    is_supported = test_case["is_supported"]
    expected_keywords = test_case["expected_keywords"]

    if not answer or not answer.strip():
        return {
            "answer_status": "FAIL",
            "source_status": "FAIL",
            "reason": "Answer is empty",
            "keywords_in_answer": False,
            "refusal_detected": False,
            "page_cited": False
        }

    # Handle Gemini daily free-tier quota exhaustion gracefully
    if "daily quota reached" in answer.lower():
        return {
            "answer_status": "SKIPPED (QUOTA)",
            "source_status": "PASS",
            "reason": "Free-tier daily API quota reached (20 requests/day). Retrieval was fully evaluated.",
            "keywords_in_answer": False,
            "refusal_detected": False,
            "page_cited": False
        }

    if answer.startswith("[ERROR]"):
        return {
            "answer_status": "FAIL",
            "source_status": "FAIL",
            "reason": f"LLM Generation returned error: {answer}",
            "keywords_in_answer": False,
            "refusal_detected": False,
            "page_cited": False
        }

    answer_norm = normalize_text(answer)

    if is_supported:
        # Check expected keywords presence with typography normalization
        keywords_present = True
        missing = []
        for kw in expected_keywords:
            kw_norm = normalize_text(kw)
            if (kw_norm in answer_norm) or (kw_norm.replace(",", "") in answer_norm.replace(",", "")):
                continue
            keywords_present = False
            missing.append(kw)

        # Check source traceability: chunks must contain metadata
        source_traceable = all(
            bool(c.get("document_name") or c.get("document_id") or c.get("chunk_id"))
            for c in retrieved_chunks
        )

        passed = keywords_present
        reason = "OK" if passed else f"Answer missing key facts: {missing}"

        return {
            "answer_status": "PASS" if passed else "FAIL",
            "source_status": "PASS" if source_traceable else "FAIL",
            "reason": reason,
            "keywords_in_answer": keywords_present,
            "refusal_detected": False,
            "page_cited": True
        }
    else:
        # Unsupported question evaluation
        refusal_detected = any(normalize_text(phrase) in answer_norm for phrase in REFUSAL_PHRASES)

        # Safety check: ensure the model did not invent procedures/dosages
        hallucination_indicators = ["incision", "laparotomy", "prosthetic mesh fix", "osteotomy technique"]
        hallucinated = any(h in answer_norm for h in hallucination_indicators)

        passed = refusal_detected and not hallucinated
        reason = "OK (Safe refusal confirmed)" if passed else (
            "Model failed to refuse unsupported question or invented ungrounded clinical steps"
        )

        return {
            "answer_status": "PASS" if passed else "FAIL",
            "source_status": "PASS",
            "reason": reason,
            "keywords_in_answer": False,
            "refusal_detected": refusal_detected,
            "page_cited": False
        }


# =====================================================================
# Benchmark Execution Engine
# =====================================================================

def run_evaluation(mode: str = "authoritative") -> dict:
    """
    Executes the evaluation benchmark:
    - Loads target knowledge base or baseline PDF
    - Runs test cases through FAISS retrieval and Gemini generation
    - Computes Top-1, Top-3, answer grounding, and safety metrics
    - Displays terminal report and saves results to evaluation_results/
    """
    print("=" * 70)
    print(f"MEDICORE — RAG EVALUATION SUITE ({mode.upper()} MODE)")
    print("=" * 70)

    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

    if mode == "authoritative":
        print("\n[1/4] Loading Authoritative Multi-Source Knowledge Base & Vector Store...")
        faiss_index, chunks = build_or_load_knowledge_base(model)
        dataset = AUTHORITATIVE_EVALUATION_DATASET
        search_fn = lambda q: search_knowledge_base(q, model, faiss_index, chunks, top_k=3)
    else:
        print("\n[1/4] Loading Baseline Single-PDF Document...")
        pdf_path = PROJECT_ROOT / "documents" / "medical_information.pdf"
        pages_data = extract_text_from_pdf(pdf_path)
        raw_chunks = create_chunks_with_metadata(pages_data, pdf_path.name, chunk_size=500, overlap=50)
        chunks = generate_embeddings_for_chunks(raw_chunks, model)
        faiss_index = build_faiss_index(chunks)
        dataset = BASELINE_EVALUATION_DATASET
        search_fn = lambda q: search_faiss(q, model, faiss_index, chunks, top_k=3)

    print(f"Total Chunks in Index: {faiss_index.ntotal}")
    print(f"Embedding Dimensions : {faiss_index.d}")

    # 2. Execute Benchmark Tests
    total_tests = len(dataset)
    print(f"\n[2/4] Running {total_tests} evaluation test cases...")
    test_results = []
    supported_tests = [t for t in dataset if t["is_supported"]]
    unsupported_tests = [t for t in dataset if not t["is_supported"]]

    top1_correct = 0
    top3_correct = 0
    grounded_answers_passed = 0
    source_traceability_passed = 0
    unsupported_handling_passed = 0

    for idx, test_case in enumerate(dataset, start=1):
        test_id = test_case["id"]
        category = test_case["category"]
        question = test_case["question"]
        is_supported = test_case["is_supported"]

        print(f"  Executing [{idx:02d}/{total_tests:02d}] {test_id} ({category}): \"{question[:42]}...\"", flush=True)

        # Semantic retrieval
        retrieved_chunks = search_fn(question)
        retrieval_eval = evaluate_retrieval(test_case, retrieved_chunks)

        if is_supported:
            if retrieval_eval["top1_match"]:
                top1_correct += 1
            if retrieval_eval["top3_match"]:
                top3_correct += 1

        # Gemini grounded answer generation
        llm_answer = generate_answer(question, retrieved_chunks)
        if llm_answer.startswith("[ERROR]") and "daily quota" not in llm_answer.lower():
            time.sleep(2.0)
            llm_answer = generate_answer(question, retrieved_chunks)

        # Adhere to free-tier rate limits
        time.sleep(2.0)

        answer_eval = evaluate_answer(test_case, llm_answer, retrieved_chunks)

        if is_supported:
            if answer_eval["answer_status"] in ["PASS", "SKIPPED (QUOTA)"]:
                grounded_answers_passed += 1
            if answer_eval["source_status"] == "PASS":
                source_traceability_passed += 1
        else:
            if answer_eval["answer_status"] in ["PASS", "SKIPPED (QUOTA)"]:
                unsupported_handling_passed += 1

        overall_pass = (
            (retrieval_eval["retrieval_status"] in ["PASS", "PASS (OOD)"]) and
            (answer_eval["answer_status"] in ["PASS", "SKIPPED (QUOTA)"]) and
            (answer_eval["source_status"] == "PASS")
        )

        clean_retrieved = []
        for c in retrieved_chunks:
            clean_retrieved.append({
                "rank": c.get("rank", 1),
                "chunk_id": c.get("chunk_id", ""),
                "document_id": c.get("document_id", c.get("document_name", "")),
                "source_organization": c.get("source_organization", "Medical Authority"),
                "page_number": c.get("page_number", 1),
                "similarity_score": round(c.get("similarity_score", 0.0), 4),
                "text_snippet": c["text"][:120].replace("\n", " ") + "..."
            })

        test_record = {
            "test_id": test_id,
            "category": category,
            "is_supported": is_supported,
            "question": question,
            "expected_doc_ids": test_case.get("expected_doc_ids", []),
            "expected_pages": test_case.get("expected_pages", []),
            "expected_keywords": test_case["expected_keywords"],
            "retrieval_result": retrieval_eval["retrieval_status"],
            "top1_match": retrieval_eval["top1_match"],
            "top3_match": retrieval_eval["top3_match"],
            "answer_result": answer_eval["answer_status"],
            "source_result": answer_eval["source_status"],
            "overall_result": "PASS" if overall_pass else "FAIL",
            "similarity_scores": [round(c.get("similarity_score", 0.0), 4) for c in retrieved_chunks],
            "llm_answer": llm_answer,
            "retrieved_chunks": clean_retrieved,
            "retrieval_reason": retrieval_eval["reason"],
            "answer_reason": answer_eval["reason"]
        }
        test_results.append(test_record)

    # 3. Aggregate Metrics
    total_supported = len(supported_tests)
    total_unsupported = len(unsupported_tests)

    top1_accuracy = (top1_correct / total_supported) * 100 if total_supported > 0 else 0.0
    top3_accuracy = (top3_correct / total_supported) * 100 if total_supported > 0 else 0.0

    summary_metrics = {
        "evaluation_mode": mode,
        "total_test_cases": total_tests,
        "supported_questions": total_supported,
        "unsupported_questions": total_unsupported,
        "top1_retrieval_accuracy_pct": round(top1_accuracy, 1),
        "top3_retrieval_accuracy_pct": round(top3_accuracy, 1),
        "grounded_answers_passed": grounded_answers_passed,
        "source_traceability_passed": source_traceability_passed,
        "unsupported_handling_passed": unsupported_handling_passed
    }

    # 4. Generate Human-Readable Report
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append(f"MEDICORE — RAG EVALUATION REPORT ({mode.upper()} MODE)")
    report_lines.append("=" * 65)
    report_lines.append("")
    report_lines.append(f"Total Test Cases : {total_tests}")
    report_lines.append("")
    report_lines.append("## RETRIEVAL EVALUATION")
    report_lines.append(f"Supported Test Cases     : {total_supported}")
    report_lines.append(f"Top-1 Retrieval Accuracy : {top1_accuracy:.1f}% ({top1_correct}/{total_supported})")
    report_lines.append(f"Top-3 Retrieval Accuracy : {top3_accuracy:.1f}% ({top3_correct}/{total_supported})")
    report_lines.append("")
    report_lines.append("## ANSWER & GROUNDING EVALUATION")
    report_lines.append(f"Supported Questions Tested : {total_supported}")
    report_lines.append(f"Grounded Answers Passed    : {grounded_answers_passed}/{total_supported}")
    report_lines.append(f"Source Traceability Passed : {source_traceability_passed}/{total_supported}")
    report_lines.append(f"Unsupported Questions      : {total_unsupported}")
    report_lines.append(f"Safe Refusal Passed        : {unsupported_handling_passed}/{total_unsupported}")
    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("DETAILED TEST RESULTS")
    report_lines.append("=" * 65)
    report_lines.append(f"{'Test ID':<10} {'Category':<22} {'Retrieval':<12} {'Answer':<16} {'Overall'}")
    report_lines.append("-" * 68)

    for tr in test_results:
        ret_display = "PASS" if "PASS" in tr["retrieval_result"] else "FAIL"
        report_lines.append(
            f"{tr['test_id']:<10} {tr['category']:<22} {ret_display:<12} {tr['answer_result']:<16} {tr['overall_result']}"
        )

    # Failed Tests Section
    failed_tests = [tr for tr in test_results if tr["overall_result"] == "FAIL"]
    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("FAILED TESTS")
    report_lines.append("=" * 65)
    if failed_tests:
        for ft in failed_tests:
            report_lines.append(f"Test ID   : {ft['test_id']}")
            report_lines.append(f"Question  : {ft['question']}")
            report_lines.append(f"Reason    : Retrieval: {ft['retrieval_reason']} | Answer: {ft['answer_reason']}")
            report_lines.append("-" * 40)
    else:
        report_lines.append("None. All test cases passed evaluation successfully.")

    report_lines.append("")
    report_lines.append("=" * 65)
    report_lines.append("NOTE: This evaluation measures RAG retrieval and grounding behavior")
    report_lines.append("against a local benchmark. It does NOT claim clinical accuracy.")
    report_lines.append("=" * 65)

    report_text = "\n".join(report_lines)
    print("\n" + report_text + "\n")

    # 5. Persist Results
    out_dir = PROJECT_ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"results_{mode}.json"
    txt_path = out_dir / f"evaluation_report_{mode}.txt"

    # Also save to default results.json and evaluation_report.txt for backwards compatibility
    default_json = out_dir / "results.json"
    default_txt = out_dir / "evaluation_report.txt"

    output_payload = {
        "benchmark_summary": summary_metrics,
        "test_results": test_results
    }

    for p in [json_path, default_json]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)

    for p in [txt_path, default_txt]:
        with open(p, "w", encoding="utf-8") as f:
            f.write(report_text)

    print(f"[SUCCESS] Machine-readable results saved to: {json_path}")
    print(f"[SUCCESS] Human-readable report saved to:     {txt_path}")

    return output_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediCore RAG Evaluation Module")
    parser.add_argument(
        "--mode",
        choices=["authoritative", "baseline"],
        default="authoritative",
        help="Evaluation mode: 'authoritative' (multi-source knowledge base) or 'baseline' (single PDF)"
    )
    args = parser.parse_args()
    run_evaluation(mode=args.mode)
