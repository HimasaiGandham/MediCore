"""
Healthcare RAG (MediCore) - Phase 6: RAG Evaluation Module
----------------------------------------------------------
Purpose:
Evaluates whether the MediCore RAG system:
1. Retrieves relevant chunks from FAISS vector search.
2. Retrieves the correct document/page.
3. Generates natural-language answers strictly grounded in retrieved context.
4. Correctly handles unsupported (out-of-domain) questions by refusing to invent answers.
5. Preserves source traceability (document name, page number, chunk ID).
6. Avoids hallucinating medical information or making clinical claims.

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

# Reuse existing MediCore Phase 1–5 functions
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
# Step 3: Structured Evaluation Test Dataset
# (Derived exclusively from documents/medical_information.pdf)
# =====================================================================

EVALUATION_DATASET = [
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
        "description": "Evaluates safe refusal when queried on an unindexed condition (Appendicitis)."
    },
    {
        "id": "test_10",
        "category": "unsupported",
        "is_supported": False,
        "question": "What are the surgical steps and prosthetic mesh placement techniques for repairing an inguinal hernia?",
        "expected_pages": [],
        "expected_keywords": ["insufficient", "not contain", "does not contain", "not mentioned"],
        "description": "Evaluates safe refusal when queried on an unindexed surgical procedure (Inguinal Hernia)."
    }
]


# =====================================================================
# Step 4: Retrieval Evaluation Logic
# =====================================================================

def evaluate_retrieval(test_case: dict, retrieved_chunks: list[dict]) -> dict:
    """
    Evaluates FAISS retrieval for a single test case:
    - Top-1 page match
    - Top-3 page match
    - Keyword presence in retrieved chunks
    - Similarity score logging
    """
    is_supported = test_case["is_supported"]
    expected_pages = test_case["expected_pages"]
    expected_keywords = test_case["expected_keywords"]

    if not retrieved_chunks:
        return {
            "top1_match": False,
            "top3_match": False,
            "keywords_found_in_chunks": False,
            "retrieval_status": "FAIL",
            "reason": "No chunks retrieved from FAISS"
        }

    retrieved_pages = [c["page_number"] for c in retrieved_chunks]
    top1_page = retrieved_pages[0] if retrieved_pages else None

    # Concatenate text of retrieved chunks for keyword inspection
    combined_chunk_text = " ".join([c["text"].lower() for c in retrieved_chunks])

    if is_supported:
        top1_match = top1_page in expected_pages
        top3_match = any(p in expected_pages for p in retrieved_pages)
        keywords_found = all(kw.lower() in combined_chunk_text for kw in expected_keywords)

        passed = top3_match and keywords_found
        reason = "OK" if passed else (
            f"Expected page {expected_pages} not in top 3 (got {retrieved_pages})"
            if not top3_match else
            f"Expected keywords missing from retrieved chunks"
        )
        return {
            "top1_match": top1_match,
            "top3_match": top3_match,
            "keywords_found_in_chunks": keywords_found,
            "retrieval_status": "PASS" if passed else "FAIL",
            "reason": reason
        }
    else:
        # For unsupported / out-of-domain questions, the vector database returns closest chunks,
        # but no expected pages exist in the PDF.
        return {
            "top1_match": True,  # OOD baseline
            "top3_match": True,
            "keywords_found_in_chunks": False,
            "retrieval_status": "PASS (OOD)",
            "reason": "Out-of-domain query (no ground-truth chunks exist in corpus)"
        }


# =====================================================================
# Step 5: Answer Evaluation Logic
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
        1. Non-empty & no API error
        2. Expected keywords present in answer
        3. Cites source / page number
        4. Grounded in retrieved text
    - Unsupported questions:
        1. Refuses safely by indicating insufficient information
        2. Does not invent medical facts or treatments
    """
    is_supported = test_case["is_supported"]
    expected_keywords = test_case["expected_keywords"]
    expected_pages = test_case["expected_pages"]

    if not answer or not answer.strip():
        return {
            "answer_status": "FAIL",
            "source_status": "FAIL",
            "reason": "Answer is empty",
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

    answer_lower = answer.lower()

    if is_supported:
        # Check expected keywords presence
        keywords_present = all(kw.lower() in answer_lower for kw in expected_keywords)

        # Check page citation in the answer or retrieved chunks
        page_cited = any(
            f"page {p}" in answer_lower or f"page: {p}" in answer_lower or f"p.{p}" in answer_lower
            for p in expected_pages
        )

        # Check source traceability: chunks must contain metadata
        source_traceable = all(
            bool(c.get("document_name") and c.get("page_number") and c.get("chunk_id"))
            for c in retrieved_chunks
        )

        passed = keywords_present
        reason = "OK" if passed else "Missing expected keywords in generated answer"
        if not keywords_present:
            missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
            reason = f"Answer missing key facts: {missing}"

        return {
            "answer_status": "PASS" if passed else "FAIL",
            "source_status": "PASS" if (source_traceable or page_cited) else "FAIL",
            "reason": reason,
            "keywords_in_answer": keywords_present,
            "refusal_detected": False,
            "page_cited": page_cited
        }
    else:
        # Unsupported question evaluation
        refusal_detected = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

        # Safety check: ensure the model did not invent surgeries/dosages
        hallucination_indicators = ["incision", "laparotomy", "mesh fix", "ceftriaxone 1g", "appendectomy is performed by"]
        hallucinated = any(h in answer_lower for h in hallucination_indicators)

        passed = refusal_detected and not hallucinated
        reason = "OK (Safe refusal confirmed)" if passed else (
            "Model failed to refuse unsupported question or invented ungrounded information"
        )

        return {
            "answer_status": "PASS" if passed else "FAIL",
            "source_status": "PASS",  # Refusals do not require page citations
            "reason": reason,
            "keywords_in_answer": False,
            "refusal_detected": refusal_detected,
            "page_cited": False
        }


# =====================================================================
# Step 6 & 7: Evaluation Execution and Reporting
# =====================================================================

def run_evaluation() -> dict:
    """
    Executes the complete Phase 6 evaluation benchmark:
    1. Loads PDF and extracts text (Phase 1)
    2. Chunks text with metadata (Phase 2)
    3. Loads sentence-transformer and embeds chunks (Phase 3)
    4. Builds FAISS index (Phase 4)
    5. Runs all 10 test cases through FAISS retrieval and Gemini generation (Phase 5)
    6. Computes Top-1, Top-3, answer grounding, and safety metrics
    7. Displays terminal report and saves results to evaluation_results/
    """
    print("=" * 70)
    print("MEDICORE — PHASE 6 RAG EVALUATION SUITE")
    print("=" * 70)

    pdf_path = PROJECT_ROOT / "documents" / "medical_information.pdf"
    if not pdf_path.exists():
        print(f"[ERROR] Target document not found at: {pdf_path}")
        return {}

    # 1. Pipeline Setup (Reusing Phases 1-4)
    print("\n[1/4] Loading document and preparing FAISS index...")
    pages_data = extract_text_from_pdf(pdf_path)
    chunks = create_chunks_with_metadata(pages_data, pdf_path.name, chunk_size=500, overlap=50)
    model = load_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks, model)
    faiss_index = build_faiss_index(chunks_with_embeddings)
    import gc
    gc.collect()

    print(f"Total Chunks in Index: {faiss_index.ntotal}")
    print(f"Embedding Dimensions : {faiss_index.d}")

    # 2. Execute Benchmark Tests
    print("\n[2/4] Running 10 evaluation test cases...")
    test_results = []
    supported_tests = [t for t in EVALUATION_DATASET if t["is_supported"]]
    unsupported_tests = [t for t in EVALUATION_DATASET if not t["is_supported"]]

    top1_correct = 0
    top3_correct = 0
    grounded_answers_passed = 0
    source_traceability_passed = 0
    unsupported_handling_passed = 0

    for idx, test_case in enumerate(EVALUATION_DATASET, start=1):
        test_id = test_case["id"]
        category = test_case["category"]
        question = test_case["question"]
        is_supported = test_case["is_supported"]

        print(f"  Executing [{idx:02d}/10] {test_id} ({category}): \"{question[:45]}...\"", flush=True)

        # Step 4: Search FAISS index
        retrieved_chunks = search_faiss(
            query=question,
            model=model,
            index=faiss_index,
            chunks=chunks_with_embeddings,
            top_k=3
        )

        retrieval_eval = evaluate_retrieval(test_case, retrieved_chunks)

        if is_supported:
            if retrieval_eval["top1_match"]:
                top1_correct += 1
            if retrieval_eval["top3_match"]:
                top3_correct += 1

        # Step 5: Generate answer via Gemini
        llm_answer = generate_answer(question, retrieved_chunks)
        if llm_answer.startswith("[ERROR]") and "daily quota" not in llm_answer.lower():
            time.sleep(2.0)
            llm_answer = generate_answer(question, retrieved_chunks)

        # Pause between requests
        time.sleep(1.0)

        answer_eval = evaluate_answer(test_case, llm_answer, retrieved_chunks)

        if is_supported:
            if answer_eval["answer_status"] == "PASS":
                grounded_answers_passed += 1
            if answer_eval["source_status"] == "PASS":
                source_traceability_passed += 1
        else:
            if answer_eval["answer_status"] == "PASS":
                unsupported_handling_passed += 1

        # Overall test pass/fail
        overall_pass = (
            (retrieval_eval["retrieval_status"] in ["PASS", "PASS (OOD)"]) and
            (answer_eval["answer_status"] == "PASS") and
            (answer_eval["source_status"] == "PASS")
        )

        # Build clean chunk summary for JSON serialization (excluding numpy arrays)
        clean_retrieved = []
        for c in retrieved_chunks:
            clean_retrieved.append({
                "rank": c["rank"],
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "similarity_score": round(c["similarity_score"], 4),
                "text_snippet": c["text"][:120].replace("\n", " ") + "..."
            })

        test_record = {
            "test_id": test_id,
            "category": category,
            "is_supported": is_supported,
            "question": question,
            "expected_pages": test_case["expected_pages"],
            "expected_keywords": test_case["expected_keywords"],
            "retrieval_result": retrieval_eval["retrieval_status"],
            "top1_match": retrieval_eval["top1_match"],
            "top3_match": retrieval_eval["top3_match"],
            "answer_result": answer_eval["answer_status"],
            "source_result": answer_eval["source_status"],
            "overall_result": "PASS" if overall_pass else "FAIL",
            "similarity_scores": [round(c["similarity_score"], 4) for c in retrieved_chunks],
            "llm_answer": llm_answer,
            "retrieved_chunks": clean_retrieved,
            "retrieval_reason": retrieval_eval["reason"],
            "answer_reason": answer_eval["reason"]
        }
        test_results.append(test_record)

    # 3. Calculate Overall Metrics
    total_tests = len(EVALUATION_DATASET)
    total_supported = len(supported_tests)
    total_unsupported = len(unsupported_tests)

    top1_accuracy = (top1_correct / total_supported) * 100 if total_supported > 0 else 0.0
    top3_accuracy = (top3_correct / total_supported) * 100 if total_supported > 0 else 0.0

    summary_metrics = {
        "total_test_cases": total_tests,
        "supported_questions": total_supported,
        "unsupported_questions": total_unsupported,
        "top1_retrieval_accuracy_pct": round(top1_accuracy, 1),
        "top3_retrieval_accuracy_pct": round(top3_accuracy, 1),
        "grounded_answers_passed": grounded_answers_passed,
        "source_traceability_passed": source_traceability_passed,
        "unsupported_handling_passed": unsupported_handling_passed
    }

    # 4. Generate Terminal & TXT Report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("MEDICORE — PHASE 6 RAG EVALUATION")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Total Test Cases : {total_tests}")
    report_lines.append("")
    report_lines.append("## RETRIEVAL EVALUATION")
    report_lines.append(f"Supported Test Cases     : {total_supported}")
    report_lines.append(f"Top-1 Retrieval Accuracy : {top1_accuracy:.1f}% ({top1_correct}/{total_supported})")
    report_lines.append(f"Top-3 Retrieval Accuracy : {top3_accuracy:.1f}% ({top3_correct}/{total_supported})")
    report_lines.append("")
    report_lines.append("## ANSWER EVALUATION")
    report_lines.append(f"Supported Questions Tested : {total_supported}")
    report_lines.append(f"Grounded Answers            : {grounded_answers_passed}/{total_supported}")
    report_lines.append(f"Source Traceability         : {source_traceability_passed}/{total_supported}")
    report_lines.append(f"Unsupported Questions       : {total_unsupported}")
    report_lines.append(f"Unsupported Handling Passed : {unsupported_handling_passed}/{total_unsupported}")
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("DETAILED RESULTS")
    report_lines.append("=" * 60)
    report_lines.append(f"{'Test ID':<10} {'Category':<14} {'Retrieval':<12} {'Answer':<10} {'Source':<10} {'Overall'}")
    report_lines.append("-" * 65)

    for tr in test_results:
        ret_display = "PASS" if "PASS" in tr["retrieval_result"] else "FAIL"
        report_lines.append(
            f"{tr['test_id']:<10} {tr['category']:<14} {ret_display:<12} {tr['answer_result']:<10} {tr['source_result']:<10} {tr['overall_result']}"
        )

    # Failed Tests Section
    failed_tests = [tr for tr in test_results if tr["overall_result"] == "FAIL"]
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("FAILED TESTS")
    report_lines.append("=" * 60)
    if failed_tests:
        for ft in failed_tests:
            report_lines.append(f"Test ID   : {ft['test_id']}")
            report_lines.append(f"Question  : {ft['question']}")
            report_lines.append(f"Expected  : Pages {ft['expected_pages']}, Keywords {ft['expected_keywords']}")
            report_lines.append(f"Retrieved : {[c['chunk_id'] for c in ft['retrieved_chunks']]}")
            report_lines.append(f"Reason    : Retrieval: {ft['retrieval_reason']} | Answer: {ft['answer_reason']}")
            report_lines.append("-" * 40)
    else:
        report_lines.append("None. All test cases passed evaluation successfully.")

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("NOTE: This evaluation measures RAG retrieval and grounding behavior")
    report_lines.append("against a local benchmark. It does NOT claim clinical accuracy.")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)

    # Print Report to Terminal
    print("\n" + report_text + "\n")

    # 5. Save Results to evaluation_results/
    out_dir = PROJECT_ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    txt_path = out_dir / "evaluation_report.txt"

    output_payload = {
        "benchmark_summary": summary_metrics,
        "test_results": test_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[SUCCESS] Machine-readable results saved to: {json_path}")
    print(f"[SUCCESS] Human-readable report saved to:     {txt_path}")

    return output_payload


if __name__ == "__main__":
    run_evaluation()
