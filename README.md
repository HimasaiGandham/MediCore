# 📊 MediCore — Clinical RAG Evaluation Suite (Phase 7)
> **Branch:** `feature/evaluation`  
> **Role in MediCore:** Automated testing lab verifying search accuracy and safety boundaries.

---

### 🌟 What is this branch all about?
How do we prove that MediCore actually works accurately? This branch contains the **Phase 7 Clinical Evaluation Suite** (`src/evaluation.py`) 🧪 — a rigorous testing suite that measures search precision, keyword recall, source tracking, and out-of-domain refusal.

---

### 💡 Why do we need this in MediCore?
In healthcare, you must test before you trust! We need to mathematically verify:
- Does the system find the exact right document for Stage 2 hypertension? (Yes! 100% Top-1 accuracy) 🎯
- Does it safely refuse to answer when asked about unindexed surgical procedures? (Yes! Safe refusal) 🛑
- Can it run tests locally with zero API costs? (Yes! `--retrieval-only` runs 100% offline) ⚡

---

### ⚙️ How does it work in simple words?
1. **Curated Benchmark Cases:** Includes 13 curated clinical test cases (11 supported queries, 2 unsupported out-of-domain queries) 📋.
2. **Zero-API Retrieval Mode:** Runs tests locally against the FAISS index without calling any paid external APIs 🆓.
3. **Detailed Scorecard:** Generates both a machine-readable JSON report (`results.json`) and an easy-to-read text report (`evaluation_report.txt`) 📄.

---

### 🤝 How this branch contributes to MediCore
- 🏆 **Verified 100% Retrieval Accuracy:** 100% Top-1 precision and 100% Top-3 recall on the benchmark dataset.
- 🚫 **Verified Refusal Safety:** 100% safe refusal on out-of-domain questions.
- 📑 **Comprehensive Audit Reports:** Keeps permanent records of system accuracy in `evaluation_results/`.

---

### 🦄 What makes this branch unique?
This branch is the **official benchmark and quality assurance engine** of MediCore. It provides verifiable proof of search precision and safety.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
