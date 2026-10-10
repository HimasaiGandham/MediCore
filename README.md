# 🔍 MediCore — Hybrid Search (Keyword + Semantic)
> **Branch:** `feature/hybrid-search`  
> **Role in MediCore:** Combining exact keyword search with AI meaning-based search for maximum accuracy.

---

### 🌟 What is this branch all about?
There are two main ways to search a library:
1. **Keyword Search (BM25):** Looks for exact terms like drug names ("Metformin 500mg") or clinical codes ("ICD-10 I10") 🏷️.
2. **Semantic Search (Vectors):** Looks for meaning (so "shortness of breath" matches "dyspnea") 🧬.

This branch brings them together into **Hybrid Search** 🤝 — getting the best of both worlds!

---

### 💡 Why do we need this in MediCore?
- Sometimes a doctor types an **exact medication name** like *"Atorvastatin"* — keyword search finds the exact match instantly.
- Other times a student types a **general symptom** like *"feeling dizzy after standing up"* — semantic search understands the concept.

Hybrid search combines both scores so you never miss a result! 🎯

---

### ⚙️ How does it work in simple words?
1. **Step 1 (Keyword Search):** Searches for exact words using traditional text matching 🔤.
2. **Step 2 (Semantic Search):** Searches by conceptual meaning using vector embeddings 🧠.
3. **Step 3 (Score Fusion):** Combines both scores to rank the most relevant medical passages at the top 🏆.

---

### 🤝 How this branch contributes to MediCore
- 🎯 **Best-in-Class Search Accuracy:** Catches both exact medical jargon and natural-language symptom descriptions.
- 💊 **Precise Drug & Code Retrieval:** Never loses exact numbers, dosages, or acronyms.
- 🔍 **Comprehensive Coverage:** Maximizes evidence retrieval across diverse clinical documents.

---

### 🦄 What makes this branch unique?
This branch bridges **traditional information retrieval (keyword matching) and modern vector search**, providing hybrid search intelligence.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
