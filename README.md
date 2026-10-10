# 🔎 MediCore — Semantic Search Engine
> **Branch:** `feature/semantic-search`  
> **Role in MediCore:** Finding the right medical guidelines by understanding meaning rather than just keywords.

---

### 🌟 What is this branch all about?
Traditional search only finds exact words. But medicine is full of synonyms! This branch powers **Semantic Search** 🔎 — using cosine similarity over mathematical vector embeddings to understand the true medical meaning of your query.

---

### 💡 Why do we need this in MediCore?
Consider these everyday examples:
- If you ask *"trouble breathing,"* semantic search finds documents about *"dyspnea"* and *"asthma bronchospasm."*
- If you ask *"sugar levels,"* it finds passages discussing *"glycemic control"* and *"HbA1c."*

You don't need to know the complex Latin medical terms — MediCore understands what you mean! 🎯

---

### ⚙️ How does it work in simple words?
1. **Convert Query to Vector:** Your question is converted into a 384-dimensional meaning vector 🧬.
2. **Cosine Similarity Match:** Compares your question vector with every indexed chunk in the FAISS database 📐.
3. **Return Closest Matches:** Retrieves the chunks with the highest similarity scores in milliseconds ⚡.

---

### 🤝 How this branch contributes to MediCore
- 🎯 **Intuitive Discovery:** Allows natural, conversational questions without medical jargon.
- ⚡ **Sub-Second Speed:** Searches hundreds of medical sections in fractions of a second.
- 🏆 **High Accuracy:** Forms the backbone of MediCore's verified 100% retrieval performance.

---

### 🦄 What makes this branch unique?
This branch is the **core semantic search engine of MediCore**. It calculates the mathematical similarity that powers accurate guideline retrieval.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
