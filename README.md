# 🔝 MediCore — Intelligent Evidence Reranking
> **Branch:** `feature/reranking`  
> **Role in MediCore:** Giving search results a second, deeper look to put the best evidence at the very top.

---

### 🌟 What is this branch all about?
When a search engine looks through thousands of paragraphs, it might return 10 possible matches. But which ONE is the most directly helpful? This branch uses **Cross-Encoder Reranking** 🔝 to re-sort the top results so the absolute best answer is always #1!

---

### 💡 Why do we need this in MediCore?
Think of it like a two-stage medical review:
1. **Stage 1 (Fast Search):** A quick assistant grabs 10 relevant medical folders from the archive in 0.05 seconds 🏃‍♂️.
2. **Stage 2 (Reranker):** A senior doctor looks through those 10 folders, compares them closely with the patient's specific question, and puts the most relevant folder right on top 🩺🏆.

---

### ⚙️ How does it work in simple words?
1. **Initial Search:** Fast vector search retrieves the top 5–10 candidate chunks 📦.
2. **Deep Comparison:** A specialized cross-encoder model scores each candidate directly against the user question 🔬.
3. **Re-Order:** Sorts the chunks by relevance score so the most relevant evidence is fed to the answer generator 🥇.

---

### 🤝 How this branch contributes to MediCore
- 🥇 **Pinpoint Precision:** Pushes the most directly relevant guideline text to the top.
- 📉 **Eliminates Clutter:** Drops less-relevant background passages down the list.
- 🎯 **Sharper AI Answers:** The AI answers using the highest-quality evidence available.

---

### 🦄 What makes this branch unique?
This branch acts as the **fine-tuning lens of MediCore's search engine**, upgrading fast initial retrieval with deep secondary relevance scoring.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
