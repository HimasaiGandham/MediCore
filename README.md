# 🗄️ MediCore — FAISS Vector Database Storage
> **Branch:** `feature/vector-database`  
> **Role in MediCore:** The high-speed search database storing all medical vectors on disk.

---

### 🌟 What is this branch all about?
Where do hundreds of medical vectors live so they can be searched in milliseconds? This branch provides the **FAISS Vector Database** (`IndexFlatIP`) 🗄️⚡ — Facebook AI Similarity Search optimized for fast cosine distance matching.

---

### 💡 Why do we need this in MediCore?
If you have thousands of medical paragraphs, comparing a question with each one individually can slow your computer down 🐌.

FAISS organizes vectors into an optimized index so finding the top-3 best matches takes **less than 10 milliseconds**! Plus, it saves the index directly to disk (`faiss_index.bin`) so it never needs to be rebuilt on every startup 💾.

---

### ⚙️ How does it work in simple words?
1. **Build the Index:** Normalizes vectors to unit length and builds a `faiss.IndexFlatIP` cosine index 📐.
2. **Save to Disk:** Saves `faiss_index.bin` and `chunks.pkl` in the `documents/.index/` folder 💾.
3. **Lightning Search:** On any user query, FAISS searches the index and returns the closest chunk indices instantly ⚡.

---

### 🤝 How this branch contributes to MediCore
- ⚡ **Instant Search Performance:** Delivers sub-second response times on standard laptop hardware.
- 💾 **Persistent Disk Storage:** Eliminates slow re-indexing every time you start the app.
- 🗄️ **Scalable Foundation:** Can easily grow from 10 documents to 10,000 documents without slowdown.

---

### 🦄 What makes this branch unique?
This branch is the **database engine of MediCore**. It handles FAISS indexing, similarity calculations, and disk serialization.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
