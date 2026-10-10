# ✂️ MediCore — Smart Medical Document Chunking
> **Branch:** `feature/chunking`  
> **Role in MediCore:** Slicing long medical books into bite-sized, readable paragraphs.

---

### 🌟 What is this branch all about?
When a medical guideline is 100 pages long, an AI cannot read the entire book in one glance. This branch specializes in **Chunking** 📄✂️ — neatly cutting long documents into small, manageable pieces (chunks) while keeping their full context intact!

---

### 💡 Why do we need this in MediCore?
Imagine looking for the definition of "Stage 2 Hypertension":
- You don't need the librarian to drop an 800-page book on your head 📚💥.
- You just need the specific 3 sentences on page 14 that explain the blood pressure thresholds 🎯.

Chunking makes sure our search engine finds the exact paragraph you need without losing the surrounding context.

---

### ⚙️ How does it work in simple words?
1. **Sliding Window:** It takes around 500 characters of text per chunk with a 50-character overlap so sentences never get cut awkwardly in half 🪟.
2. **Attaching ID Badges:** Every chunk gets labeled with its document name, page number, section, and health agency 🏷️.
3. **Ready for Search:** These organized pieces are handed over to the search engine so they can be indexed and found instantly 🔍.

---

### 🤝 How this branch contributes to MediCore
- 🎯 **Pinpoint Accuracy:** Allows MediCore to retrieve the exact paragraph answering your question.
- 🔗 **Zero Lost Meaning:** Overlapping chunks ensure no vital medical instruction is cut between pieces.
- 📍 **Page-Level Provenance:** Preserves exact page numbers for every single chunk.

---

### 🦄 What makes this branch unique?
This branch is dedicated exclusively to **text segmentation and metadata labeling**. It solves the foundational problem of how long clinical text is broken down before any AI models touch it.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
