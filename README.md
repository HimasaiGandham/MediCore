# 🧬 MediCore — Medical Vector Embeddings
> **Branch:** `feature/embeddings`  
> **Role in MediCore:** Translating medical words into mathematical meaning fingerprints.

---

### 🌟 What is this branch all about?
Computers do not understand English words like "heart attack" or "blood pressure" — they only understand numbers 🔢. This branch uses **AI Embedding Models** (`sentence-transformers/all-MiniLM-L6-v2`) to turn medical sentences into lists of numbers called **vectors** 🧬.

---

### 💡 Why do we need this in MediCore?
A regular computer search only looks for exact letters:
- If you search for *"high blood pressure,"* a simple search will miss a document that says *"severe hypertension,"* even though they mean the exact same thing! 🤦‍♂️

Embeddings capture the **actual meaning** behind words. In vector space, *"high blood pressure"* and *"hypertension"* sit right next to each other! 🎯

---

### ⚙️ How does it work in simple words?
1. **Read Text Chunk:** Takes a clinical paragraph (e.g., *"Stage 2 hypertension is systolic 140 or higher"*) 📄.
2. **Pass Through AI Transformer:** The sentence transformer model converts the text into a 384-dimensional number array 🧠.
3. **Meaning Fingerprint:** This list of numbers represents the pure medical meaning of that passage, ready to be compared with user questions 📊.

---

### 🤝 How this branch contributes to MediCore
- 🧠 **True Semantic Understanding:** MediCore understands medical concepts, not just keyword spelling.
- ⚡ **Lightweight & Fast:** Uses the efficient `all-MiniLM-L6-v2` model that runs quickly on standard laptops.
- 🔒 **Zero Data Leakage:** Generates embeddings 100% locally on your machine without sending text to external servers.

---

### 🦄 What makes this branch unique?
This branch is the **bridge between human language and mathematical search**. It turns raw medical paragraphs into semantic vector fingerprints.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
