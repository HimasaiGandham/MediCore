# 🧹 MediCore — Document Cleaning & Text Normalization
> **Branch:** `feature/document-cleaning`  
> **Role in MediCore:** Scrubbing messy PDFs and raw files into clean, readable medical text.

---

### 🌟 What is this branch all about?
Raw medical PDFs are full of messy formatting: weird header symbols, page numbers in the middle of sentences, broken words with hyphens, and random line breaks 📄🧹. This branch is the **sanitation crew** that scrubs raw text clean before anything else processes it.

---

### 💡 Why do we need this in MediCore?
If a PDF says:  
`"hyper- \n tension is trea---ted with ACE-inhibitors \x00 12"`,  
a simple search engine might get confused and miss the word `"hypertension"`.

This branch repairs broken words, removes useless symbols, and organizes headers so the AI reads clean, perfect English!

---

### ⚙️ How does it work in simple words?
1. **Extract Raw Text:** Reads words directly from PDF pages and JSON documents 📥.
2. **Fix Broken Words:** Reconnects words split across line breaks (e.g., `hyper-` + `tension` &rarr; `hypertension`) 🩹.
3. **Strip Useless Characters:** Removes non-printable characters, weird formatting artifacts, and extra whitespace 🧼.
4. **Section Parsing:** Identifies headings like "Symptoms," "Treatment," and "Dosage" for clean structure 📑.

---

### 🤝 How this branch contributes to MediCore
- ✨ **High-Quality Search:** Clean text ensures the search engine never misses a medical term.
- 🎯 **Accurate AI Reading:** The AI model never gets confused by garbage characters.
- 🩺 **Standardized Format:** All documents follow the same clean format regardless of who published them.

---

### 🦄 What makes this branch unique?
This branch specializes in **data pre-processing and text hygiene**. It ensures that high-quality input leads to high-quality medical answers.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
