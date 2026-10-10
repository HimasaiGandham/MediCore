# 📥 MediCore — Medical Document Ingestion Pipeline
> **Branch:** `feature/document-ingestion`  
> **Role in MediCore:** Automatically loading and importing official clinical guidance files.

---

### 🌟 What is this branch all about?
MediCore is powered by trusted medical authorities like the **World Health Organization (WHO)**, **CDC**, **NHS**, and **ICMR**. This branch is the **intake pipeline** 🚜 that loads these authoritative files and prepares them for the system.

---

### 💡 Why do we need this in MediCore?
Before an AI can answer questions about tuberculosis, stroke, diabetes, or dengue, the system must have a safe, repeatable way to import new guidelines into its library 📚.

This branch provides the loaders that read PDF manuals and JSON guideline records and registers them into the active knowledge catalog.

---

### ⚙️ How does it work in simple words?
1. **Scan the Directory:** Checks the `documents/authoritative/` folder for clinical JSONs and PDFs 📂.
2. **Read Metadata:** Gathers the issuing organization, title, version, and year 🏷️.
3. **Ingest Content:** Loads every section, symptom list, and emergency protocol into structured Python data objects 📥.
4. **Catalog Registration:** Adds the document to `documents/catalog.json` so the rest of MediCore knows it exists 📋.

---

### 🤝 How this branch contributes to MediCore
- 🏥 **Official Authority Library:** Supplies MediCore with 9+ real-world authoritative guidelines.
- 🔄 **Easy Updates:** Makes it simple to add new medical guidelines in the future.
- 🛡️ **Verified Sources Only:** Ensures that only verified medical files enter the search pipeline.

---

### 🦄 What makes this branch unique?
This branch is the **gateway for all clinical knowledge**. It handles file I/O, format parsing, and dataset registration for all medical resources.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
