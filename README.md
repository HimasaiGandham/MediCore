# 💡 MediCore — Medical Query Understanding & Clarification
> **Branch:** `feature/query-understanding`  
> **Role in MediCore:** Fixing typos, expanding medical acronyms, and refining user questions.

---

### 🌟 What is this branch all about?
When people are stressed or typing quickly, they make typos! They might write *"diabtes"* instead of *"diabetes"* or use clinical shorthand like *"HTN"* for *"hypertension"* 💡. This branch provides **Query Understanding & Rewriting** so MediCore always gets the meaning right.

---

### 💡 Why do we need this in MediCore?
A regular search engine might fail if you misspell a single letter:
- Searching for *"hyper tension"* vs *"hypertension"*
- Searching for *"BP"* vs *"blood pressure"*
- Searching for *"DM Type 2"* vs *"Type 2 Diabetes Mellitus"*

This branch acts like an attentive medical clerk who understands what you mean and cleans up your question before searching! 🩺

---

### ⚙️ How does it work in simple words?
1. **Typo Correction:** Fixes common spelling mistakes in medical terms ✏️.
2. **Acronym Expansion:** Expands medical shorthand (e.g., "MI" &rarr; "Myocardial Infarction / Heart Attack") 🔤.
3. **Optimized Search String:** Passes the clarified question to the FAISS search engine for maximum accuracy 🎯.

---

### 🤝 How this branch contributes to MediCore
- 🎯 **Higher Search Accuracy:** Prevents missed results due to simple typos or abbreviations.
- 😌 **Effortless for Users:** Users don't need to worry about perfect spelling or formal terminology.
- ⚡ **Better Clinical Precision:** Aligns everyday casual questions with formal medical textbook language.

---

### 🦄 What makes this branch unique?
This branch is the **intelligent question parser of MediCore**. It refines and clarifies user queries before any search takes place.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
