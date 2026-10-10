# ⚡ MediCore — Smart Caching & Quick Memory
> **Branch:** `feature/caching`  
> **Role in MediCore:** Making medical searches lightning-fast without repeating heavy work.

---

### 🌟 What is this branch all about?
Medical books and guidelines are huge! Reading hundreds of pages and converting them into mathematical AI vectors takes time and computer power 🧠. This branch gives MediCore a **smart memory (caching)** so it remembers things it has already calculated.

---

### 💡 Why do we need this in MediCore?
Imagine asking a librarian the exact same question ten times:
- **Without caching:** The librarian walks to the basement, climbs a ladder, finds the book, and reads 50 pages every single time 🐢.
- **With caching:** The librarian keeps a sticky note right on their desk! The second time you ask, you get the answer in a split second ⚡.

---

### ⚙️ How does it work in simple words?
1. **Save to Disk:** Once medical guidelines are indexed into the FAISS vector database, they are saved locally in `documents/.index/` 💾.
2. **Instant Reload:** When you open the MediCore web app, it loads the saved index from disk instead of re-reading everything from scratch 🚀.
3. **Keep AI Models Ready:** It keeps the AI embedding model active in computer memory so queries respond immediately ⏱️.

---

### 🤝 How this branch contributes to MediCore
- ⚡ **Super Fast Searches:** Users get answers in milliseconds.
- 💻 **Saves Laptop Memory:** Stops computer freezing and prevents thread allocation errors on Windows.
- 🔋 **Zero Wasted Power:** No unnecessary re-processing of already indexed medical guidelines.

---

### 🦄 What makes this branch unique?
This branch focuses 100% on **speed, performance, and disk persistence**. It runs quietly in the background to supercharge the entire MediCore pipeline without altering clinical content.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
