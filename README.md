# 📑 MediCore — Clinical Citations & Source Attribution
> **Branch:** `feature/citations`  
> **Role in MediCore:** Providing exact proof, page numbers, and official links for every answer.

---

### 🌟 What is this branch all about?
In medicine, you can never say *"trust me, bro"*! Every medical statement must have **verifiable proof** 🩺📜. This branch is responsible for **Citations & Provenance** — linking every word the AI says back to an official government or health authority document.

---

### 💡 Why do we need this in MediCore?
If an AI tells a student or clinician: *"Give epinephrine 0.5 mg intramuscularly for anaphylaxis,"* the user must be able to verify:
- Who said that? *(WHO Guidelines on Anaphylaxis)*
- Which section? *(Emergency Management Protocol)*
- What page or document? *(WHO-GUIDE-SEP-2023)*
- Where is the official guideline? *(Clickable official link)*

Citations eliminate guesswork and give users total confidence!

---

### ⚙️ How does it work in simple words?
1. **Track the Origin:** Whenever information is retrieved from the library, its source tag is locked to the answer 🏷️.
2. **Build Evidence Cards:** The system creates transparent citation blocks showing:
   - Issuing Authority (e.g., CDC, NHS, WHO, ICMR)
   - Document Title & ID
   - Page & Section Reference
   - Direct web link to the official authority portal 🌐.
3. **Clear Presentation:** In the UI, every answer is accompanied by an expandable source card with exact quotes 🔍.

---

### 🤝 How this branch contributes to MediCore
- 🛡️ **Prevents Medical Hallucinations:** The AI cannot claim something without citing the passage.
- 🎓 **Academic Trust:** Makes MediCore suitable for study, review, and reference.
- 🔍 **Audit Trail:** Every consultation has a transparent source trail.

---

### 🦄 What makes this branch unique?
This branch focuses exclusively on **trust, transparency, and provenance attribution**, turning raw answers into verifiable clinical references.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
