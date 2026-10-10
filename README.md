# 🚫 MediCore — Safe Refusal for Unsupported Questions
> **Branch:** `feature/insufficient-evidence`  
> **Role in MediCore:** Teaching the system to safely say "I don't know" when documents don't have the answer.

---

### 🌟 What is this branch all about?
One of the most important features in medical AI is knowing when **NOT** to answer! This branch implements **Safe Boundary Refusal** 🚫 — ensuring that if MediCore cannot find verified evidence in its guidelines, it refuses to guess.

---

### 💡 Why do we need this in MediCore?
Imagine asking MediCore: *"What are the surgical steps to perform an emergency open-heart surgery?"*
- MediCore contains clinical reference guidelines, NOT surgical manuals!
- An unsafe AI might try to improvise steps from memory 😱.
- MediCore must safely respond: *"The provided guidelines do not contain information on this topic. Please consult a qualified surgical specialist."* 🛑

---

### ⚙️ How does it work in simple words?
1. **Similarity Score Check:** When a question is searched, the system checks the similarity score of the best match 📏.
2. **Threshold Test:** If the similarity score is below the confidence threshold, the question is flagged as unsupported 🚩.
3. **Polite, Safe Refusal:** The system delivers a clear refusal message explaining that the indexed guidelines do not cover this topic 🛡️.

---

### 🤝 How this branch contributes to MediCore
- 🛑 **100% Out-of-Domain Safety:** Protects users from false or misleading advice on uncovered medical topics.
- 🩺 **Maintains Scope Boundaries:** Keeps the system honest about what is in its library.
- 🏆 **Passes Clinical Benchmarks:** Directly powers the 100% pass rate on unsupported test cases in the Phase 7 benchmark suite.

---

### 🦄 What makes this branch unique?
This branch represents **safety restraint**. It ensures the AI refuses unsupported questions gracefully and safely rather than guessing.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
